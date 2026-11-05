"""Engineering fixtures only; no quality claims from these invented rows."""
import copy
import datetime as dt
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from v84_replay import begin_session, check_request, confusion, make_order, new_progress, run_steps
from v84_replay_data import Pool, EmailSubstitution


def request():
    return dict(campaign='v84-en-replay-20261105',start_date='2026-11-05',end_date='2027-05-04',
                session_id='2026-11-05',budget_rows=50000,budget_steps=3125,expected_min_step=8500)


def rows(n=48):
    return [dict(identity=str(i),family=str(i),text=str(i),source='engineering-fixture',
                 y=i%2,hard=i in (3,7),parent_exposures=2 if i%3==0 else 0) for i in range(n)]


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.fixed=nn.Linear(2,3);self.fixed.requires_grad_(False)
        self.dropout=nn.Dropout(.1);self.head=nn.Linear(3,1)
    def forward(self,x):return self.head(self.dropout(self.fixed(x))).squeeze(-1)


def tokenize(tokenizer,texts,length):
    return dict(x=torch.tensor([[int(t)/50.,1.] for t in texts]))


def state(rs):
    parent={'progress':{'last_report':{'development':{'fp':10,'log_loss':.2}}}}
    p=new_progress(rs,parent,.001);begin_session(p,request());return p


class ReplayTest(unittest.TestCase):
    def test_email_fast_path_preserves_normalization(self):
        import v77_mail
        optimized=EmailSubstitution(v77_mail.ADDRESS)
        for value in ('a'*10000,'already emailtoken normalized','a+b@example.com',
                      'a+b@one.com+z@two.co.uk','text name@sub.example.org tail'):
            self.assertEqual(optimized.sub(' emailtoken ',value),v77_mail.ADDRESS.sub(' emailtoken ',value))

    def test_date_window_future_and_budget(self):
        check_request(request(),dt.date(2026,11,5))
        for update in ({'session_id':'2026-11-06'},{'budget_rows':100000},{'expected_min_step':0}):
            with self.assertRaises(ValueError):check_request(request()|update,dt.date(2026,11,5))
        with self.assertRaises(ValueError):check_request(request(),dt.date(2027,5,5))

    def test_resume_keeps_original_session_target(self):
        p=state(rows())
        with self.assertRaises(ValueError):begin_session(p,request()|{'budget_rows':10000})
        with self.assertRaises(ValueError):begin_session(p,request()|{'session_id':'2026-11-06'})
        p['completed_sessions'].append('2026-11-05');self.assertFalse(begin_session(p,request()))

    def test_known_errors_first_and_exhausted_rows_excluded(self):
        rs=rows();p=state(rs);p['exposures'][3]=3
        order=make_order(rs,p).tolist();self.assertNotIn(3,order);self.assertEqual(order[0],7)

    def test_exact_optimizer_scheduler_rng_cursor_resume(self):
        torch.set_num_threads(1);rs=rows();torch.manual_seed(123)
        model=Tiny();original=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
        def fresh():
            m=Tiny();m.load_state_dict(original)
            o=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
            s=torch.optim.lr_scheduler.LambdaLR(o,lambda k:max(0.,1-k/100))
            return m,o,s,state(rs)
        m,o,s,p=fresh();torch.set_rng_state(rng)
        run_steps(m,None,rs,[1.]*len(rs),o,s,p,tokenize,invocation_limit=4)
        expected=copy.deepcopy(m.state_dict());expected_rng=torch.get_rng_state().clone()
        m2,o2,s2,p2=fresh();torch.set_rng_state(rng)
        self.assertFalse(run_steps(m2,None,rs,[1.]*len(rs),o2,s2,p2,tokenize,invocation_limit=2))
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'state.pt';torch.save(dict(model=m2.state_dict(),optimizer=o2.state_dict(),
                 scheduler=s2.state_dict(),progress=p2,rng=torch.get_rng_state()),f)
            cp=torch.load(f,weights_only=True);m3,o3,s3,_=fresh()
            m3.load_state_dict(cp['model']);o3.load_state_dict(cp['optimizer']);s3.load_state_dict(cp['scheduler'])
            torch.set_rng_state(cp['rng']);p3=cp['progress']
            run_steps(m3,None,rs,[1.]*len(rs),o3,s3,p3,tokenize,invocation_limit=2)
        for k,v in expected.items():self.assertTrue(torch.equal(v,m3.state_dict()[k]),k)
        self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()))
        self.assertEqual(p['cursor'],p3['cursor']);self.assertEqual(p['step'],p3['step'])
        self.assertTrue(torch.equal(p['exposures'],p3['exposures']))
        self.assertEqual(p3['active_rows'],48);self.assertEqual(len(set(p3['active_seen'])),48)
        self.assertLessEqual(p3['exposures'].max().item(),3)
        self.assertTrue(torch.equal(original['fixed.weight'],m3.fixed.weight))

    def test_true_50000_unique_rows_without_exceeding_step_target(self):
        # Exercise scheduling at scale without running a neural model50000 times.
        rs=rows(50010);p=state(rs);order=make_order(rs,p).tolist()[:50000]
        self.assertEqual(len(set(order)),50000)
        self.assertEqual((len(order)+15)//16,3125)

    def test_selected_family_and_cross_label_conflicts_are_excluded(self):
        select=[dict(identity='held',family='family-held',text='unrelated selection example')]
        pool=Pool(select)
        base=dict(text='real-source-placeholder',source='fixture',hard=False,parent_exposures=0)
        pool.add(base|dict(identity='a',family='family-held',y=0))
        pool.add(base|dict(identity='b',family='conflict',y=0))
        pool.add(base|dict(identity='c',family='conflict',y=1))
        self.assertEqual(pool.export(),[])

    def test_ham_denominator_and_no_ham_is_not_zero_fpr(self):
        m=confusion([0,0,1,1],[.1,.9,.1,.9],.5)
        self.assertEqual((m['tp'],m['fp'],m['fn'],m['tn']),(1,1,1,1))
        self.assertEqual(m['fpr'],.5)
        self.assertIsNone(confusion([1],[.9],.5)['fpr'])


if __name__=='__main__':unittest.main()
