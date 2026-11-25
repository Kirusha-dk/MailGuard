import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from v86_quality_guard import should_stop, choose_recovery, experiment_tag

class QualityGuardTest(unittest.TestCase):
    def test_epoch_zero_is_not_exempt_from_patience(self):
        self.assertTrue(should_stop(dict(epoch=0,stalled=3)))
        self.assertFalse(should_stop(dict(epoch=0,stalled=2)))
        self.assertTrue(should_stop(dict(epoch=3,stalled=0)))
    def test_zero_gradient_stop_has_a_separate_immutable_archive(self):
        self.assertEqual(experiment_tag(dict(session='2026-11-24')),'v86-experiment-2026-11-24')
        r=dict(session='2026-11-24',quality_guard_applied_without_gradients=True,stopped=True,invocation_optimizer_steps=0)
        self.assertEqual(experiment_tag(r),'v86-experiment-2026-11-24-quality-stop')
        with self.assertRaises(ValueError):experiment_tag(r|dict(invocation_optimizer_steps=1))

    def test_recall_fit_cannot_hide_reduced_fixed_detection(self):
        p=dict(n=22790,spam=10589,ham=12201,fp=548,recall=.9800736613466805,
               threshold=.0594,log_loss=.099,is_development=True,is_independent_test=False,
               fixed_parent_threshold=dict(tp=10394,fp=579,threshold=.04897100105881691))
        c=copy.deepcopy(p);c.update(fp=500,log_loss=.09);c['fixed_parent_threshold']['tp']=10000
        self.assertFalse(choose_recovery(p,c)['repair_passed'])
        c['fixed_parent_threshold']['tp']=10400
        c['fixed_parent_threshold']['fp']=580
        self.assertFalse(choose_recovery(p,c)['repair_passed'])
        c['fixed_parent_threshold']['fp']=579
        self.assertTrue(choose_recovery(p,c)['repair_passed'])
        c['is_independent_test']=True
        with self.assertRaises(ValueError):choose_recovery(p,c)
if __name__=='__main__':unittest.main()
