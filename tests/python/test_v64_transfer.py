import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from prepare_v64_transfer import UniquePool,view_keys,genspam_messages,FAST_EMAIL
import random,re
from evaluate_v64_transfer import add_record
from collections import defaultdict,Counter

def row(name,label):
    return dict(identity=name,family=name,y=label,source='fixture',script='latin',origin=name,
                focus='subject test hosts  start body',context='subject test body body tail body')

class FirstUseProtocol(unittest.TestCase):
    def test_email_normalizer_preserves_frozen_matches(self):
        frozen=re.compile(r'[\w.+-]+@[\w.-]+\.[\w-]+',re.U)
        cases=['a@b.com','甲'*4000,'a@b.com '+'甲'*4000,'foo+bar@example.org','a@foo.x@b.com',
               'a@b.com+foo@bar.org','测试@域.名+甲@乙.丙']
        rng=random.Random(64)
        cases+=[''.join(rng.choice('ab甲_+-..@@\n<>:/012 ') for _ in range(rng.randint(0,160))) for _ in range(1000)]
        for text in cases:self.assertEqual(frozen.sub('X',text),FAST_EMAIL.sub('X',text))
    def test_conflicting_labels_invalidate_retained_mail(self):
        with tempfile.TemporaryDirectory() as d:
            pool=UniquePool(Path(d)/'db');pool.add(row('a',0),['body:shared']);pool.add(row('b',1),['body:shared'])
            self.assertEqual(pool.count(),0);pool.close()

    def test_bridge_to_previous_mail_invalidates_candidate_and_variants(self):
        with tempfile.TemporaryDirectory() as d:
            pool=UniquePool(Path(d)/'db',{'old:key'});pool.add(row('a',0),['new:key'])
            pool.add(row('b',0),['new:key','old:key']);pool.add(row('c',0),['new:key','variant:key'])
            self.assertEqual(pool.count(),0);pool.close()

    def test_bridge_between_two_retained_messages_invalidates_both(self):
        with tempfile.TemporaryDirectory() as d:
            pool=UniquePool(Path(d)/'db');pool.add(row('a',1),['key:a']);pool.add(row('b',1),['key:b'])
            pool.add(row('c',1),['key:a','key:b']);self.assertEqual(pool.count(),0);pool.close()

    def test_view_fingerprint_ignores_changed_subject_and_hosts(self):
        a=row('a',0);b=dict(a,focus='subject changed hosts other start body',context='subject changed body body tail body')
        self.assertEqual(view_keys(a),view_keys(b))

    def test_gen_parser_uses_published_text_fields_only(self):
        text='<MESSAGE><FROM>metadata</FROM><SUBJECT><TEXT_NORMAL>Subject &amp; name</TEXT_NORMAL></SUBJECT><MESSAGE_BODY><TEXT_NORMAL>Body one</TEXT_NORMAL><TEXT_NORMAL>Body two</TEXT_NORMAL></MESSAGE_BODY></MESSAGE>'
        self.assertEqual(list(genspam_messages(text)),[('Subject & name','Body one\nBody two')])

    def test_pair_counts_compare_matching_frozen_profiles(self):
        tally=defaultdict(Counter);changes=defaultdict(Counter)
        r=dict(label=0,cohort='independent-source',source='fixture',script='latin',
            v62ConservativeMargin=-1,v62HighRecallMargin=1,v64ConservativeMargin=-1,v64HighRecallMargin=-1)
        add_record(tally,changes,r)
        self.assertEqual(tally['all','v64','high-recall']['tn'],1)
        self.assertEqual(tally['all','v62','high-recall']['fp'],1)
        self.assertEqual(changes['all','high-recall']['fixedFP'],1)
        self.assertNotIn(('all','conservative'),changes)

if __name__=='__main__':unittest.main()
