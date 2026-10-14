import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import numpy as np
from v65_domain_rescue import split_family,decisions,fpr_zero_cutoff,tally_one,metrics,PROFILES

class DomainRescueTests(unittest.TestCase):
    def test_family_isolation_independent_of_label_and_source_metadata(self):
        a=dict(source='GenSpam',family='deadbeef'+'0'*12,y=0)
        b=dict(a,y=1)
        self.assertEqual(split_family(a),split_family(b))

    def test_every_archive_2013_and_2015_is_holdout(self):
        for source in ('SpamArchive-2013','SpamArchive-2015','Nazario-2025'):
            for i in range(50):
                self.assertEqual(split_family(dict(source=source,family=f'{i:016x}')),'holdout')

    def test_calibration_requires_zero_additional_ham_fp(self):
        rows=[dict(y=0,script='latin',v64ConservativeMargin=-1.) for _ in range(600)]
        scores=np.linspace(-1.,2.,600)
        cutoff=fpr_zero_cutoff(rows,scores)
        self.assertTrue(np.all(scores<cutoff))
        self.assertTrue(cutoff>scores.max())

    def test_existing_positive_never_lost(self):
        self.assertEqual(decisions(dict(script='latin',v64ConservativeMargin=.1,v64HighRecallMargin=1),-9,5),
                         (True,True,True))
        self.assertEqual(decisions(dict(script='latin',v64ConservativeMargin=-.1,v64HighRecallMargin=-1),8,5),
                         (False,True,False))
        self.assertEqual(decisions(dict(script='cjk',v64ConservativeMargin=-.1,v64HighRecallMargin=-1),8,5),
                         (False,False,False))

    def test_metrics_account_for_false_positives(self):
        from collections import defaultdict,Counter
        x=defaultdict(Counter)
        tally_one(x,{'y':0},(False,True,False))
        tally_one(x,{'y':1},(True,True,False))
        self.assertEqual(metrics(x['v65-conservative-rescue'])['fp'],1)
        self.assertEqual(metrics(x['v64-conservative'])['recall'],1.)
        self.assertEqual(tuple(x),PROFILES)

if __name__=='__main__':unittest.main()
