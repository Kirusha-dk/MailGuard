import sys
import unittest
from pathlib import Path
from collections import defaultdict,Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from evaluate_v64_frozen import add_result


class FrozenEvaluationProtocol(unittest.TestCase):
    def test_development_rows_are_not_counted_as_holdout(self):
        tally=defaultdict(Counter)
        r=dict(dataset='500k',scope='development-overlap',source='source',script='latin',label=1,conservativeMargin=1.,highRecallMargin=1.)
        add_result(tally,r)
        self.assertEqual(tally['500k','all','high-recall']['tp'],1)
        self.assertNotIn(('500k','no-development-overlap','high-recall'),tally)

    def test_ham_errors_are_counted_for_the_matching_profile(self):
        tally=defaultdict(Counter)
        r=dict(dataset='ham',scope='additional',source='source',script='latin',label=0,conservativeMargin=-1.,highRecallMargin=.1)
        add_result(tally,r)
        self.assertEqual(tally['ham','all','conservative']['tn'],1)
        self.assertEqual(tally['ham','all','high-recall']['fp'],1)


if __name__=='__main__':unittest.main()
