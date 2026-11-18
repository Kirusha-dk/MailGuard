import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from compare_v82_v80 import compare, grouped


class PairedV80RegressionTests(unittest.TestCase):
    def test_equal_tp_totals_do_not_pass_individual_preservation(self):
        rows = [dict(identity='a',y=1),dict(identity='b',y=1),dict(identity='c',y=0)]
        r = compare(rows,[1,0,1],[0,1,0],.5,.5)
        self.assertEqual(r['baseline']['tp'],r['current']['tp'])
        self.assertEqual(r['baseline_spam_detections_lost'],1)
        self.assertFalse(r['passes_fp_and_individual_tp_gate'])
        self.assertTrue(r['is_a_v80_regression_check'])
        self.assertFalse(r['compares_two_v82_development_checkpoints'])

    def test_group_counts_reconcile_without_serializing_text(self):
        rows = [dict(identity=str(i),y=i%2,source=str(i//2),text='PRIVATE') for i in range(6)]
        a,b = [.1,.9,.9,.9,.1,.1],[.1,.9,.1,.9,.1,.9]
        total = compare(rows,a,b,.5,.5)
        groups = grouped(rows,a,b,.5,.5,lambda r:r['source'])
        for key in ('baseline_spam_detections_lost','spam_detections_gained','false_positives_removed','false_positives_added'):
            self.assertEqual(total[key],sum(g[key] for g in groups.values()))
        self.assertTrue(total['passes_fp_and_individual_tp_gate'])
        self.assertNotIn('PRIVATE',json.dumps(groups))


if __name__ == '__main__':
    unittest.main()
