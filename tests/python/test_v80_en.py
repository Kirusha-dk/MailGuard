import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from train_v80_en import upper_fpr, recall_cutoff, choose
from v80_encoder import window, Encoder
from v80_mail import needs_neural
from v80_metrics import enrich, summary_extra


class V80Tests(unittest.TestCase):
    def test_zero_fp_small_sample_does_not_certify_ten_ppm(self):
        self.assertGreater(upper_fpr(0, 53303), .00001)
        self.assertLessEqual(upper_fpr(0, 300000), .00001)
        self.assertAlmostEqual(upper_fpr(0, 53303), 1 - math.pow(.05, 1 / 53303))
        self.assertIsNone(upper_fpr(0, 0))

    def test_recall_cutoff_counts_ties_and_rejects_nan(self):
        rows = [{'y': 1}] * 100 + [{'y': 0}] * 1000
        values = np.array([-2.] * 2 + [1.] * 98 + [0.] * 1000)
        cut = recall_cutoff(values, rows)
        self.assertEqual(cut, 1.)
        self.assertEqual(np.count_nonzero(values[:100] >= cut), 98)
        with self.assertRaises(ValueError):
            recall_cutoff(np.full(1100, np.nan), rows)

    def test_fp_limited_choice_protects_worst_source(self):
        rows = ([{'y': 0, 'source': 'ham'}] * 1000 +
                [{'y': 1, 'source': 'one'}] * 100 + [{'y': 1, 'source': 'two'}] * 25)
        narrow = np.array([0.] * 1000 + [1.] * 100 + [-1.] * 25)
        broad = np.array([0.] * 1000 + [1.] * 75 + [-1.] * 25 + [1.] * 25)
        self.assertEqual(choose({'narrow': narrow, 'broad': broad}, rows, .00001)['candidate'], 'broad')

    def test_encoder_keeps_tail_and_forbids_cross_message_quantization(self):
        self.assertEqual(window('short English message'), 'short English message')
        long = 'prefix ' * 400 + 'tail information'
        self.assertTrue(window(long)[1].endswith('tail information'))
        uninitialized = Encoder.__new__(Encoder)
        with self.assertRaisesRegex(ValueError, 'batch_size=1'):
            uninitialized.encode(['one', 'two'], batch_size=2)

    def test_fast_neural_gate_is_fixed_and_label_free(self):
        x = np.array([[-1.] * 5, [-1., -.5, -1., -1., -1.], [1.] * 5])
        self.assertEqual(needs_neural(x).tolist(), [False, True, True])

    def test_observed_zero_fp_does_not_claim_a_million_test(self):
        summary = {'metrics': {'n': 10000, 'spam': 5000, 'ham': 5000,
                               'tp': 4900, 'fn': 100, 'fp': 0, 'tn': 5000}}
        goal = summary_extra(summary)['goal']
        self.assertTrue(goal['observed_joint_goal_on_scored_mail'])
        self.assertFalse(goal['at_least_one_million_messages_scored'])
        self.assertFalse(goal['upper_95_fp_all_within_10ppm_iid_assumption'])
        self.assertFalse(goal['arbitrary_future_mail_guaranteed'])
        phish_only = dict(n=247, spam=247, ham=0, tp=247, fn=0, fp=0, tn=0)
        self.assertFalse(enrich(phish_only)['joint_98_and_10ppm_all_point_targets_met'])


if __name__ == '__main__':
    unittest.main()
