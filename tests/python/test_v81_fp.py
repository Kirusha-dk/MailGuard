import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from train_v81_fp import calibrate, paired
from v81_mail import apply_policy, decision_margins, candidate_scores


class ConstrainedRescueTests(unittest.TestCase):
    def test_rescue_cannot_create_spam_decisions(self):
        fast = np.array([[0., 0., -.2, -.2, -.2], [0., 0., .3, .2, .2],
                         [0., 0., .4, .1, .1]])
        policy = dict(enabled=True, baseline_cutoff=.1, partition='global', thresholds={'0': 0.})
        self.assertEqual(apply_policy(fast, np.array([100., -.1, 0.]), policy).tolist(),
                         [False, False, True])
        self.assertEqual((decision_margins(fast, np.array([100., -.1, 0.]), policy) >= 0).tolist(),
                         [False, False, True])
        with self.assertRaises(ValueError):
            apply_policy(fast, [0., float('nan'), 1.], policy)

    def test_guard_protects_each_spam_and_calibration_only_tightens(self):
        fast = np.ones((202, 5))
        rows = [{'y': 1}] * 200 + [{'y': 0}] * 2
        scores = np.array([2.] * 199 + [1.] + [-2., 3.])
        select = calibrate(fast, scores, rows, 'global', .1, .25)
        cal = calibrate(fast, scores - .5, rows, 'global', .1, .25, previous=select)
        self.assertLess(cal['0'], select['0'])
        policy = dict(enabled=True, baseline_cutoff=.1, partition='global', thresholds=cal)
        result = paired(np.ones(202, dtype=bool), apply_policy(fast, scores, policy), rows)
        self.assertEqual(result['baseline_spam_detections_lost'], 0)
        self.assertEqual(result['false_positives_removed'], 1)

    def test_sparse_regions_cannot_be_rescued(self):
        fast = np.ones((99, 5)); rows = [{'y': 1}] * 99
        self.assertEqual(calibrate(fast, np.ones(99), rows, 'global', .1, .1), {})

    def test_equal_recall_must_not_hide_lost_baseline_spam(self):
        result = paired(np.array([True, False]), np.array([False, True]), [{'y': 1}, {'y': 1}])
        self.assertEqual(result['baseline']['tp'], result['proposed']['tp'])
        self.assertEqual(result['baseline_spam_detections_lost'], 1)
        self.assertEqual(result['spam_detections_gained'], 1)

    def test_one_or_two_similar_ham_anchors_cannot_supply_three_email_support(self):
        expert = dict(input='neighbors', ham=np.array([[1., 0.], [1., 0.], [0., 1.]]),
                      spam=np.array([[0., 1.]]))
        bundle = {'_baseline': {}, 'experts': {'neighbors': expert}}
        arguments = (np.zeros((1, 8)), np.array([[1., 0.]]), np.zeros((1, 7)))
        self.assertAlmostEqual(candidate_scores(bundle, 'neighbors', *arguments)[0], 0.)
        expert['ham'][2] = [1., 0.]
        self.assertLess(candidate_scores(bundle, 'neighbors', *arguments)[0], -5.)


if __name__ == '__main__':
    unittest.main()
