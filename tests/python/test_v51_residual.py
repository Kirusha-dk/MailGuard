import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from train_v51_residual import residual_threshold, scored_union, gate_features

class RescueTests(unittest.TestCase):
    def test_disabled_rescue_keeps_every_base_decision(self):
        base = np.array([True, False, True])
        scores = scored_union(base, [.1, .99, .01], 2.)
        np.testing.assert_array_equal(scores >= 2., base)

    def test_inherited_fp_consumes_allowance(self):
        y = np.array([0, 0, 0, 1])
        base = np.array([True, False, False, False])
        scores = np.array([.1, .8, .6, .9])
        t, inherited = residual_threshold(y, base, scores, 1)
        pred = scored_union(base, scores, t) >= t
        self.assertEqual(inherited, 1)
        self.assertEqual(int(((y == 0) & pred).sum()), 1)
        self.assertTrue(pred[-1])

    def test_over_budget_base_disables_rescue(self):
        t, _ = residual_threshold([0, 0, 1], [True, False, False], [.1, .99, .99], 0)
        self.assertEqual(t, 2.)

    def test_feature_extremes_are_finite(self):
        self.assertTrue(np.isfinite(gate_features(np.array([0., 1.]), np.array([1., 0.]))).all())

if __name__ == '__main__':
    unittest.main()
