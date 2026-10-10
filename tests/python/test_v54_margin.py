import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from train_v54_margin import nb_ratio, margin_probability
from train_v51_residual import residual_threshold, scored_union


class MarginChecks(unittest.TestCase):
    def test_class_ratio_reverses_with_labels(self):
        x = csr_matrix([[4., 0.], [3., 0.], [0., 2.], [0., 4.]])
        y = np.array([1, 1, 0, 0])
        r = nb_ratio(x, y)
        self.assertGreater(r[0], 0)
        self.assertLess(r[1], 0)
        np.testing.assert_allclose(r, -nb_ratio(x, 1 - y))

    def test_ranking_and_frozen_decisions(self):
        class Dummy:
            def decision_function(self, x):
                return np.array([-3., 0., 3.])
        p = margin_probability(Dummy(), None)
        self.assertTrue(np.all(np.diff(p) > 0))
        y = np.array([0, 0, 1])
        base = np.array([True, False, False])
        threshold, inherited = residual_threshold(y, base, p, 1)
        self.assertEqual(inherited, 1)
        decision = scored_union(base, p, threshold) >= threshold
        np.testing.assert_array_equal(decision, [True, False, True])
        disabled, _ = residual_threshold(y, base, p, 0)
        np.testing.assert_array_equal(scored_union(base, p, disabled) >= disabled, base)


if __name__ == '__main__':
    unittest.main()
