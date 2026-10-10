import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from benchmark_v48_calibrated import threshold_for_budget

class BudgetTests(unittest.TestCase):
    def test_ties_cannot_exceed_budget(self):
        y = np.array([0, 0, 0, 1])
        p = np.array([.9, .9, .7, .95])
        t = threshold_for_budget(y, p, 1)
        self.assertEqual(int(((p >= t) & (y == 0)).sum()), 0)
        self.assertTrue(p[-1] >= t)

    def test_zero_budget_and_exact_boundary(self):
        y = np.array([0, 0, 1])
        p = np.array([.8, .6, .9])
        for budget in (0, 1):
            t = threshold_for_budget(y, p, budget)
            self.assertEqual(int(((p >= t) & (y == 0)).sum()), budget)

    def test_invalid_scores_rejected(self):
        with self.assertRaises(ValueError):
            threshold_for_budget([0, 1], [float('nan'), .8], 0)

if __name__ == '__main__':
    unittest.main()
