import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from v57_runtime import protect_scores


class ProtectedTests(unittest.TestCase):
    def test_protected_votes_survive_every_probability_threshold(self):
        for threshold in (0., .5, .9999, 1.):
            scores = protect_scores([True, False, False], [0., .9, .1])
            decisions = scores >= threshold
            self.assertTrue(decisions[0])
            np.testing.assert_array_equal(decisions[1:], np.array([.9, .1]) >= threshold)

    def test_original_rescue_scores_are_not_mutated(self):
        rescue = np.array([.2, .4], dtype=np.float32)
        protected = protect_scores([True, False], rescue)
        np.testing.assert_array_equal(rescue, np.array([.2,.4], dtype=np.float32))
        self.assertEqual(protected.dtype, np.float64)
        self.assertEqual(protected[0], 2.)


if __name__ == '__main__':
    unittest.main()
