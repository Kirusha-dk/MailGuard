import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from check_v82_replay_preflight import batch_starts


class ReplayPreflightTests(unittest.TestCase):
    def test_original_batches_and_quantile_boundary_are_preserved(self):
        scores = [float(i) for i in range(1000)]
        starts = batch_starts(scores, 701.)
        self.assertTrue({0, 32, 64, 96, 992, 672, 704} <= set(starts))
        self.assertTrue(all(start % 32 == 0 for start in starts))
        self.assertEqual(starts, sorted(set(starts)))
        self.assertLessEqual(len(starts), 17)

    def test_short_batch_and_empty_cache(self):
        self.assertEqual(batch_starts([.1, .2], .15), [0])
        with self.assertRaises(ValueError):
            batch_starts([], .5)


if __name__ == '__main__':
    unittest.main()
