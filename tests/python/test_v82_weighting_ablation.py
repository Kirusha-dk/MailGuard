import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from run_v82_weighting_ablation import fork_progress, weights_for
from audit_v82_errors import paired


class WeightingAblationTests(unittest.TestCase):
    def test_candidate_preserves_ham_cost_and_global_mean(self):
        rows = [dict(source='large', y=0)] * 100 + [dict(source='small', y=0)] + [dict(source='spam', y=1)] * 2
        candidate = weights_for(rows, 'candidate')
        self.assertAlmostEqual(sum(candidate) / len(rows), 1.)
        self.assertEqual(candidate[0], candidate[100])
        self.assertAlmostEqual(candidate[0] / candidate[-1], 2.)
        control = weights_for(rows, 'control')
        expected = [min(12., 5000 / 100) * 2] * 100 + [24.] + [12.] * 2
        mean = sum(expected) / len(expected)
        self.assertEqual(control, [v / mean for v in expected])

    def test_original_source_balance_for_large_groups(self):
        rows = [dict(source='a', y=0)] * 1000 + [dict(source='b', y=0)] * 5000
        weights = weights_for(rows, 'control')
        self.assertAlmostEqual(sum(weights[:1000]), sum(weights[1000:]))

    def test_fork_preserves_position_and_has_own_budget(self):
        parent = dict(step=500, cursor=8000, epoch=0, active_session=None,
                      active_target=None, stopped=False, completed_sessions=['2026-10-20'])
        snapshot = copy.deepcopy(parent)
        fork = fork_progress(parent)
        self.assertEqual(parent, snapshot)
        self.assertEqual(fork['cursor'], parent['cursor'])
        self.assertEqual(fork['active_target'], 1000)
        self.assertEqual(fork['active_start'], 500)
        self.assertEqual(fork['completed_sessions'], [])
        for field, value in [('step', 1000), ('cursor', 0), ('active_session', 'unfinished'), ('stopped', True)]:
            invalid = dict(parent, **{field: value})
            with self.assertRaises(ValueError):
                fork_progress(invalid)

    def test_equal_tp_can_hide_lost_spam(self):
        rows = [dict(identity=str(i), y=y) for i, y in enumerate([1, 1, 0, 0])]
        result = paired(rows, [.9, .1, .9, .1], [.1, .9, .1, .1], .5, .5)
        self.assertEqual(result['baseline']['tp'], result['current']['tp'])
        self.assertEqual(result['baseline_spam_detections_lost'], 1)
        self.assertEqual(result['false_positives_removed'], 1)


if __name__ == '__main__':
    unittest.main()
