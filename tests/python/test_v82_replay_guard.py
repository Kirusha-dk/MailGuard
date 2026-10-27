import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from v82_replay_guard import replay_differences, require_exact_reference_scores, locked_decision_replay


class ReplayGuardTests(unittest.TestCase):
    def test_small_score_drift_with_unchanged_locked_decisions(self):
        rows = [dict(identity='a', y=1), dict(identity='b', y=0)]
        result = locked_decision_replay(rows, [.8, .2], [.80000001, .20000001], .5)
        self.assertTrue(result['all_individual_decisions_reproduced'])
        self.assertGreater(result['max_absolute_score_difference'], 0)

    def test_equal_tp_totals_do_not_pass_if_identities_change(self):
        rows = [dict(identity='a', y=1), dict(identity='b', y=1)]
        result = locked_decision_replay(rows, [.8, .2], [.2, .8], .5)
        self.assertEqual(result['baseline']['tp'], result['current']['tp'])
        self.assertFalse(result['all_individual_decisions_reproduced'])

    def setUp(self):
        self.metrics = dict(n=200, spam=100, ham=100, tp=98, fp=5, threshold=.01, log_loss=.2)

    def test_small_threshold_drift_is_reported_without_authorizing_training(self):
        result = replay_differences(self.metrics, dict(self.metrics, threshold=.0100000001))
        self.assertTrue(result['counts_match'])
        self.assertFalse(result['exact_metrics_match'])
        self.assertFalse(result['training_authorized'])
        self.assertIn('threshold', result['numerical_differences'])

    def test_changed_fp_is_a_decision_count_failure(self):
        result = replay_differences(self.metrics, dict(self.metrics, fp=6))
        self.assertEqual(result['interpretation'], 'decision_counts_changed')
        self.assertFalse(result['counts_match'])

    def test_exact_counts_do_not_allow_changed_scores_or_swapped_identity(self):
        require_exact_reference_scores([.1, .2], [.1, .2], ['a', 'b'], ['a', 'b'])
        for scores, ids in [([.1, .200000001], ['a', 'b']), ([.1, .2], ['b', 'a']),
                            ([.1, float('nan')], ['a', 'b'])]:
            with self.assertRaises(ValueError):
                require_exact_reference_scores([.1, .2], scores, ['a', 'b'], ids)


if __name__ == '__main__':
    unittest.main()
