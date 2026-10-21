import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from summarize_v82_audit_cache import analyze, log_loss, preservation_frontier, validate_alignment


class CachedAuditTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(identity=str(i), source='a' if i < 2 else 'b', y=i % 2,
                          family=str(i), text='PRIVATE_BODY ' + 'word ' * (i * 50)) for i in range(6)]

    def test_scalar_frontier_preserves_individual_tp_and_includes_ties(self):
        old = [.8, .9, .1, .8, .1, .1]
        new = [.4, .2, .1, .7, .2, .9]
        result = preservation_frontier(self.rows, old, new, .5)
        self.assertEqual(result['threshold'], .2)
        paired = result['paired_select']
        self.assertEqual(paired['baseline_spam_detections_lost'], 0)
        self.assertEqual(paired['spam_detections_gained'], 1)
        self.assertEqual(paired['current']['fp'], 2)
        # Any higher cutoff loses the original TP with score .2.
        self.assertTrue(result['selected_on_same_development'])
        self.assertFalse(result['deployable_policy'])

    def test_permutation_labels_duplicates_and_invalid_probabilities_are_rejected(self):
        ids = [r['identity'] for r in self.rows]; labels = [r['y'] for r in self.rows]
        validate_alignment(self.rows, ids, labels, [.5] * 6)
        for identities, ys, scores in ((ids[::-1], labels, [.5] * 6),
                                       (ids, labels[::-1], [.5] * 6),
                                       (ids, labels, [1.01] * 6),
                                       (ids, labels, [float('nan')] * 6)):
            with self.assertRaises(ValueError):
                validate_alignment(self.rows, identities, ys, scores)
        with self.assertRaises(ValueError):
            validate_alignment([self.rows[0]] * 2, [ids[0]] * 2, [0] * 2, [.5] * 2)

    def test_group_transitions_reconcile_and_no_message_body_is_saved(self):
        old = [.8, .9, .1, .8, .1, .1]; new = [.4, .2, .1, .7, .2, .9]
        result = analyze(self.rows, self.rows, old, new, new, .5, .5)
        for key in ('paired_by_source', 'paired_by_word_count'):
            self.assertEqual(sum(g['baseline_spam_detections_lost'] for g in result[key].values()), 1)
            self.assertEqual(sum(g['spam_detections_gained'] for g in result[key].values()), 1)
            self.assertEqual(sum(g['false_positives_removed'] for g in result[key].values()), 1)
        self.assertNotIn('PRIVATE_BODY', json.dumps(result))
        self.assertEqual(result['source_standardized_seen_train_fpr'], result['observed_select_fpr'])

    def test_loss_is_unweighted_and_handles_probability_endpoints(self):
        self.assertAlmostEqual(log_loss(self.rows, [.5] * 6), math.log(2))
        self.assertTrue(math.isfinite(log_loss(self.rows, [r['y'] for r in self.rows])))
        with self.assertRaises(ValueError): log_loss([], [])
        with self.assertRaises(ValueError): preservation_frontier(self.rows, [0.] * 6, [.5] * 6, .5)


if __name__ == '__main__':
    unittest.main()
