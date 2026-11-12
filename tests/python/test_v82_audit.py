import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from audit_v82_errors import describe, metric, paired, seen_train


class ReadOnlyDevelopmentAuditTests(unittest.TestCase):
    def test_equal_tp_totals_do_not_hide_individual_spam_losses(self):
        rows = [dict(identity='a', y=1), dict(identity='b', y=1), dict(identity='c', y=0)]
        result = paired(rows, [.9, .1, .9], [.1, .9, .1], .5, .5)
        self.assertEqual(result['baseline']['tp'], result['current']['tp'])
        self.assertEqual(result['baseline_spam_detections_lost'], 1)
        self.assertEqual(result['first_lost_spam_identities'], ['a'])
        self.assertEqual(result['false_positives_removed'], 1)
        self.assertFalse(result['is_a_v80_regression_check'])

    def test_group_totals_reconcile_and_message_text_is_not_serialized(self):
        rows = [dict(identity=str(i), source='s' + str(i % 2), y=i % 2,
                     text='SECRET_PRIVATE_CONTENT ' + 'word ' * (i * 30)) for i in range(6)]
        result = describe(rows, [.9, .2, .4, .8, .7, .1], .5)
        for grouping in ('by_source', 'by_word_count', 'by_character_window'):
            for key in ('n', 'spam', 'ham', 'tp', 'fp', 'fn', 'tn'):
                self.assertEqual(result['at_locked_development_threshold'][key],
                                 sum(group[key] for group in result[grouping].values()))
        self.assertNotIn('SECRET_PRIVATE_CONTENT', json.dumps(result))

    def test_nonfinite_scores_and_misaligned_inputs_cannot_be_reported(self):
        with self.assertRaises(ValueError): metric([{'y': 0}], [float('nan')], .5)
        with self.assertRaises(ValueError): metric([{'y': 0}], [], .5)
        with self.assertRaises(ValueError): metric([{'y': 2}], [.5], .5)
        self.assertIsNone(metric([{'y': 1}], [.8], .5)['fpr'])

    def test_training_sample_contains_only_seen_rows_and_is_label_blind(self):
        rows = [dict(identity=str(i), y=i % 2) for i in range(30)]
        sample, count = seen_train(rows, dict(epoch=0, cursor=8), 82, 5)
        eligible = sorted(rows, key=lambda r: hashlib.sha256(f"82:0:{r['identity']}".encode()).hexdigest())[:8]
        self.assertEqual(count, 8)
        self.assertTrue({r['identity'] for r in sample} <= {r['identity'] for r in eligible})
        changed = [dict(r, y=1-r['y']) for r in rows]
        again, _ = seen_train(changed, dict(epoch=0, cursor=8), 82, 5)
        self.assertEqual([r['identity'] for r in sample], [r['identity'] for r in again])


if __name__ == '__main__':
    unittest.main()
