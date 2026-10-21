import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from check_v82_inference_replay import compare_scores, inference_context_conclusion, replay_metrics
from summarize_v82_audit_cache import log_loss


class InferenceReplayTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(identity=str(i), y=i % 2, text='SECRET_BODY') for i in range(4)]

    def test_identical_contexts_do_not_imply_historical_loss_was_reproduced(self):
        scores = [.2, .8, .3, .9]
        report = dict(development=dict(n=4, spam=2, ham=2, tp=2, fp=0, threshold=.5,
                                      log_loss=log_loss(self.rows, scores) + 1e-7))
        result = replay_metrics(self.rows, scores, report)
        comparison = compare_scores(self.rows, scores, scores, .5)
        self.assertTrue(result['source_report_counts_reproduced'])
        self.assertFalse(result['source_report_log_loss_reproduced'])
        self.assertEqual(inference_context_conclusion(result, result, comparison), 'contexts_identical_on_this_replay')
        self.assertEqual(comparison['exactly_equal_scores'], 4)

    def test_numeric_jitter_and_individual_classification_changes_are_separate(self):
        old = [.2, .50000001, .3, .49999999]
        new = [.20000001, .49999999, .3, .50000001]
        result = compare_scores(self.rows, old, new, .5)
        self.assertEqual(result['exactly_equal_scores'], 1)
        self.assertAlmostEqual(result['max_absolute_score_difference'], 2e-8)
        transitions = result['paired_at_locked_threshold']
        self.assertEqual(transitions['baseline']['tp'], transitions['current']['tp'])
        self.assertEqual(transitions['baseline_spam_detections_lost'], 1)
        self.assertEqual(transitions['spam_detections_gained'], 1)
        self.assertNotIn('SECRET_BODY', json.dumps(result))

    def test_invalid_scores_are_rejected_before_comparison(self):
        for scores in ([.2], [float('nan')] * 4, [1.1] * 4):
            with self.assertRaises(ValueError): compare_scores(self.rows, [.5] * 4, scores, .5)

    def test_context_explanation_requires_original_context_to_reproduce_loss(self):
        changed = dict(max_absolute_score_difference=1e-7)
        yes = dict(source_report_log_loss_reproduced=True)
        no = dict(source_report_log_loss_reproduced=False)
        self.assertEqual(inference_context_conclusion(yes, no, changed),
                         'original_no_grad_reproduces_historical_loss_but_inference_mode_does_not')
        self.assertEqual(inference_context_conclusion(no, no, changed),
                         'context_difference_observed_but_historical_replay_not_explained')


if __name__ == '__main__':
    unittest.main()
