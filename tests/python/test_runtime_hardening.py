import contextlib
import io
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import v77_mail as core
import v80_mail as v80
from v80_metrics import summary_extra


class RuntimeHardeningTests(unittest.TestCase):
    def result(self, score=1., cutoff=0., language='en'):
        return dict(score=score, cutoff=cutoff, language=language, spam=score >= cutoff)

    def test_batch_length_abstentions_and_invalid_predictions(self):
        self.assertEqual(core.checked_predictions([None], 1, ['en']), [None])
        for results in ([], [self.result(), self.result()], [self.result(float('nan'))],
                        [self.result(cutoff=float('inf'))], [self.result(language='xx')],
                        [dict(self.result(), spam=False)]):
            with self.assertRaises(ValueError):
                core.checked_predictions(results, 1, ['en'])

    def test_nonfinite_threshold_and_multidimensional_metrics_rejected(self):
        for scores, labels, cutoff in [([1.], [1], float('nan')), ([[1.]], [[1]], 0.)]:
            with self.assertRaises(ValueError):
                core.measure(scores, labels, cutoff)

    def run_cli(self, root, predictor):
        ham, spam, out = root / 'ham', root / 'spam', root / 'out'
        ham.mkdir(); spam.mkdir()
        (ham / 'one.eml').write_text('Subject: Meeting\n\nPlease confirm the scheduled meeting tomorrow with our colleagues.')
        (spam / 'two.eml').write_text('Subject: Reward\n\nClaim the special reward immediately before the promotional offer expires.')
        model = root / 'model.joblib'; model.write_bytes(b'fixture')
        argv = ['check', '--model', str(model), '--ham', str(ham), '--spam', str(spam), '--output', str(out)]
        with patch.object(sys, 'argv', argv), patch.object(core.joblib, 'load', return_value={}), contextlib.redirect_stdout(io.StringIO()):
            core.main(predictor=predictor, allowed_languages=('en',))
        return out

    def test_abstention_is_counted_and_never_converted_to_ham(self):
        with tempfile.TemporaryDirectory() as directory:
            out = self.run_cli(Path(directory), lambda *args: [None, self.result()])
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['files_scored'], 1)
            self.assertEqual(summary['files_unscored'], 1)
            self.assertEqual(summary['excluded']['prediction_abstentions'], 1)
            self.assertEqual(summary['metrics']['ham'], 0)
            self.assertFalse((out / 'predictions.csv.partial').exists())

    def test_short_predictor_cannot_publish_a_successful_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'results for'):
                self.run_cli(root, lambda *args: [self.result()])
            self.assertFalse((root / 'out/summary.json').exists())
            self.assertFalse((root / 'out/predictions.csv').exists())

    def test_goal_uses_ham_denominator_even_in_spam_heavy_corpus(self):
        metric = dict(n=1000100, spam=1000000, ham=100, tp=990000, fn=10000, fp=1, tn=99)
        result = summary_extra({'metrics': metric})
        self.assertTrue(result['goal']['observed_joint_goal_per_all_messages'])
        self.assertFalse(result['goal']['observed_joint_goal_on_scored_mail'])
        self.assertEqual(result['goal']['primary_fp_denominator'], 'ham')

    def test_every_lexical_candidate_has_identical_scores_without_neural_work(self):
        fast = np.random.default_rng(80).normal(size=(17, 5))
        full = np.column_stack([fast, np.zeros((17, 3))])
        word, char, tfidf, pw, pc = fast.T
        oracle = {'v79-word': word, 'v79-char': char, 'v79-tfidf': tfidf,
                  'v79-blend': .25 * (word + char) + .5 * tfidf,
                  'phish-word': pw, 'phish-mean': .5 * (pw + pc),
                  'fast-or': np.maximum(tfidf, .5 * (pw + pc))}
        for candidate, expected in oracle.items():
            np.testing.assert_array_equal(v80.candidate_scores({}, full)[candidate], expected)
            bundle = dict(version=v80.VERSION, policies={'recall-98': dict(candidate=candidate, cutoff=.1)})
            with patch.object(v80, 'supported', return_value=True), patch.object(v80, 'fast_features', return_value=fast), \
                 patch.object(v80, 'feature_matrix', side_effect=AssertionError('Unneeded neural work')), \
                 patch.object(v80, 'Encoder', side_effect=AssertionError('Unneeded encoder load')):
                predictions = v80.predict(bundle, ['message'] * 17, 'recall-98')
            np.testing.assert_array_equal([p['score'] for p in predictions], expected)
            self.assertEqual([p['spam'] for p in predictions], (expected >= .1).tolist())

    def test_cli_skips_encoder_only_for_lexical_policy(self):
        for candidate, expected in [('fast-or', False), ('semantic', True)]:
            bundle = dict(policies={'recall-98': dict(candidate=candidate)})
            args = SimpleNamespace(profile='recall-98', model='fixture/model.joblib')
            with patch.object(v80, 'attach') as attach:
                v80.attach_cli(bundle, args)
            attach.assert_called_once_with(bundle, args, load_encoder=expected)


if __name__ == '__main__':
    unittest.main()
