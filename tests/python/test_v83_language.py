import datetime as dt
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from train_v83_language import (CONFIG, check_progress, read_partition,
                                validate_request, validate_rows, validation_report)


class LanguagePilotGuardTests(unittest.TestCase):
    def request(self):
        return dict(language='ru', campaign='v83-ru-20261030-pilot', session_id='2026-10-30',
                    budget_steps=100, expected_min_step=0)

    def test_language_date_budget_and_cold_restart_guard(self):
        valid = self.request(); today = dt.date(2026, 10, 30)
        validate_request(valid, 'ru', today)
        for key, value in [('language', 'es'), ('session_id', '2026-10-31'),
                           ('budget_steps', 501), ('budget_steps', True), ('expected_min_step', -1)]:
            with self.assertRaises(ValueError):
                validate_request(dict(valid, **{key: value}), 'ru', today)
        with self.assertRaises(ValueError):
            check_progress(dict(step=0, active_session=None), dict(valid, expected_min_step=1))

    def test_interrupted_budget_cannot_be_extended_or_renamed(self):
        progress = dict(step=25, active_session='2026-10-30', active_start=0, active_target=100)
        check_progress(progress, self.request())
        for change in (dict(budget_steps=200), dict(session_id='2026-10-31')):
            with self.assertRaises(ValueError):
                check_progress(progress, dict(self.request(), **change))
        self.assertEqual(progress['active_target'], 100)

    def test_test_partition_is_never_opened(self):
        with self.assertRaisesRegex(ValueError, 'never opens TEST'):
            read_partition(Path('/does-not-exist'), 'test')

    def rows(self, count=60):
        return [dict(identity=str(i), family=str(i), source='public-email',
                     text='Russian or Spanish fixture text', y=i % 2, language='ru') for i in range(count)]

    def test_both_classes_counts_duplicates_and_language_are_required(self):
        rows = self.rows()
        expected = dict(n=60, spam=30, ham=30)
        validate_rows(rows, expected, 30, 'ru')
        for invalid in (rows[:-1], [dict(row, language='es') for row in rows], rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):
                validate_rows(invalid, expected, 30, 'ru')

    def test_small_select_does_not_fake_a_98_percent_target(self):
        rows = self.rows()
        result = validation_report(rows, [.1 if row['y'] == 0 else .9 for row in rows])
        self.assertEqual(result['fitted_98_status'], 'insufficient_for_98_target')
        self.assertIsNone(result['at_fitted_98_recall'])
        self.assertFalse(result['threshold_selected_on_same_development'])
        self.assertEqual(result['at_fixed_0_5']['tp'], 30)
        self.assertFalse(result['is_independent_test'])
        self.assertFalse(result['quality_gain_demonstrated'])

    def test_larger_select_fits_only_development_and_rejects_nan(self):
        rows = self.rows(400)
        result = validation_report(rows, [.1 if row['y'] == 0 else .9 for row in rows])
        self.assertEqual(result['fitted_98_status'], 'available')
        self.assertGreaterEqual(result['at_fitted_98_recall']['recall'], .98)
        self.assertFalse(result['is_independent_test'])
        self.assertEqual(sum(v['tp'] for v in result['by_source_fixed_0_5'].values()), 200)
        with self.assertRaises(ValueError):
            validation_report(rows, [float('nan')] * len(rows))


TORCH_READY = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None


@unittest.skipUnless(TORCH_READY, 'Pinned Torch/Transformers are required for optimizer continuation test')
class LanguagePilotContinuationTests(unittest.TestCase):
    def test_imported_engine_preserves_adam_scheduler_rng_cursor_for_new_config(self):
        import tempfile
        import torch
        from transformers import BertConfig, BertModel
        from v82_neural import (NeuralMail, begin_session, new_progress, optimizer_and_scheduler,
                                restore_checkpoint, run_steps, save_checkpoint, trainable_state)
        torch.set_num_threads(1)
        config = dict(CONFIG, rank=2, alpha=4, batch_size=2, max_length=16, max_epochs=5)
        rows = [dict(identity=str(i), text='fixture '+str(i), y=i % 2) for i in range(8)]
        class Tokenizer:
            def __call__(self, heads, tails, **kwargs):
                ids = [[1]+[3+ord(c)%20 for c in text[:8]]+[2] for text in heads]
                width = max(map(len, ids))
                return dict(input_ids=torch.tensor([r+[0]*(width-len(r)) for r in ids]),
                            attention_mask=torch.tensor([[1]*len(r)+[0]*(width-len(r)) for r in ids]))
        def make():
            torch.manual_seed(83)
            backbone = BertModel(BertConfig(vocab_size=24, hidden_size=16, num_hidden_layers=2,
                num_attention_heads=2, intermediate_size=32, hidden_dropout_prob=.1,
                attention_probs_dropout_prob=.1, attn_implementation='eager'))
            model = NeuralMail(backbone, config)
            return model, *optimizer_and_scheduler(model, 20, config)
        def advance(bundle, progress):
            run_steps(bundle[0], Tokenizer(), rows, [1.]*8, bundle[1], bundle[2], progress, config)
        full = make(); a = new_progress(); begin_session(a, '2026-10-30', 6); advance(full, a)
        split = make(); b = new_progress(); begin_session(b, '2026-10-30', 6)
        b['active_target'] = 3; advance(split, b); b['active_target'] = 6
        metadata = dict(language='ru', config=config, train='frozen-fixture')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            save_checkpoint(path, *split, b, metadata)
            resumed = make(); c = restore_checkpoint(path, *resumed, metadata); advance(resumed, c)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                restore_checkpoint(path, *make(), dict(metadata, language='es'))
        self.assertEqual(a, c)
        for name, value in trainable_state(full[0]).items():
            self.assertTrue(torch.equal(value, trainable_state(resumed[0])[name]), name)
        self.assertEqual(full[2].state_dict(), resumed[2].state_dict())


if __name__ == '__main__':
    unittest.main()
