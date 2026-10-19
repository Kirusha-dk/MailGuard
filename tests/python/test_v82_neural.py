import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import BertConfig, BertModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from v82_neural import (CONFIG, NeuralMail, begin_session, new_progress, optimizer_and_scheduler,
                        restore_checkpoint, run_steps, save_checkpoint, trainable_state)


class Tokenizer:
    def __call__(self, heads, tails, **kwargs):
        ids = [[1] + [3 + ord(c) % 20 for c in text[:8]] + [2] for text in heads]
        width = max(map(len, ids))
        return dict(input_ids=torch.tensor([r + [0] * (width - len(r)) for r in ids]),
                    attention_mask=torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in ids]))


class NeuralContinuationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = dict(CONFIG, rank=2, alpha=4, batch_size=2, max_epochs=5, max_length=16)
        self.rows = [dict(identity=str(i), text='English letter ' + str(i), y=i % 2) for i in range(8)]
        self.metadata = dict(config=self.config, train='fixed-train-fingerprint')

    def make(self):
        torch.manual_seed(82)
        base = BertModel(BertConfig(vocab_size=24, hidden_size=16, num_hidden_layers=2,
                                   num_attention_heads=2, intermediate_size=32,
                                   hidden_dropout_prob=.1, attention_probs_dropout_prob=.1,
                                   attn_implementation='eager'))
        model = NeuralMail(base, self.config)
        optimizer, scheduler = optimizer_and_scheduler(model, 20, self.config)
        return model, optimizer, scheduler

    def advance(self, bundle, progress):
        return run_steps(bundle[0], Tokenizer(), self.rows, [1.] * 8,
                         bundle[1], bundle[2], progress, self.config)

    def test_internal_adapters_learn_and_original_weights_stay_frozen(self):
        bundle = self.make(); model = bundle[0]
        frozen = {k: p.detach().clone() for k, p in model.named_parameters() if not p.requires_grad}
        before = trainable_state(model); progress = new_progress(); begin_session(progress, 'day-1', 3)
        self.advance(bundle, progress); after = trainable_state(model)
        self.assertTrue(any(not torch.equal(after[k], before[k]) for k in after if k.endswith('.b')))
        for name, value in model.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(value, frozen[name]), name)

    def test_resume_preserves_adam_scheduler_dropout_rng_and_cursor_exactly(self):
        full = self.make(); a = new_progress(); begin_session(a, 'day-1', 6); self.advance(full, a)
        split = self.make(); b = new_progress(); begin_session(b, 'day-1', 3); self.advance(split, b)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            save_checkpoint(path, *split, b, self.metadata)
            resumed = self.make()
            c = restore_checkpoint(path, *resumed, self.metadata)
            c['active_target'] = 6; self.advance(resumed, c)
        for name, value in trainable_state(full[0]).items():
            self.assertTrue(torch.equal(value, trainable_state(resumed[0])[name]), name)
        self.assertEqual(a, c)
        self.assertEqual(full[2].state_dict(), resumed[2].state_dict())

    def test_changed_configuration_or_data_rejects_checkpoint(self):
        bundle = self.make()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            save_checkpoint(path, *bundle, new_progress(), self.metadata)
            changed = copy.deepcopy(self.metadata); changed['train'] = 'another-dataset'
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                restore_checkpoint(path, *bundle, changed)

    def test_retry_cannot_reset_budget_or_repeat_completed_day(self):
        progress = new_progress(); self.assertTrue(begin_session(progress, 'day-1', 100))
        progress['step'] = 25
        self.assertTrue(begin_session(progress, 'day-1', 100))
        self.assertEqual(progress['active_target'], 100)
        with self.assertRaises(ValueError):
            begin_session(progress, 'day-2', 100)
        progress['completed_sessions'].append('day-1'); progress['active_session'] = None
        self.assertFalse(begin_session(progress, 'day-1', 100))


if __name__ == '__main__':
    unittest.main()
