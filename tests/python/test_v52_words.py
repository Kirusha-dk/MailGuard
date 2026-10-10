import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from train_v52_wordnet import word_encode, WordNet, TOKENS

class WordTests(unittest.TestCase):
    def test_deterministic_casefold_and_empty(self):
        x = word_encode(['HELLO world', 'hello WORLD', ''])
        np.testing.assert_array_equal(x[0], x[1])
        self.assertFalse(x[2].any())
        self.assertEqual(x.shape[1], TOKENS)

    def test_middle_changes_representation(self):
        words = ['head'] * 600 + ['middle'] * 100 + ['tail'] * 600
        changed = ['head'] * 600 + ['phishing'] * 100 + ['tail'] * 600
        self.assertFalse(np.array_equal(word_encode([' '.join(words)]), word_encode([' '.join(changed)])))

    def test_empty_logits_and_gradients_finite(self):
        torch.set_num_threads(2)
        model = WordNet()
        x = torch.from_numpy(word_encode(['', 'claim cash', 'project meeting']))
        logits = model(x)
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        self.assertTrue(torch.isfinite(model.embedding.weight.grad).all())

if __name__ == '__main__':
    unittest.main()
