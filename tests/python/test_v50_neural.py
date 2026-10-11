import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from train_v50_neural import ByteCNN, encode, mix, WINDOW

class NeuralTests(unittest.TestCase):
    def test_utf8_head_tail_and_padding(self):
        x = encode(['abc', 'Ж' * 2000])
        self.assertEqual(x.shape, (2, 2 * WINDOW))
        self.assertEqual(x[0, 0], ord('a'))
        self.assertEqual(x[0, WINDOW], ord('a'))
        self.assertEqual(x[0, 3], 0)

    def test_gradient_and_restored_state(self):
        torch.set_num_threads(2)
        model = ByteCNN()
        x = torch.from_numpy(encode(['team meeting', 'claim lottery cash']))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), torch.tensor([0., 1.]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.embedding.weight.grad).all())
        restored = ByteCNN()
        restored.load_state_dict(model.state_dict())
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(x), restored(x))

    def test_blend_endpoints(self):
        sparse, neural = np.array([.2, .8]), np.array([.7, .3])
        np.testing.assert_allclose(mix(sparse, neural, 0), sparse)
        np.testing.assert_allclose(mix(sparse, neural, 1), neural)

if __name__ == '__main__':
    unittest.main()
