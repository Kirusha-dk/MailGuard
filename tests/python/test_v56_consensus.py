import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from train_v56_consensus import budget, gate_features, gate_partition, candidate_score, limited


class ConsensusTests(unittest.TestCase):
    def test_label_and_source_are_not_gate_features(self):
        row = {'focus': 'subject note hosts example.org start hello', 'context': 'subject note body hello'}
        a = dict(row, y=0, source='SpamArchive-2017', identity='a', family='b')
        b = dict(row, y=1, source='Enron', identity='other', family='other')
        np.testing.assert_array_equal(gate_features([a], [.8], [.7], .68), gate_features([b], [.8], [.7], .68))

    def test_gate_features_are_finite_for_sentinel_and_zero(self):
        rows = [{'focus': 'subject hosts start', 'context': '日本語 https://example.org'}] * 2
        x = gate_features(rows, np.array([2., 0.]), np.array([1., 0.]), .68)
        self.assertTrue(np.isfinite(x).all())
        self.assertGreater(x[0, 6], 0)

    def test_family_deterministically_stays_in_one_gate_partition(self):
        family = 'f' * 64
        self.assertEqual(gate_partition(family), gate_partition(family))
        partitions = {gate_partition(f'{i:064x}') for i in range(20)}
        self.assertEqual(partitions, {'fit', 'select'})

    def test_sampling_does_not_use_labels(self):
        rows = [dict(identity=f'{i:04}', source=s, y=i % 2) for i in range(8) for s in ('Enron', 'SpamArchive-2017')]
        altered = [dict(r, y=1-r['y']) for r in rows]
        a = [(r['identity'], r['source']) for r in limited(rows, 4)]
        b = [(r['identity'], r['source']) for r in limited(altered, 4)]
        self.assertEqual(a, b)
        self.assertEqual(len(a), 4)

    def test_consensus_promotes_old_vote_without_forcing_every_old_positive(self):
        old = np.array([2., .2, 2.]); new = np.array([.8, .9, .1])
        s = candidate_score('consensus-rank', {}, None, old, new, .68)
        np.testing.assert_allclose(s, [.9, .9, .55])
        self.assertFalse(s[2] >= .8)

    def test_fp_allowance_counts_ham(self):
        self.assertEqual(budget([0] * 2500 + [1] * 25000, .0002), 0)
        self.assertEqual(budget([0] * 2500 + [1] * 25000, .002), 5)


if __name__ == '__main__':
    unittest.main()
