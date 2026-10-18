import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import v79_data as data
import v79_mail as runtime
import train_v79_en as train


class EnglishStub:
    f = None

    def __init__(self):
        self.f = self

    def predict(self, text, k, threshold, errors):
        return [(0.99, '__label__en')]


class V79Tests(unittest.TestCase):
    def test_duplicate_does_not_override_original_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = data.Pool(Path(directory) / 'pool.sqlite', EnglishStub())
            text = 'Please review the attached project report before the meeting tomorrow.'
            pool.add(text, 0, 'Original-email')
            pool.add(text, 0, 'Merged-public', prior=True)
            self.assertEqual(pool.db.execute('SELECT source FROM rows').fetchall(), [('Original-email',)])
            self.assertEqual(pool.db.execute('SELECT count(*) FROM prior').fetchone()[0], 1)
            self.assertEqual(pool.crosscheck['original_0_merged_0'], 1)
            pool.db.close()

    def test_label_conflict_removes_both_and_prevents_reinsertion(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = data.Pool(Path(directory) / 'pool.sqlite', EnglishStub())
            text = 'Please review the attached project report before the meeting tomorrow.'
            pool.add(text, 0, 'Original-email')
            pool.add(text, 1, 'Merged-public')
            pool.add(text, 0, 'Third-email')
            self.assertEqual(pool.db.execute('SELECT count(*) FROM rows').fetchone()[0], 0)
            self.assertEqual(pool.db.execute('SELECT count(*) FROM conflicts').fetchone()[0], 1)
            self.assertEqual(pool.crosscheck['original_0_merged_1'], 1)
            pool.db.close()

    def test_language_threshold_and_label_both_matter(self):
        class Detector:
            f = None
            def __init__(self, probability, label):
                self.f = self; self.result = [(probability, label)]
            def predict(self, *args):
                return self.result
        text = 'This message contains enough English words for language identification.'
        self.assertFalse(data.english_prediction(Detector(.99, '__label__fr'), text)[0])
        self.assertFalse(data.english_prediction(Detector(.79, '__label__en'), text)[0])
        self.assertTrue(data.english_prediction(Detector(.99, '__label__en'), text)[0])

    def test_source_cap_applies_when_aggregate_cap_would_pass(self):
        rows = [{'y': 0, 'source': 'rare'}] * 500 + [{'y': 0, 'source': 'common'}] * 9500
        values = np.array([1.] * 5 + [0.] * 9995)
        threshold = train.group_cutoff(values, rows, .001)
        self.assertGreater(threshold, 1.)
        self.assertEqual(np.sum(values >= threshold), 0)

    def test_candidate_choice_protects_worst_source(self):
        rows = ([{'y': 0, 'source': 'ham'}] * 1000 +
                [{'y': 1, 'source': 'one'}] * 100 + [{'y': 1, 'source': 'two'}] * 25)
        aggregate_only = np.array([0.] * 1000 + [1.] * 100 + [-1.] * 25)
        diverse = np.array([0.] * 1000 + [1.] * 75 + [-1.] * 25 + [1.] * 25)
        cfg = train.choose({'aggregate': aggregate_only, 'diverse': diverse}, rows, .001)
        self.assertEqual(cfg['candidate'], 'diverse')
        self.assertEqual(cfg['worst_select_source_recall'], .75)

    def test_bundled_language_model_checksum_prevents_replacement(self):
        with self.assertRaisesRegex(ValueError, 'checksum'):
            runtime.attach_language_model({'version': runtime.VERSION,
                                           'language_model_bytes': b'not the pinned identifier'})

    def test_fingerprint_changes_when_identity_or_family_changes(self):
        first = [{'identity': 'one', 'family': 'shared'}, {'identity': 'two', 'family': 'other'}]
        self.assertEqual(data.fingerprint(first), data.fingerprint(first[::-1]))
        self.assertNotEqual(data.fingerprint(first), data.fingerprint([dict(first[0], family='changed'), first[1]]))


if __name__ == '__main__':
    unittest.main()
