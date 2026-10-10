import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
import tempfile
import joblib
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from benchmark_v49_nbsvm import nb_ratio, transform
from benchmark_v48_calibrated import features, threshold_for_budget

class RatioTests(unittest.TestCase):
    def test_spam_and_ham_features_have_opposite_signs(self):
        x = csr_matrix([[10, 0], [0, 10]])
        ratio = nb_ratio(x, [1, 0])
        self.assertGreater(ratio[0], 0)
        self.assertLess(ratio[1], 0)
        self.assertTrue(np.isfinite(ratio).all())

    def test_requires_both_classes(self):
        with self.assertRaises(ValueError):
            nb_ratio(csr_matrix([[1, 0]]), [1])

    def test_saved_model_preserves_scores_and_calibration(self):
        texts = ['meeting agenda project review', 'team schedule notes',
                 'claim cash lottery prize now', 'free winner money offer']
        y = np.array([0, 0, 1, 1])
        counts = features(texts)
        tfidf = TfidfTransformer(sublinear_tf=True).fit(counts)
        artifact = {'tfidf': tfidf, 'nbRatio': nb_ratio(counts, y)}
        x = transform(artifact, texts)
        model = LogisticRegression(solver='liblinear').fit(x, y)
        artifact['model'] = model
        scores = model.predict_proba(x)[:, 1]
        artifact['threshold'] = threshold_for_budget(y, scores, 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.joblib'
            joblib.dump(artifact, path)
            saved = joblib.load(path)
            restored = saved['model'].predict_proba(transform(saved, texts))[:, 1]
        np.testing.assert_allclose(restored, scores)
        self.assertEqual(saved['threshold'], artifact['threshold'])
        self.assertEqual(int(((restored >= saved['threshold']) & (y == 0)).sum()), 0)

if __name__ == '__main__':
    unittest.main()
