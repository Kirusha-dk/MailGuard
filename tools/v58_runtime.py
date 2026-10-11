"""The protected consensus with a threshold calibrated on residual ham."""
from pathlib import Path
import joblib
import numpy as np
from benchmark_v48_calibrated import threshold_for_budget
from train_v55_multilingual import file_sha
from train_v56_consensus import load_components
from v57_runtime import predict_texts


def residual_threshold(labels, base, scores, minimum):
    labels, base, scores = np.asarray(labels), np.asarray(base, dtype=bool), np.asarray(scores)
    residual = ~base
    if not np.any((labels == 0) & residual):
        return 2.
    return max(minimum, threshold_for_budget(labels[residual], scores[residual], 0))


def load_bundle(root):
    root = Path(root)
    artifact = joblib.load(root / 'model.joblib')
    if artifact.get('version') != 'v58-calibrated-protected-consensus':
        raise ValueError('Expected a frozen v58 bundle')
    for name, digest in artifact['components'].items():
        if file_sha(root / name) != digest:
            raise ValueError(f'Component hash mismatch: {name}')
    old, neural = load_components(root / 'v54')
    new = joblib.load(root / 'v55/model.joblib')
    return artifact, old, neural, new
