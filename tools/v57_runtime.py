"""Local inference with frozen v54 decisions protected by a sentinel vote."""
from pathlib import Path
import joblib
import numpy as np
from train_v55_multilingual import file_sha, views, predict
from train_v56_consensus import load_components, predict_margin_union, gate_features, candidate_score


def protect_scores(base, rescue):
    scores = np.asarray(rescue, dtype=np.float64).copy()
    scores[np.asarray(base, dtype=bool)] = 2.
    return scores


def load_bundle(root):
    root = Path(root)
    artifact = joblib.load(root / 'model.joblib')
    if artifact.get('version') != 'v57-preserved-consensus' or not artifact.get('protectedV54Decisions'):
        raise ValueError('Expected a frozen v57 bundle')
    for name, digest in artifact['components'].items():
        if file_sha(root / name) != digest:
            raise ValueError(f'Component hash mismatch: {name}')
    old, neural = load_components(root / 'v54')
    new = joblib.load(root / 'v55/model.joblib')
    return artifact, old, neural, new


def predict_texts(bundle, texts, profile='conservative'):
    artifact, old_artifact, neural, new_artifact = bundle
    setting = artifact['profiles'][profile]
    rows = []
    for text in texts:
        focus, context = views(text)
        rows.append({'focus': focus, 'context': context})
    old = predict_margin_union(old_artifact, neural, texts).astype(np.float64)
    new = predict(new_artifact['model'], rows, new_artifact['idf'], new_artifact['ratio']).astype(np.float64)
    x = gate_features(rows, old, new, old_artifact['threshold'])
    rescue = candidate_score(setting['selected'], artifact['gates'], x, old, new, old_artifact['threshold'])
    return protect_scores(old >= old_artifact['threshold'], rescue)
