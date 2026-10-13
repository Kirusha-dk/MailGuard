"""Frozen v60 sparse / Unicode neural ensemble; identical training/runtime features."""
from pathlib import Path
import joblib
import numpy as np
import torch
from train_v55_multilingual import file_sha, predict, views
from train_v56_consensus import gate_features
from v59_runtime import load_bundle as load59, predict_texts as predict59
from v60_neural import UnicodeCNN, predict as neural_predict


def stack_features(rows, sparse, neural, previous):
    # No source, collection month, filename, label or annotation is a feature.
    def logit(p):
        p = np.clip(p, 1e-7, 1-1e-7)
        return np.log(p/(1-p))
    x = gate_features(rows, previous, sparse, .5)
    return np.column_stack([x, neural, logit(neural), logit(previous),
                            logit(sparse) - logit(neural), logit(sparse) - logit(previous)])


def candidate_score(name, gates, x, sparse, neural, previous):
    if name == 'v59-frozen':
        return previous
    if name == 'sparse-only':
        return sparse
    if name == 'neural-only':
        return neural
    if name.startswith('blend-'):
        weight = float(name.split('-', 1)[1])
        def logit(p):
            p = np.clip(p, 1e-7, 1-1e-7)
            return np.log(p/(1-p))
        z = (1-weight)*logit(sparse) + weight*logit(neural)
        return 1/(1+np.exp(-np.clip(z, -40, 40)))
    return gates[name].predict_proba(x)[:, 1].astype(np.float64)


def load_bundle(root):
    root = Path(root)
    artifact = joblib.load(root / 'model.joblib')
    if artifact.get('version') != 'v60-unicode-ensemble':
        raise ValueError('Expected v60 model')
    for name, digest in artifact['components'].items():
        if file_sha(root / name) != digest:
            raise ValueError('Component hash mismatch: ' + name)
    model = UnicodeCNN()
    model.load_state_dict(torch.load(root / 'unicode.pt', map_location='cpu', weights_only=True)['state'])
    return artifact, model, load59(root / 'v59')


def predict_texts(bundle, texts, profile='conservative', original_texts=None):
    artifact, model, previous_bundle = bundle
    rows = [dict(zip(('focus', 'context'), views(t))) for t in texts]
    core = artifact['core']
    sparse = predict(core['model'], rows, core['idf'], core['ratio']).astype(np.float64)
    setting = artifact['profiles'][profile]
    neural = neural_predict(model, rows)
    previous = predict59(previous_bundle, texts, profile, original_texts=original_texts)
    x = stack_features(rows, sparse, neural, previous)
    return candidate_score(setting['selected'], artifact['gates'][profile], x, sparse, neural, previous)
