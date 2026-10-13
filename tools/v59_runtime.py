"""Offline v59 inference, with the original parser retained for frozen scores."""
from pathlib import Path
import joblib
import numpy as np
from train_v55_multilingual import file_sha, predict, views
from train_v56_consensus import load_components, predict_margin_union, gate_features


def load_bundle(root):
    root = Path(root)
    artifact = joblib.load(root / 'model.joblib')
    if artifact.get('version') != 'v59-multilingual-ham':
        raise ValueError('Expected v59 model')
    for name, digest in artifact['components'].items():
        if file_sha(root / name) != digest:
            raise ValueError(f'Component hash mismatch: {name}')
    old, neural = load_components(root / 'v54')
    stream = joblib.load(root / 'v55/model.joblib')
    return artifact, old, neural, stream


def predict_texts(bundle, texts, profile='conservative', original_texts=None):
    artifact, old, neural, stream = bundle
    rows = [dict(zip(('focus', 'context'), views(text))) for text in texts]
    core = artifact['core']
    new = predict(core['model'], rows, core['idf'], core['ratio']).astype(np.float64)
    selected = artifact['profiles'][profile]['selected']
    if selected == 'new-only':
        return new
    originals = texts if original_texts is None else original_texts
    if len(originals) != len(texts):
        raise ValueError('Original text batch differs')
    ref = predict_margin_union(old, neural, originals).astype(np.float64)
    original_rows = [dict(zip(('focus', 'context'), views(text))) for text in originals]
    stream_scores = predict(stream['model'], original_rows, stream['idf'], stream['ratio'])
    x = np.column_stack([gate_features(rows, ref, new, old['threshold']), stream_scores])
    return artifact['gates'][selected].predict_proba(x)[:, 1]
