"""Offline v59 inference, with the original parser retained for frozen scores."""
from pathlib import Path
import joblib
import numpy as np
from train_v55_multilingual import file_sha, predict, views
from train_v56_consensus import load_components, predict_margin_union, gate_features, candidate_score


def prior_decisions(frozen, original_rows, old, stream, old_threshold, profile):
    cfg = frozen['profiles'][profile]
    x = gate_features(original_rows, old, stream, old_threshold)
    rescue = candidate_score(cfg['selected'], frozen['gates'], x, old, stream, old_threshold)
    return (np.asarray(old) >= old_threshold) | (rescue >= cfg['threshold'])


def specialist_scores(x, new, prior):
    # Route CJK text through the model that has actually learned CJK ham.
    # Other scripts keep the frozen v57 positive vote; calibration still counts
    # every inherited FP. No source name or label participates in routing.
    return np.where((x[:, 6] <= .25) & np.asarray(prior), 2., new).astype(np.float64)


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
    if selected == 'cjk-specialist':
        prior = prior_decisions(artifact['priorGate'], original_rows, ref, stream_scores, old['threshold'], profile)
        return specialist_scores(x, new, prior)
    return artifact['gates'][selected].predict_proba(x)[:, 1]
