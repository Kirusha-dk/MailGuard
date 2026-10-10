#!/usr/bin/env python3
"""TRAIN-only maximum-margin text rescue, preserving frozen v51 decisions."""
import argparse
import gc
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
import torch
from scipy.special import expit
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from benchmark_v48_calibrated import prepare, arrays, features
from rspamd_benchmark_audit import load_frozen_v40
from train_v47_phishing_aware import metrics_at
from train_v50_neural import ByteCNN, predict_neural
from train_v51_residual import predict_scores, residual_threshold, scored_union


def nb_ratio(x, y):
    p = np.asarray(x[y == 1].sum(axis=0)).ravel() + 1.
    q = np.asarray(x[y == 0].sum(axis=0)).ravel() + 1.
    return np.log((p / p.sum()) / (q / q.sum()))


def margin_probability(model, x):
    # Monotonic rank transform, not a calibrated spam probability.
    return expit(np.clip(model.decision_function(x), -20., 20.))


def transform_rescue(artifact, raw):
    x = artifact['rescueTfidf'].transform(raw)
    ratio = artifact.get('rescueRatio')
    if ratio is not None:
        x = normalize(x.multiply(ratio).tocsr(), copy=False)
    return x


def load_components(root):
    artifact = joblib.load(Path(root) / 'model.joblib')
    neural = ByteCNN()
    neural.load_state_dict(torch.load(Path(root) / 'neural.pt', map_location='cpu', weights_only=True)['state'])
    return artifact, neural


def predict_margin_union(artifact, neural, texts):
    raw = features(texts)
    base = artifact['protectedArtifact']
    sparse = base['model'].predict_proba(base['tfidf'].transform(raw))[:, 1]
    protected = predict_scores(base, sparse, predict_neural(neural, texts)) >= base['threshold']
    rescue = artifact.get('rescueModel')
    scores = np.zeros(len(texts)) if rescue is None else margin_probability(rescue, transform_rescue(artifact, raw))
    return scored_union(protected, scores, artifact['threshold'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--output', default='reports/v54-smoke')
    args = ap.parse_args()
    if args.limit < 0:
        ap.error('Invalid limit')
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    paths = list(Path('.cache/v51-artifact').rglob('v51-full/model.joblib'))
    if len(paths) != 1:
        raise RuntimeError('Expected exactly one frozen v51 artifact')
    baseline = joblib.load(paths[0])
    neural = ByteCNN()
    neural.load_state_dict(torch.load(paths[0].parent / 'neural.pt', map_location='cpu', weights_only=True)['state'])
    frozen, audit = load_frozen_v40('.cache/v40-fresh50k')
    data, dedup = prepare(frozen, args.limit)
    train = data['train']
    select = [r for r in data['val'] if int(r['identity'][:8], 16) % 2 == 0]
    cal = [r for r in data['val'] if int(r['identity'][:8], 16) % 2 == 1]
    yt, _ = arrays(train)
    ys, ss = arrays(select)
    yc, cs = arrays(cal)
    for name, y in [('train', yt), ('selection', ys), ('calibration', yc)]:
        if set(y) != {0, 1}:
            raise ValueError(f'{name} needs both classes')
    tfidf = TfidfTransformer(sublinear_tf=True)
    xt = tfidf.fit_transform(features([r['ngram_text'] for r in train]))
    raws = features([r['ngram_text'] for r in select])
    xs = tfidf.transform(raws)
    def protected(rows, raw):
        texts = [r['ngram_text'] for r in rows]
        sparse = baseline['model'].predict_proba(baseline['tfidf'].transform(raw))[:, 1]
        return predict_scores(baseline, sparse, predict_neural(neural, texts)) >= baseline['threshold']
    bs = protected(select, raws)
    base_metric = metrics_at(ys, bs.astype(float), .5, ss)
    best_key = (base_metric['recall'], -base_metric['fp'])
    chosen = dict(name='v51-fallback', model=None, ratio=None, metric=base_metric)
    candidates = []
    for representation in ('tfidf', 'nb-tfidf'):
        ratio = nb_ratio(xt, yt) if representation == 'nb-tfidf' else None
        xtrain = normalize(xt.multiply(ratio).tocsr(), copy=False) if ratio is not None else xt
        xselect = normalize(xs.multiply(ratio).tocsr(), copy=False) if ratio is not None else xs
        for c in (.1, 1., 4.):
            model = LinearSVC(C=c, class_weight={0: 2., 1: 1.}, dual='auto',
                              max_iter=4000, tol=1e-4, random_state=54)
            model.fit(xtrain, yt)
            p = margin_probability(model, xselect)
            threshold, inherited = residual_threshold(ys, bs, p, int(len(select) * .0012))
            m = metrics_at(ys, scored_union(bs, p, threshold), threshold, ss)
            name = f'{representation}-C{c}'
            candidates.append(dict(name=name, metric=m, inheritedFP=inherited))
            print('v54 selection', name, m['recall'], m['fp'], flush=True)
            key = (m['recall'], -m['fp'])
            if key > best_key:
                best_key = key
                chosen = dict(name=name, model=model, ratio=ratio, metric=m)
        del xtrain, xselect
        gc.collect()
    del xt, xs, raws
    gc.collect()
    rawc = features([r['ngram_text'] for r in cal])
    bc = protected(cal, rawc)
    artifact = dict(baseline, version='v54-margin-rescue', protectedArtifact=baseline,
                    rescueModel=chosen['model'], rescueTfidf=tfidf, rescueRatio=chosen['ratio'])
    pc = np.zeros(len(cal)) if chosen['model'] is None else margin_probability(chosen['model'], transform_rescue(artifact, rawc))
    threshold, inherited = residual_threshold(yc, bc, pc, int(len(cal) * .0012))
    if chosen['model'] is None:
        threshold = 2.
    artifact['threshold'] = threshold
    shutil.copyfile(paths[0].parent / 'neural.pt', out / 'neural.pt')
    joblib.dump(artifact, out / 'model.joblib', compress=3)
    test = data['test']
    ye, se = arrays(test)
    # Batch to keep both sparse matrices and neural encodings bounded.
    scores = np.concatenate([predict_margin_union(artifact, neural, [r['ngram_text'] for r in test[i:i+1000]])
                             for i in range(0, len(test), 1000)])
    m = metrics_at(ye, scores, threshold, se)
    report = dict(version=54, selected=chosen['name'], selection=chosen['metric'], candidates=candidates,
                  calibration=metrics_at(yc, scored_union(bc, pc, threshold), threshold, cs),
                  calibrationInheritedFP=inherited, test=m, audit=audit, dedup=dedup,
                  estimatorRefitAfterCalibration=False, protectedV51Decisions=True,
                  warning='Reused engineering data; no customer mail or independent production evidence.')
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    summary = ('# MailGuard v54 margin rescue\n\n' + report['warning'] + '\n\n'
               f"Selected: {chosen['name']}\n\n| N | Recall | FP | FN |\n|---:|---:|---:|---:|\n"
               f"| {len(test)} | {m['recall']:.2%} | {m['fp']} | {m['fn']} |\n")
    (out / 'report.md').write_text(summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
