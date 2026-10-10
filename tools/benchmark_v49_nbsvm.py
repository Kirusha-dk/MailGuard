#!/usr/bin/env python3
"""TRAIN-only NB feature ratios; independently selected and calibrated experts."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from benchmark_v48_calibrated import features, prepare, arrays, threshold_for_budget
from rspamd_benchmark_audit import load_frozen_v40
from train_v47_phishing_aware import metrics_at


def nb_ratio(x, labels):
    """Smoothed log likelihood ratio, fit only on training labels."""
    labels = np.asarray(labels)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError('Both training classes are required')
    spam = np.asarray(x[labels == 1].sum(axis=0)).ravel() + 1.
    ham = np.asarray(x[labels == 0].sum(axis=0)).ravel() + 1.
    return np.log((spam / spam.sum()) / (ham / ham.sum())).astype(np.float32)


def transform(artifact, texts):
    x = artifact['tfidf'].transform(features(texts))
    ratio = artifact.get('nbRatio')
    return x.multiply(ratio).tocsr() if ratio is not None else x


def assess(rows, scores, threshold):
    y, sources = arrays(rows)
    return metrics_at(y, scores, threshold, sources)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--output', default='reports/v49-smoke')
    parser.add_argument('--v48-root', default='.cache/v48-artifact')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frozen, fingerprint = load_frozen_v40('.cache/v40-fresh50k')
    data, dedup = prepare(frozen, args.limit)
    train, val, test = (data[k] for k in ('train', 'val', 'test'))
    select = [r for r in val if int(r['identity'][:8], 16) % 2 == 0]
    calibration = [r for r in val if int(r['identity'][:8], 16) % 2 == 1]
    yt, _ = arrays(train)
    ys, ss = arrays(select)
    yc, sc = arrays(calibration)
    counts = features([r['ngram_text'] for r in train])
    ratio = nb_ratio(counts, yt)
    tfidf = TfidfTransformer(sublinear_tf=True)
    xt = tfidf.fit_transform(counts).multiply(ratio).tocsr()
    xs = tfidf.transform(features([r['ngram_text'] for r in select])).multiply(ratio).tocsr()
    # Keep a stricter 60% development allowance relative to 100 FP / 100k total mail.
    rate = .6 * 100 / 100000
    selection_budget = int(len(select) * rate)
    candidates = []
    for c, hp in ((.3, 1.), (1., 1.), (4., 1.), (1., .5)):
        model = LogisticRegression(C=c, solver='liblinear', max_iter=400,
                                   class_weight={0: hp, 1: 1.}, random_state=49)
        model.fit(xt, yt)
        if np.any(model.n_iter_ >= model.max_iter):
            raise RuntimeError('Candidate failed to converge')
        scores = model.predict_proba(xs)[:, 1]
        threshold = threshold_for_budget(ys, scores, selection_budget)
        metric = metrics_at(ys, scores, threshold, ss)
        name = f'NB-C{c}-ham{hp}'
        artifact = dict(version='v49-nbsvm', tfidf=tfidf, nbRatio=ratio,
                        model=model, featureVersion='v47-canonical-v48-hash', name=name)
        candidates.append((metric, artifact))
        print('v49 selection', name, metric, flush=True)
    # Full mode includes the unchanged v48 model as a no-regression candidate.
    if not args.limit:
        files = list(Path(args.v48_root).rglob('v48-full/model.joblib'))
        if len(files) != 1:
            raise RuntimeError('Missing frozen v48 baseline')
        baseline = joblib.load(files[0])
        baseline['name'] = 'v48-baseline-recalibrated-on-validation'
        p = baseline['model'].predict_proba(transform(baseline, [r['ngram_text'] for r in select]))[:, 1]
        t = threshold_for_budget(ys, p, selection_budget)
        candidates.append((metrics_at(ys, p, t, ss), baseline))
    selected_metric, artifact = max(candidates, key=lambda z: (z[0]['recall'], -z[0]['fp']))
    pc = artifact['model'].predict_proba(transform(artifact, [r['ngram_text'] for r in calibration]))[:, 1]
    threshold = threshold_for_budget(yc, pc, int(len(calibration) * rate))
    artifact['threshold'] = threshold
    # Freeze before engineering test scores. Never refit on calibration or errors.
    joblib.dump(artifact, out / 'model.joblib', compress=3)
    pt = artifact['model'].predict_proba(transform(artifact, [r['ngram_text'] for r in test]))[:, 1]
    test_metric = assess(test, pt, threshold)
    report = dict(version=49, chosen=artifact['name'], goal=dict(recall=.98, maxFp=100, count=100000),
                  selection=selected_metric, calibration=assess(calibration, pc, threshold),
                  candidates=[dict(name=a['name'], metrics=m) for m, a in candidates],
                  test=test_metric, count=len(test), dedup=dedup, fingerprint=fingerprint,
                  modelRefitAfterCalibration=False,
                  warning='Reused engineering test; 98% production quality is not established.')
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    text = ('# MailGuard v49 NB feature experts\n\n' + report['warning'] + '\n\n'
            f"Selected: {artifact['name']}\n\n| N | Recall | FP | FN |\n|---:|---:|---:|---:|\n"
            f"| {len(test)} | {test_metric['recall']:.2%} | {test_metric['fp']} | {test_metric['fn']} |\n")
    (out / 'report.md').write_text(text)
    print(text, flush=True)


if __name__ == '__main__':
    main()
