#!/usr/bin/env python3
"""Frozen train -> model selection -> calibration -> reused engineering test.

No external inference API. The selected estimator is never refit after calibration.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

from rspamd_benchmark_audit import load_frozen_v40
from train_v47_phishing_aware import from_eml, metrics_at


def threshold_for_budget(labels, scores, budget):
    """Smallest score boundary satisfying an exact ham budget, including ties."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or budget < 0:
        raise ValueError('Invalid scores or budget')
    ham = np.sort(scores[labels == 0])[::-1]
    if not len(ham):
        raise ValueError('Calibration requires ham')
    if budget >= len(ham):
        return float('-inf')
    return float(np.nextafter(ham[budget], np.inf))


def features(texts):
    return hstack([
        HashingVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                          n_features=2**19, alternate_sign=False, norm=None).transform(texts),
        HashingVectorizer(ngram_range=(1, 2), n_features=2**18,
                          alternate_sign=False, norm=None).transform(texts),
    ], format='csr')


def prepare(data, limit):
    result, seen, audit = {}, {}, {}
    for split in ('train', 'val', 'test'):
        rows, dropped, conflicts = [], 0, 0
        for r in data[split]:
            c = from_eml(r['path'].read_bytes())
            key = c['identity']
            if key in seen:
                dropped += 1
                conflicts += seen[key] != r['y']
                continue
            seen[key] = r['y']
            rows.append(dict(c, y=r['y'], source=r['source']))
        # Stable sampling independent of labels; full mode retains every unique row.
        rows.sort(key=lambda r: r['identity'])
        result[split] = rows[:limit] if limit else rows
        audit[split] = dict(available=len(rows), evaluated=len(result[split]),
                            duplicateContent=dropped, conflictingLabels=conflicts)
    return result, audit


def arrays(rows):
    return np.asarray([r['y'] for r in rows]), [r['source'] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=1000,
                    help='Fast iteration size per split; 0 uses all frozen rows')
    ap.add_argument('--max-fp', type=int, default=100)
    ap.add_argument('--test-size', type=int, default=50000)
    ap.add_argument('--output', default='reports/v48')
    args = ap.parse_args()
    if args.limit < 0 or args.max_fp < 0 or args.test_size <= 0:
        ap.error('Invalid size or budget')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frozen, fingerprint = load_frozen_v40('.cache/v40-fresh50k')
    data, dedup = prepare(frozen, args.limit)
    train, val, test = (data[k] for k in ('train', 'val', 'test'))
    # Hash partition is label-blind and fixed before scoring.
    select = [r for r in val if int(r['identity'][:8], 16) % 2 == 0]
    calibrate = [r for r in val if int(r['identity'][:8], 16) % 2 == 1]
    for name, rows in [('train', train), ('selection', select), ('calibration', calibrate), ('test', test)]:
        if set(r['y'] for r in rows) != {0, 1}:
            raise ValueError(f'{name} must contain both classes')
    tfidf = TfidfTransformer(sublinear_tf=True)
    xt = tfidf.fit_transform(features([r['ngram_text'] for r in train]))
    xs = tfidf.transform(features([r['ngram_text'] for r in select]))
    yt, _ = arrays(train)
    ys, ss = arrays(select)
    # Budget is per all 50k messages, not an assumed fixed 25k ham count.
    # Development allows 60% of that rate to leave margin for sampling variation.
    dev_rate = .6 * args.max_fp / args.test_size
    selection_budget = int(len(select) * dev_rate)
    candidates = []
    for alpha in (1e-6, 5e-6, 2e-5):
        model = SGDClassifier(loss='log_loss', alpha=alpha, max_iter=80,
                              tol=1e-4, average=True, random_state=48)
        model.fit(xt, yt)
        ps = model.predict_proba(xs)[:, 1]
        threshold = threshold_for_budget(ys, ps, selection_budget)
        metric = metrics_at(ys, ps, threshold, ss)
        candidates.append((metric['recall'], -metric['fp'], alpha, model, metric))
        print('v48 selection', alpha, metric, flush=True)
    _, _, alpha, model, selected = max(candidates, key=lambda c: c[:2])
    yc, sc = arrays(calibrate)
    pc = model.predict_proba(tfidf.transform(features([r['ngram_text'] for r in calibrate])))[:, 1]
    calibration_budget = int(len(calibrate) * dev_rate)
    threshold = threshold_for_budget(yc, pc, calibration_budget)
    calibration = metrics_at(yc, pc, threshold, sc)
    # Freeze this exact estimator and threshold before reading test scores.
    artifact = dict(version='v48-calibrated', model=model, tfidf=tfidf,
                    threshold=threshold, alpha=alpha, featureVersion='v47-canonical-v48-hash')
    joblib.dump(artifact, out / 'model.joblib', compress=3)
    ye, se = arrays(test)
    pe = model.predict_proba(tfidf.transform(features([r['ngram_text'] for r in test])))[:, 1]
    evaluation = metrics_at(ye, pe, threshold, se)
    with (out / 'predictions.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['identity', 'source', 'label', 'score', 'predictedSpam', 'error'])
        for r, p in zip(test, pe):
            pred = int(p >= threshold)
            writer.writerow([r['identity'], r['source'], r['y'], float(p), pred,
                             'FP' if pred and not r['y'] else 'FN' if r['y'] and not pred else ''])
    report = dict(version=48, scope='standalone text classifier; no Rspamd OR',
                  warning='Reused historical engineering test, not independent production evidence.',
                  fingerprint=fingerprint, canonicalDedup=dedup,
                  counts={k: dict(Counter(str(r['y']) for r in v)) for k, v in data.items()},
                  goal=dict(testSize=args.test_size, maxFp=args.max_fp, recall=.9),
                  selection=selected, calibration=calibration, test=evaluation,
                  full50kGate=(len(test) == args.test_size and evaluation['recall'] >= .9
                               and evaluation['fp'] <= args.max_fp),
                  estimatorRefitAfterCalibration=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    text = ('# MailGuard v48\n\n' + report['warning'] + '\n\n'
            '| Test messages | Spam recall | FP | FN |\n|---:|---:|---:|---:|\n'
            f"| {len(test)} | {evaluation['recall']:.2%} | {evaluation['fp']} | {evaluation['fn']} |\n\n"
            f"Full 50k engineering gate passed: {report['full50kGate']}\n")
    (out / 'report.md').write_text(text)
    print(text, flush=True)


if __name__ == '__main__':
    main()
