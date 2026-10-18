"""Fit English experts on trusted development sources; freeze before GenSpam.

The whole GenSpam provider is protected from fitting. The label-conflicted v76/v77
external provider is not used for fitting, calibration or reported quality.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
import v77_mail as core
import v78_mail as runtime
from v78_data import prepare


def source_weights(rows):
    counts = Counter((r['source'], r['y']) for r in rows)
    weights = np.asarray([min(12., 5000 / counts[r['source'], r['y']]) for r in rows])
    return weights / weights.mean()


def char_subset(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['source'], []).append(row)
    return [r for values in groups.values() for r in values[:4000]]


def fit_expert(rows, kind):
    if kind == 'char':
        rows = char_subset(rows)
    weights = source_weights(rows)
    labels = np.asarray([r['y'] for r in rows], dtype=np.int8)
    if kind == 'tfidf':
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=220000,
            min_df=2, sublinear_tf=True, dtype=np.float32, lowercase=False)
        x = vectorizer.fit_transform(core.content_view(r['text'], 'word') for r in rows)
        extra = {'vectorizer': vectorizer}
    else:
        vectorizer = core.vectorizer(kind)
        chunks = [vectorizer.transform(core.content_view(r['text'], kind) for r in rows[i:i + 256])
                  for i in range(0, len(rows), 256)]
        x = vstack(chunks, format='csr'); del chunks
        p = np.asarray(x[labels == 1].multiply(weights[labels == 1, None]).sum(axis=0)).ravel() + 2
        q = np.asarray(x[labels == 0].multiply(weights[labels == 0, None]).sum(axis=0)).ravel() + 2
        ratio = (np.log(p / p.sum()) - np.log(q / q.sum())).astype(np.float32)
        x = normalize(x.multiply(ratio).tocsr(), copy=False)
        extra = {'ratio': ratio, 'kind': kind}
    classifier = LinearSVC(C=2., dual=True, max_iter=3000, tol=.001, random_state=78)
    classifier.fit(x, labels, sample_weight=weights)
    print('Trained', kind, len(rows), 'rows', x.nnz, 'nonzeros', flush=True)
    return {'model': classifier, 'training_rows': len(rows), **extra}


def cutoff(values, truth, budget):
    values, truth = np.asarray(values), np.asarray(truth)
    if values.shape != truth.shape or not np.isfinite(values).all() or not set(np.unique(truth)) <= {0, 1}:
        raise ValueError('Invalid calibration scores or labels')
    if not 0 <= budget < 1:
        raise ValueError('Invalid calibration FP budget')
    ham = np.sort(values[truth == 0])[::-1]
    if len(ham) < 500:
        raise ValueError('Insufficient calibration HAM')
    allowance = int(len(ham) * budget)
    return float(np.nextafter(ham[allowance], np.inf))


def choose_on_select(scores, rows, budget):
    truth = np.asarray([r['y'] for r in rows]); options = []
    for name, values in scores.items():
        threshold = cutoff(values, truth, budget)
        metric = core.measure(values, truth, threshold)
        source_recalls = []
        for source in sorted({r['source'] for r in rows}):
            ix = np.array([i for i, r in enumerate(rows) if r['source'] == source and r['y'] == 1], dtype=int)
            if len(ix) >= 25:
                source_recalls.append(float(np.mean(values[ix] >= threshold)))
        key = (min(source_recalls), metric['recall'], -metric['fp'])
        options.append((key, name, threshold, metric))
    # Candidate insertion order provides deterministic ties.
    best = max(enumerate(options), key=lambda entry: (entry[1][0], -entry[0]))[1]
    return {'candidate': best[1], 'select_cutoff': best[2], 'selection': best[3],
            'worst_select_source_recall': best[0][0], 'calibration_fpr_budget': budget}


def evaluate(bundle, rows):
    scores = runtime.candidates(bundle, [r['text'] for r in rows])
    truth = [r['y'] for r in rows]
    return {name: core.measure(scores[cfg['candidate']], truth, cfg['cutoff'])
            for name, cfg in bundle['policies'].items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', default='.cache/v78/data.json.gz')
    p.add_argument('--source-archive', help='Optional local ZIP containing the two pinned public archives')
    p.add_argument('--output', default='reports/v78')
    p.add_argument('--baseline', default='reports/v77-frozen/model.joblib')
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    data = prepare(args.cache, archive_path=args.source_archive)
    manifest = data['provenance']
    (out / 'frozen.json').write_text(json.dumps(manifest, indent=2))
    parts = data['development']
    bundle = {'version': runtime.VERSION, 'experts': {}, 'policies': {}, 'manifest': manifest}
    for kind in ('word', 'char', 'tfidf'):
        bundle['experts'][kind] = fit_expert(parts['train'], kind)
        # Reproducible checkpoint: completed expert can be inspected if a later fit fails.
        joblib.dump(bundle['experts'][kind], out / (kind + '.joblib'), compress=3)
    selection = runtime.candidates(bundle, [r['text'] for r in parts['select']])
    calibration = runtime.candidates(bundle, [r['text'] for r in parts['cal']])
    cal_truth = [r['y'] for r in parts['cal']]
    for name, budget in runtime.PROFILES.items():
        policy = choose_on_select(selection, parts['select'], budget)
        candidate = policy['candidate']
        policy['cutoff'] = max(policy['select_cutoff'], cutoff(calibration[candidate], cal_truth, budget))
        policy['calibration'] = core.measure(calibration[candidate], cal_truth, policy['cutoff'])
        bundle['policies'][name] = policy
    modelpath = out / 'model.joblib'
    joblib.dump(bundle, modelpath, compress=3)
    sha = hashlib.sha256(modelpath.read_bytes()).hexdigest()
    print('MODEL AND POLICIES LOCKED BEFORE EXTERNAL TEST', sha, json.dumps(bundle['policies']), flush=True)
    test = data['test']
    metrics = evaluate(bundle, test)
    baseline = None
    if args.baseline and Path(args.baseline).is_file():
        old = joblib.load(args.baseline)
        scores = core.candidates(old['experts']['en'], [r['text'] for r in test])
        truth = [r['y'] for r in test]
        baseline = {name: core.measure(scores[cfg['candidate']], truth, cfg['cutoff'])
                    for name, cfg in old['policies']['en'].items()}
    if sha != hashlib.sha256(modelpath.read_bytes()).hexdigest():
        raise ValueError('Evaluation changed the model')
    report = {'version': runtime.VERSION, 'model_sha256': sha, 'manifest': manifest,
        'policies': bundle['policies'], 'test': metrics, 'v77_same_test': baseline,
        'test_is_10000_english_emails': len(test) == 10000,
        'recall80_observed_on_this_source': {k: v['recall'] >= .8 for k, v in metrics.items()},
        'test_fpr_within_calibration_budget': {k: v['fpr'] <= runtime.PROFILES[k] for k, v in metrics.items()},
        'recall80_on_arbitrary_mail_guaranteed': False,
        'warning': 'Entirely held-out historical GenSpam source, not live customer mail. Old public source; semantic overlap cannot be completely ruled out.'}
    (out / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
