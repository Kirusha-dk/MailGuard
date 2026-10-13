#!/usr/bin/env python3
"""Train on multilingual ham, compare only after locking model and thresholds."""
import argparse
import csv
import gc
import json
import shutil
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import save_npz, load_npz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier

from benchmark_v48_calibrated import threshold_for_budget
from train_v47_phishing_aware import metrics_at
from train_v55_multilingual import FEATURES, corpus_rows, features, file_sha, predict, transformed, views
from train_v56_consensus import load_components, predict_margin_union, gate_features, candidate_score

PROFILES = {'conservative': .0002, 'high-recall': .002}


def group(source):
    return 'chinese' if source == 'TREC06-Chinese' else 'archive' if source.startswith('SpamArchive-') else 'legacy'


def metrics(rows, scores, threshold):
    return metrics_at(np.asarray([r['y'] for r in rows]), np.asarray(scores), threshold, [r['source'] for r in rows])


def threshold(rows, scores, fpr):
    y = np.asarray([r['y'] for r in rows])
    return float(threshold_for_budget(y, np.asarray(scores, dtype=np.float64), int((y == 0).sum() * fpr)))


def score_batches(core, rows):
    return np.concatenate([predict(core['model'], rows[i:i+256], core['idf'], core['ratio'])
                           for i in range(0, len(rows), 256)]).astype(np.float64)


def macro_recall(rows, scores, cutoff):
    groups = np.asarray([group(r['source']) for r in rows])
    values = []
    for name in sorted(set(groups)):
        ix = np.flatnonzero(groups == name)
        if any(rows[i]['y'] for i in ix):
            values.append(metrics([rows[i] for i in ix], scores[ix], cutoff)['recall'])
    return (min(values), float(np.mean(values)))


def reference_scores(rows, old, neural, old_stream, cached):
    old_scores, stream_scores = np.empty(len(rows)), np.empty(len(rows))
    for start in range(0, len(rows), 128):
        batch = rows[start:start+128]
        old_scores[start:start+len(batch)] = [cached[r['identity']] if r['identity'] in cached else np.nan for r in batch]
        positions = [i for i, r in enumerate(batch) if r['identity'] not in cached]
        if positions:
            if any('text' not in batch[i] for i in positions):
                raise ValueError('Missing raw canonical reference input')
            old_scores[start + np.asarray(positions)] = predict_margin_union(old, neural, [batch[i].get('old_text', batch[i]['text']) for i in positions])
        # The frozen v55 baseline receives its original parser representation.
        old_views = []
        for r in batch:
            if 'old_text' in r:
                focus, context = views(r['old_text']); old_views.append(dict(focus=focus, context=context))
            else:
                old_views.append(r)
        stream_scores[start:start+len(batch)] = predict(old_stream['model'], old_views, old_stream['idf'], old_stream['ratio'])
        if start % 4096 == 0:
            print('Reference inference', start, '/', len(rows), flush=True)
    return old_scores, stream_scores


def main():
    import torch
    torch.set_num_threads(4)
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='.cache/v59-data')
    p.add_argument('--output', default='reports/v59-full')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--feature-cache', default='.cache/v59-feature-cache')
    args = p.parse_args()
    data, out = Path(args.data), Path(args.output); out.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((data / 'provenance.json').read_text())
    for split, digest in provenance['splitHashes'].items():
        if file_sha(data / f'{split}.jsonl.gz') != digest:
            raise ValueError('Locked split hash mismatch')
    train, selection, cal = [list(corpus_rows(data / f'{s}.jsonl.gz')) for s in ('train', 'select', 'cal')]
    if any(r.get('weakLabel') for r in selection + cal):
        raise AssertionError('Weak label outside TRAIN')
    # Equalize large source x class groups; weak Japanese ham has limited weight.
    counts = Counter((r['source'], r['y']) for r in train)
    weights = np.asarray([min(1., 12000 / counts[r['source'], r['y']]) * (6. if r['y'] == 0 else 1.)
                          * (.25 if r.get('weakLabel') else 1.) for r in train], dtype=np.float64)
    rng = np.random.default_rng(59)
    order = rng.permutation(len(train))
    train = [train[i] for i in order]
    weights = weights[order]
    y_train = np.asarray([r['y'] for r in train])
    cache = Path(args.feature_cache); cache.mkdir(parents=True, exist_ok=False)
    df = np.zeros(FEATURES, dtype=np.float64)
    totals = [np.zeros(FEATURES, dtype=np.float64) for _ in (0, 1)]
    for start in range(0, len(train), 256):
        raw = features(train[start:start+256]); y = np.asarray([r['y'] for r in train[start:start+256]])
        save_npz(cache / f'{start:08d}.npz', raw, compressed=True)
        df += np.bincount(raw.indices, minlength=FEATURES)
        for label in (0, 1):
            # Frequency ratios also respect the source caps, without the HAM cost multiplier.
            w = weights[start:start+len(y)][y == label] / (6. if label == 0 else 1.)
            totals[label] += np.asarray(raw[y == label].multiply(w[:, None]).sum(axis=0)).ravel()
        if start % 8192 == 0:
            print('TRAIN feature cache', start, '/', len(train), flush=True)
    idf = (np.log((1 + len(train)) / (1 + df)) + 1).astype(np.float32)
    pos, neg = totals[1] + 1., totals[0] + 1.
    nb = np.log((pos / pos.sum()) / (neg / neg.sum())).astype(np.float32)
    del df, totals, pos, neg
    candidates = [dict(name=f'{rep}-alpha{alpha}', model=SGDClassifier(loss='log_loss', alpha=alpha, average=True, random_state=59),
                       idf=idf, ratio=nb if rep == 'nb' else None)
                  for rep in ('tfidf', 'nb') for alpha in (1e-5, 3e-6)]
    for epoch in range(args.epochs):
        # Initial global shuffle mixes sources/classes; each epoch also shuffles
        # blocks and rows. Only TRAIN features are cached, never TEST features.
        blocks = rng.permutation(np.arange(0, len(train), 256))
        for start in blocks:
            raw = load_npz(cache / f'{start:08d}.npz')
            within = rng.permutation(raw.shape[0])
            raw = raw[within]
            ix = start + within; y = y_train[ix]
            for rep in (None, nb):
                x = transformed(raw, idf, rep)
                for c in candidates:
                    if c['ratio'] is rep:
                        c['model'].partial_fit(x, y, classes=np.asarray([0, 1]), sample_weight=weights[ix])
        print('v59 core epoch', epoch+1, '/', args.epochs, 'rows', len(train), flush=True)
    shutil.rmtree(cache)
    # Use only the model-selection portion here; gate TRAIN has disjoint families.
    fit_ix = np.asarray([i for i, r in enumerate(selection) if int(r['family'][-12:], 16) % 10 < 6])
    choose_ix = np.asarray([i for i, r in enumerate(selection) if int(r['family'][-12:], 16) % 10 >= 6])
    choose_rows = [selection[i] for i in choose_ix]
    core_results = []
    for c in candidates:
        scores = score_batches(c, choose_rows)
        cut = threshold(choose_rows, scores, PROFILES['high-recall'])
        c['key'] = macro_recall(choose_rows, scores, cut)
        core_results.append(dict(name=c['name'], selection=metrics(choose_rows, scores, cut), macroRecall=c['key']))
        print('Core selection:', json.dumps(core_results[-1]), flush=True)
    core = max(candidates, key=lambda c: c['key'])
    old_root = Path('.cache/v56/v56-full')
    old, neural = load_components(old_root / 'v54')
    old_stream = joblib.load(old_root / 'v55/model.joblib')
    with Path('.cache/v54-evidence/predictions.csv').open() as f:
        cached = {r['identity']: float(r['score']) for r in csv.DictReader(f)}
    # Legacy validation lacks its canonical text in compact v55 rows. Restore it only for reference inference.
    from rspamd_benchmark_audit import load_frozen_v40
    from train_v47_phishing_aware import from_eml as old_parse
    frozen, _ = load_frozen_v40('.cache/v40-fresh50k')
    wanted = {r['identity']: r for r in selection + cal if r['identity'] not in cached and 'text' not in r}
    for entry in frozen['val']:
        parsed = old_parse(entry['path'].read_bytes())
        if parsed['identity'] in wanted:
            wanted[parsed['identity']]['text'] = parsed['ngram_text']
    del frozen, wanted, train, weights, candidates
    gc.collect()
    ref_select, stream_select = reference_scores(selection, old, neural, old_stream, cached)
    new_select = score_batches(core, selection)
    x = gate_features(selection, ref_select, new_select, old['threshold'])
    # Use the previously trained multilingual score too, while allowing old positives to be rejected.
    x = np.column_stack([x, stream_select])
    gates = {}
    labels = np.asarray([r['y'] for r in selection])
    for ham_weight in (4., 16.):
        name = f'gate-ham{int(ham_weight)}'
        gate = HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=7, min_samples_leaf=40, l2_regularization=8., random_state=59)
        gate.fit(x[fit_ix], labels[fit_ix], sample_weight=np.where(labels[fit_ix] == 0, ham_weight, 1.))
        gates[name] = gate
    profiles, selection_results = {}, []
    ref_cal, stream_cal = reference_scores(cal, old, neural, old_stream, cached)
    new_cal = score_batches(core, cal)
    xc = np.column_stack([gate_features(cal, ref_cal, new_cal, old['threshold']), stream_cal])
    for profile, fpr in PROFILES.items():
        options = []
        for name in ('new-only', *gates):
            scores = new_select[choose_ix] if name == 'new-only' else gates[name].predict_proba(x[choose_ix])[:, 1]
            cut = threshold(choose_rows, scores, fpr)
            key = macro_recall(choose_rows, scores, cut)
            options.append((key, name, cut))
            selection_results.append(dict(profile=profile, candidate=name, threshold=cut, selection=metrics(choose_rows, scores, cut)))
        _, name, select_cut = max(options)
        scores = new_cal if name == 'new-only' else gates[name].predict_proba(xc)[:, 1]
        # Never relax the selection FP limit when calibrating.
        cut = max(select_cut, threshold(cal, scores, fpr))
        profiles[profile] = dict(selected=name, threshold=cut, targetHamFpr=fpr, calibration=metrics(cal, scores, cut))
    artifact = dict(version='v59-multilingual-ham', core=core, gates=gates, profiles=profiles, protectedV54Decisions=False,
                    sourceWeights='source x class capped at 12000; HAM cost 6; weak Japanese HAM weight .25',
                    splitManifestSha256=provenance['manifestSha256'], components={})
    for folder in ('v54', 'v55'):
        shutil.copytree(old_root / folder, out / folder)
    for name in ('v54/model.joblib', 'v54/neural.pt', 'v55/model.joblib'):
        artifact['components'][name] = file_sha(out / name)
    joblib.dump(artifact, out / 'model.joblib', compress=3)
    locked_model_sha = file_sha(out / 'model.joblib')
    print('MODEL AND THRESHOLDS LOCKED', locked_model_sha, json.dumps(profiles), flush=True)
    from v59_runtime import load_bundle, predict_texts
    bundle = load_bundle(out)
    probe = cal[:32]
    for profile in PROFILES:
        expected = new_cal[:32] if profiles[profile]['selected'] == 'new-only' else gates[profiles[profile]['selected']].predict_proba(xc[:32])[:, 1]
        observed = predict_texts(bundle, [r['text'] for r in probe], profile, [r.get('old_text', r['text']) for r in probe])
        if not np.allclose(expected, observed, rtol=1e-6, atol=1e-7):
            raise AssertionError('Saved runtime differs from training inference')
    # TEST is opened for inference only after all choices are final.
    test = list(corpus_rows(data / 'test.jsonl.gz'))
    if any(r.get('weakLabel') for r in test):
        raise AssertionError('Weak HAM cannot enter benchmark')
    ref_test, stream_test = reference_scores(test, old, neural, old_stream, cached)
    new_test = score_batches(core, test)
    xt = np.column_stack([gate_features(test, ref_test, new_test, old['threshold']), stream_test])
    frozen_gate = joblib.load(old_root / 'model.joblib')
    baseline_scores, scores = {}, {}
    for profile in PROFILES:
        cfg = profiles[profile]
        scores[profile] = new_test if cfg['selected'] == 'new-only' else gates[cfg['selected']].predict_proba(xt)[:, 1]
        cfg57 = frozen_gate['profiles'][profile]
        # Actual frozen v57 fixed OR rule, original decoding and unchanged thresholds.
        original_rows = [dict(focus=views(r['old_text'])[0], context=views(r['old_text'])[1]) for r in test]
        x57 = gate_features(original_rows, ref_test, stream_test, old['threshold'])
        rescue = candidate_score(cfg57['selected'], frozen_gate['gates'], x57, ref_test, stream_test, old['threshold'])
        baseline_scores[profile] = ((ref_test >= old['threshold']) | (rescue >= cfg57['threshold'])).astype(float)
    if locked_model_sha != file_sha(out / 'model.joblib'):
        raise AssertionError('Model changed after TEST was opened')
    result = dict(version=59, testCount=len(test), profiles={}, coreCandidates=core_results, gateSelection=selection_results,
                  modelSha256=locked_model_sha, estimatorAndThresholdLockedBeforeTest=True, testLabelsUsedForTraining=False,
                  oldErrorExamplesAddedToTrain=False, productionGoalValidated=False, provenance=provenance)
    for profile in PROFILES:
        result['profiles'][profile] = dict(configuration=profiles[profile], test=metrics(test, scores[profile], profiles[profile]['threshold']),
                                           v57SameTest=metrics(test, baseline_scores[profile], .5))
    with (out / 'predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['identity', 'family', 'source', 'label', 'conservativeScore', 'highRecallScore', 'v57ConservativeSpam', 'v57HighRecallSpam'])
        for i, row in enumerate(test):
            writer.writerow([row['identity'], row['family'], row['source'], row['y'], scores['conservative'][i], scores['high-recall'][i], int(baseline_scores['conservative'][i]), int(baseline_scores['high-recall'][i])])
    (out / 'report.json').write_text(json.dumps(result, indent=2))
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    shutil.copyfile(data / 'manifest.csv', out / 'split-manifest.csv')
    md = '# MailGuard v59 — new public mail\n\nNew model and thresholds were locked before opening TEST. No old TEST errors were added to training.\n\n'
    md += '| Model/profile | Mail | Ham | Spam recall | FP | FN | FP / ham |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for profile, r in result['profiles'].items():
        for version, metric in (('v57', r['v57SameTest']), ('v59', r['test'])):
            ham = metric['tn'] + metric['fp']
            md += f'| {version} {profile} | {len(test)} | {ham} | {metric["recall"]:.2%} | {metric["fp"]} | {metric["fn"]} | {metric["fpr"]:.4%} |\n'
    md += '\n' + provenance['caveat'] + '\n\nWeak Japanese ham is training-only. This result does not validate customer production quality.\n'
    (out / 'report.md').write_text(md)
    print(md, flush=True)


if __name__ == '__main__':
    main()
