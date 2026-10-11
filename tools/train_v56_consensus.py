#!/usr/bin/env python3
"""Frozen v54/v55 consensus; gate fitting, selection and calibration are separate.

All corpora are reused engineering data. Frozen v54 predates the family split:
its training may contain relatives of legacy holdout mail. This experiment
does not establish independent production quality or multilingual ham safety.
"""
import argparse
import csv
import gc
import gzip
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from benchmark_v48_calibrated import threshold_for_budget
from rspamd_benchmark_audit import load_frozen_v40
from train_v47_phishing_aware import from_eml, metrics_at
from train_v55_multilingual import CORPUS_SHA, compact, corpus_rows, file_sha, predict, views

PROFILES = {'conservative': .0002, 'high-recall': .002}
WARNING = ('Reused, previously inspected engineering data; not an independent production benchmark. '
           'Gate fit, selection, calibration and test families are disjoint. '
           'Frozen v54 predates this split and may have trained on relatives of legacy holdout mail. '
           'No test labels fit the gate or threshold. Multilingual legitimate mail is underrepresented. '
           'A ham FP allowance is a development target, not a guaranteed production rate.')


def load_components(root):
    from train_v54_margin import load_components as load
    return load(root)


def predict_margin_union(artifact, neural, texts):
    from train_v54_margin import predict_margin_union as predict_old
    return predict_old(artifact, neural, texts)


def is_new(source):
    return source.startswith(('SpamArchive-', 'CSDMC'))


def gate_partition(family):
    return 'fit' if int(family[-12:], 16) % 10 < 6 else 'select'


def budget(y, rate):
    return int(np.count_nonzero(np.asarray(y) == 0) * rate)


def gate_features(rows, old, new, old_threshold):
    old, new = np.asarray(old, dtype=np.float64), np.asarray(new, dtype=np.float64)
    def logit(p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))
    structural = []
    for r in rows:
        text = r['context'][:1800]
        head = r['focus'][:1000]
        letters = [c for c in text if c.isalpha()]
        cjk = sum('\u3040' <= c <= '\u30ff' or '\u3400' <= c <= '\u9fff' or '\uac00' <= c <= '\ud7af' for c in letters)
        nonlatin = sum(ord(c) > 127 for c in letters)
        structural.append([cjk / max(1, len(letters)), nonlatin / max(1, len(letters)),
                           np.log1p(len(text)), np.log1p(len(head)),
                           text.count('urltoken'), text.count('emailtoken'),
                           text.count('numbertoken'), head.count('['),
                           int('unsubscribe' in text), int('wrote:' in text),
                           int('forwarded' in text), int('re:' in head[:150])])
    return np.column_stack([old, new, logit(np.minimum(old, 1.)), logit(new),
                            old >= old_threshold, logit(new) - logit(np.minimum(old, 1.)),
                            np.asarray(structural, dtype=np.float64)])


def candidate_score(name, gates, x, old, new, old_threshold):
    if name == 'v55-only':
        return np.asarray(new, dtype=np.float64)
    if name == 'consensus-rank':
        # Continuous ranking retains information inside the frozen positive vote.
        # Calibration can still reject old positives; their FP are not hidden.
        return np.maximum(new, .5 * (np.asarray(old) >= old_threshold) + .5 * new).astype(np.float64)
    return gates[name].predict_proba(x)[:, 1].astype(np.float64)


def load_bundle(root):
    root = Path(root)
    artifact = joblib.load(root / 'model.joblib')
    if artifact.get('version') != 'v56-consensus':
        raise ValueError('Expected a frozen v56 bundle')
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
    return candidate_score(setting['selected'], artifact['gates'], x, old, new, old_artifact['threshold'])


def limited(entries, limit):
    if not limit:
        return entries
    # Identity order and source group only, never labels or classifier errors.
    result = []
    for group in (False, True):
        rows = sorted((r for r in entries if is_new(r['source']) == group), key=lambda r: r['identity'])
        result.extend(rows[:limit // 2])
    return result


def metrics(y, scores, threshold, sources):
    return metrics_at(np.asarray(y), np.asarray(scores, dtype=np.float64), float(threshold), sources)


def balanced_key(y, scores, threshold, sources):
    mask = np.array([is_new(s) for s in sources])
    old = metrics(y[~mask], scores[~mask], threshold, list(np.asarray(sources)[~mask]))
    new = metrics(y[mask], scores[mask], threshold, list(np.asarray(sources)[mask]))
    all_metric = metrics(y, scores, threshold, sources)
    return (min(old['recall'], new['recall']), (old['recall'] + new['recall']) / 2, -all_metric['fp']), old, new, all_metric


def development(args, manifest, baseline_index, frozen, old_artifact, neural, new_artifact):
    entries = {split: limited([r for r in manifest if r['split'] == split], args.limit) for split in ('select', 'cal')}
    wanted = {r['identity']: r for rows in entries.values() for r in rows}
    rows_by_id = {}
    for old in frozen['val']:
        parsed = from_eml(old['path'].read_bytes())
        identity = parsed['identity']
        if identity in wanted:
            entry = wanted[identity]
            row = compact(dict(parsed, y=int(old['y']), source=old['source']), entry['family'])
            row['text'] = parsed['ngram_text']
            rows_by_id[identity] = row
    for raw in corpus_rows(args.corpus):
        if raw['identity'] in wanted:
            entry = wanted[raw['identity']]
            row = compact(raw, entry['family'])
            row['text'] = raw['ngram_text']
            rows_by_id[raw['identity']] = row
    if set(rows_by_id) != set(wanted):
        raise AssertionError('Development manifest reconstruction mismatch')
    outputs = {}
    for split, source_rows in entries.items():
        rows = [rows_by_id[r['identity']] for r in source_rows]
        for r, entry in zip(rows, source_rows):
            if (r['y'], r['source']) != (int(entry['label']), entry['source']):
                raise AssertionError('Development labels or sources differ from frozen manifest')
        old_scores = np.zeros(len(rows), dtype=np.float64)
        old_positions = []
        for i, r in enumerate(rows):
            if is_new(r['source']):
                label, score = baseline_index[r['identity']]
                if label != r['y']:
                    raise AssertionError('v54 cached label mismatch')
                old_scores[i] = score
            else:
                old_positions.append(i)
        for start in range(0, len(old_positions), 128):
            ix = old_positions[start:start + 128]
            old_scores[ix] = predict_margin_union(old_artifact, neural, [rows[i]['text'] for i in ix])
        new_scores = np.concatenate([predict(new_artifact['model'], rows[i:i + 256], new_artifact['idf'], new_artifact['ratio'])
                                     for i in range(0, len(rows), 256)]).astype(np.float64)
        for r in rows:
            if r['family'] != wanted[r['identity']]['family']:
                raise AssertionError('Family mismatch')
        outputs[split] = dict(rows=rows, old=old_scores, new=new_scores,
                              x=gate_features(rows, old_scores, new_scores, old_artifact['threshold']),
                              y=np.asarray([r['y'] for r in rows]), sources=[r['source'] for r in rows])
        print(f'v56 development {split}: {len(rows)} rows', flush=True)
    return outputs


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=1000)
    p.add_argument('--output', default='reports/v56-smoke')
    p.add_argument('--v55-root', default='.cache/v55-artifact/v55-full')
    p.add_argument('--v54-root', default='.cache/v54-artifact/v54-full')
    p.add_argument('--corpus', default='.cache/v54-corpus/selected.jsonl.gz')
    p.add_argument('--baseline-predictions', default='.cache/v54-evidence/predictions.csv')
    args = p.parse_args()
    if args.limit < 0 or args.limit == 1:
        p.error('Limit must be 0 (full) or at least 2')
    torch.set_num_threads(4)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    v55root, v54root = Path(args.v55_root), Path(args.v54_root)
    expected_old = {'model.joblib': '637f521786c72d6dcf74089198ff4cd3c7b6cb5c5c5b764ad837b871d536780b',
                    'neural.pt': '39d930a93866e49a11c01617f6a84714c0c69f2f033bee0051c495ab6e29a72d'}
    for name, digest in expected_old.items():
        if file_sha(v54root / name) != digest:
            raise AssertionError(f'Frozen v54 hash mismatch: {name}')
    prior = json.loads((v55root / 'report.json').read_text())
    if file_sha(v55root / 'model.joblib') != prior['modelSha256'] or file_sha(args.corpus) != CORPUS_SHA:
        raise AssertionError('Frozen v55 model or corpus hash mismatch')
    old_artifact, neural = load_components(v54root)
    new_artifact = joblib.load(v55root / 'model.joblib')
    with (v55root / 'split-manifest.csv').open() as f:
        manifest = list(csv.DictReader(f))
    with Path(args.baseline_predictions).open() as f:
        baseline_index = {r['identity']: (int(r['label']), float(r['score'])) for r in csv.DictReader(f)}
    frozen, audit = load_frozen_v40('.cache/v40-fresh50k')
    data = development(args, manifest, baseline_index, frozen, old_artifact, neural, new_artifact)
    selection, cal = data['select'], data['cal']
    fit_mask = np.array([gate_partition(r['family']) == 'fit' for r in selection['rows']])
    sel_mask = ~fit_mask
    fit_families = {r['family'] for r, m in zip(selection['rows'], fit_mask) if m}
    select_families = {r['family'] for r, m in zip(selection['rows'], sel_mask) if m}
    cal_families = {r['family'] for r in cal['rows']}
    test_entries = limited([r for r in manifest if r['split'] == 'test'], args.limit)
    test_ids = {r['identity'] for r in test_entries}
    test_families = {r['family'] for r in test_entries}
    sets = [fit_families, select_families, cal_families, test_families]
    if any(sets[i] & sets[j] for i in range(4) for j in range(i + 1, 4)):
        raise AssertionError('Gate family leakage')
    for name, labels in [('gate fit', selection['y'][fit_mask]), ('gate selection', selection['y'][sel_mask]), ('calibration', cal['y'])]:
        if set(labels) != {0, 1}:
            raise ValueError(f'{name} needs both classes')
    gates = {}
    for ham_weight in (4., 16.):
        name = f'gate-ham{int(ham_weight)}'
        model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=7, min_samples_leaf=40,
                                               l2_regularization=5., early_stopping=False, random_state=56)
        yf = selection['y'][fit_mask]
        model.fit(selection['x'][fit_mask], yf, sample_weight=np.where(yf == 0, ham_weight, 1.))
        gates[name] = model
    names = ['v55-only', 'consensus-rank'] + list(gates)
    ys = selection['y'][sel_mask]; sources = list(np.asarray(selection['sources'])[sel_mask])
    profiles, candidate_reports = {}, {}
    for profile, rate in PROFILES.items():
        candidates = []
        for name in names:
            scores = candidate_score(name, gates, selection['x'][sel_mask], selection['old'][sel_mask], selection['new'][sel_mask], old_artifact['threshold'])
            t = threshold_for_budget(ys, scores, budget(ys, rate))
            key, old_m, new_m, m = balanced_key(ys, scores, t, sources)
            if m['fp'] > budget(ys, rate):
                raise AssertionError('Selection FP allowance exceeded')
            candidates.append((key, name))
            candidate_reports.setdefault(profile, []).append(dict(name=name, selection=m, legacy=old_m, new=new_m))
            print('v56 selection', profile, name, json.dumps({'oldRecall': old_m['recall'], 'newRecall': new_m['recall'], 'fp': m['fp']}), flush=True)
        _, chosen = max(candidates)
        pc = candidate_score(chosen, gates, cal['x'], cal['old'], cal['new'], old_artifact['threshold'])
        t = threshold_for_budget(cal['y'], pc, budget(cal['y'], rate))
        cm = metrics(cal['y'], pc, t, cal['sources'])
        if cm['fp'] > budget(cal['y'], rate):
            raise AssertionError('Calibration FP allowance exceeded')
        profiles[profile] = dict(selected=chosen, threshold=t, developmentHamFpr=rate,
                                 calibration=cm, hamAllowance=budget(cal['y'], rate))
    component_hashes = {}
    for destination, source in [('v54/model.joblib', v54root / 'model.joblib'), ('v54/neural.pt', v54root / 'neural.pt'), ('v55/model.joblib', v55root / 'model.joblib')]:
        target = out / destination; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target); component_hashes[destination] = file_sha(target)
    artifact = dict(version='v56-consensus', gates=gates, profiles=profiles, defaultProfile='conservative',
                    components=component_hashes, warning=WARNING, corpusSha256=CORPUS_SHA)
    joblib.dump(artifact, out / 'model.joblib', compress=3)
    model_hash = file_sha(out / 'model.joblib')
    # Verify packaged inference against cached-component assembly on fixed development rows.
    check_rows = sorted(selection['rows'], key=lambda r: r['identity'])[:10]
    positions = {r['identity']: i for i, r in enumerate(selection['rows'])}
    ix = [positions[r['identity']] for r in check_rows]
    bundle = load_bundle(out)
    for profile, setting in profiles.items():
        actual = predict_texts(bundle, [r['text'] for r in check_rows], profile)
        expected = candidate_score(setting['selected'], gates, selection['x'][ix], selection['old'][ix], selection['new'][ix], old_artifact['threshold'])
        if not np.allclose(actual, expected, atol=1e-5, rtol=1e-5) or not np.array_equal(actual >= setting['threshold'], expected >= setting['threshold']):
            raise AssertionError('Packaged inference differs from development predictions')
    del bundle, data, selection, cal
    gc.collect()
    # Frozen v55 test scores are reused verbatim; only the small gate runs over all rows.
    with (v55root / 'predictions.csv').open() as f:
        v55_index = {r['identity']: (int(r['label']), float(r['score'])) for r in csv.DictReader(f) if r['identity'] in test_ids}
    old_texts = {}
    for raw in frozen['test']:
        parsed = from_eml(raw['path'].read_bytes())
        if parsed['identity'] in test_ids and parsed['identity'] not in old_texts:
            old_texts[parsed['identity']] = parsed['ngram_text']
    del frozen
    gc.collect()
    y_all, src_all, old_all, new_all = [], [], [], []
    predicted = {profile: [] for profile in profiles}
    identities = set()
    with (out / 'predictions.csv').open('w', newline='') as pf:
        writer = csv.writer(pf)
        writer.writerow(['identity', 'family', 'source', 'label', 'v54Spam', 'v55Spam'] + [f'{p}Spam' for p in profiles])
        def consume(rows):
            old = np.zeros(len(rows), dtype=np.float64)
            new = np.zeros(len(rows), dtype=np.float64)
            legacy_ix = []
            for i, r in enumerate(rows):
                label, new[i] = v55_index[r['identity']]
                if label != r['y']:
                    raise AssertionError('v55 cached label mismatch')
                if is_new(r['source']):
                    label, old[i] = baseline_index[r['identity']]
                    if label != r['y']:
                        raise AssertionError('v54 cached label mismatch')
                else:
                    legacy_ix.append(i)
            for start in range(0, len(legacy_ix), 128):
                ix = legacy_ix[start:start + 128]
                old[ix] = predict_margin_union(old_artifact, neural, [old_texts[rows[i]['identity']] for i in ix])
            x = gate_features(rows, old, new, old_artifact['threshold'])
            decisions = {}
            for profile, setting in profiles.items():
                scores = candidate_score(setting['selected'], gates, x, old, new, old_artifact['threshold'])
                decisions[profile] = scores >= setting['threshold']
                predicted[profile].extend(decisions[profile].tolist())
            for i, r in enumerate(rows):
                if r['identity'] in identities:
                    raise AssertionError('Duplicate test identity')
                identities.add(r['identity'])
                y_all.append(r['y']); src_all.append(r['source']); old_all.append(old[i]); new_all.append(new[i])
                writer.writerow([r['identity'], r['family'], r['source'], r['y'], int(old[i] >= old_artifact['threshold']),
                                 int(new[i] >= new_artifact['threshold'])] + [int(decisions[p][i]) for p in profiles])
            if len(y_all) % 10000 == 0:
                print('v56 holdout scored', len(y_all), flush=True)
        batch = []
        for r in corpus_rows(v55root / 'test.jsonl.gz'):
            if r['identity'] not in test_ids:
                continue
            batch.append(r)
            if len(batch) == 1000:
                consume(batch); batch = []
        if batch:
            consume(batch)
    if identities != test_ids or file_sha(out / 'model.joblib') != model_hash:
        raise AssertionError('Holdout inventory mismatch or model changed after calibration')
    y = np.asarray(y_all); mask = np.array([is_new(s) for s in src_all])
    def grouped(scores, t):
        return dict(all=metrics(y, scores, t, src_all),
                    new=metrics(y[mask], np.asarray(scores)[mask], t, list(np.asarray(src_all)[mask])),
                    legacy=metrics(y[~mask], np.asarray(scores)[~mask], t, list(np.asarray(src_all)[~mask])))
    results = {profile: grouped(np.asarray(predicted[profile], dtype=float), .5) for profile in profiles}
    report = dict(version=56, testCount=len(y), profiles=profiles, results=results, candidates=candidate_reports,
                  baselineV54=grouped(old_all, old_artifact['threshold']), baselineV55=grouped(new_all, new_artifact['threshold']),
                  splits=dict(gateFit=len(fit_mask.nonzero()[0]), gateSelect=len(sel_mask.nonzero()[0]),
                              calibration=sum(profiles['conservative']['calibration'][k] for k in ('tp', 'fn', 'fp', 'tn')),
                              test=len(y), familyOverlap=0, manifestSha256=file_sha(v55root / 'split-manifest.csv')),
                  modelSha256=model_hash, componentHashes=component_hashes, modelRefitAfterCalibration=False,
                  testLabelsUsedForGateOrThresholdFit=False, testPreviouslyInspected=True,
                  frozenV54FamilyIsolation=False, productionGoalValidated=False, packagedInferenceCheckPassed=True,
                  limit=args.limit, audit=audit, warning=WARNING)
    report['engineeringGoals'] = {profile: dict(recall95AndFp50=(r['all']['recall'] >= .95 and r['all']['fp'] <= 50 and not args.limit),
                                               bothSourceGroupsRecall95=(r['new']['recall'] >= .95 and r['legacy']['recall'] >= .95 and not args.limit))
                                  for profile, r in results.items()}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    md = '# MailGuard v56 consensus\n\n' + WARNING + '\n\n| Model / profile | Group | N | Ham | Spam recall | FP | FN |\n|---|---|---:|---:|---:|---:|---:|\n'
    for name, groups in [('v54', report['baselineV54']), ('v55', report['baselineV55'])] + list(results.items()):
        for group in ('new', 'legacy', 'all'):
            m = groups[group]; n = sum(m[k] for k in ('tp', 'fn', 'fp', 'tn'))
            md += f"| {name} | {group} | {n} | {m['fp']+m['tn']} | {m['recall']:.2%} | {m['fp']} | {m['fn']} |\n"
    md += '\nDefault profile: conservative. High-recall is an explicitly separate engineering operating point. Neither profile is a production guarantee.\n'
    (out / 'report.md').write_text(md)
    print(md, flush=True)


if __name__ == '__main__':
    main()
