"""Fit on SELECT-fit; constrain on SELECT-choose/CAL; lock before regression.

No 100k/2024 outcomes are read here. All former test sets are regression only.
"""
import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import v77_mail as core
import v81_mail as runtime
from train_v78_en import source_weights
from train_v80_en import read_rows
from v79_data import fingerprint
from v80_encoder import file_sha
from v81_features import cached_fast, cached_neural
from train_v80_en import cache_embeddings

RESERVE_IQR = .25
MIN_SPAM_PER_REGION = 100


def paired(original, proposed, rows):
    y = np.asarray([r['y'] for r in rows], dtype=bool)
    def metric(p):
        return dict(n=len(y), spam=int(y.sum()), ham=int((~y).sum()),
                    tp=int((p & y).sum()), fp=int((p & ~y).sum()),
                    fn=int((~p & y).sum()), tn=int((~p & ~y).sum()))
    return dict(baseline=metric(original), proposed=metric(proposed),
                baseline_spam_detections_lost=int((original & ~proposed & y).sum()),
                spam_detections_gained=int((~original & proposed & y).sum()),
                false_positives_removed=int((original & ~proposed & ~y).sum()),
                false_positives_added=int((~original & proposed & ~y).sum()))


def calibrate(fast, scores, rows, partition, cutoff, reserve, previous=None):
    groups = runtime.regions(fast, partition)
    protected = (runtime.base_scores(fast) >= cutoff) & np.array([r['y'] == 1 for r in rows])
    thresholds = {}
    for group in sorted(set(groups.tolist())):
        values = scores[protected & (groups == group)]
        key = str(group)
        if len(values) < MIN_SPAM_PER_REGION or (previous is not None and key not in previous):
            continue
        cut = float(np.min(values) - reserve)
        thresholds[key] = min(cut, previous[key]) if previous is not None else cut
    return thresholds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('.cache/v79/frozen'))
    p.add_argument('--extra', type=Path, default=Path('.cache/v80/extra'))
    p.add_argument('--baseline', type=Path, default=Path('.cache/v81/baseline/model.joblib'))
    p.add_argument('--out', type=Path, default=Path('reports/v81'))
    p.add_argument('--selection-mode', choices=('select-only', 'joint'), default='joint')
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    cache = Path('.cache/v81/features'); cache.mkdir(parents=True, exist_ok=True)
    if file_sha(args.baseline) != runtime.BASELINE_SHA:
        raise ValueError('Wrong baseline')
    shutil.copytree(args.baseline.parent, args.out / 'baseline')
    bundle = dict(version=runtime.VERSION, experts={})
    runtime.attach(bundle, SimpleNamespace(model=args.out / 'model.joblib'))
    old = bundle['_baseline']; cutoff = old['policies']['recall-98']['cutoff']
    if old['policies']['recall-98']['candidate'] != 'fast-or':
        raise ValueError('Unexpected high-recall baseline formula')
    rows = {}
    for part in ('select', 'cal'):
        original = read_rows(args.data / (part + '.jsonl.gz'))
        extra = read_rows(args.extra / (part + '.jsonl.gz'))
        if fingerprint(original) != old['original_manifest']['development_fingerprints'][part] or fingerprint(extra) != old['extra_manifest']['fingerprints'][part]:
            raise ValueError('Development partition drift')
        rows[part] = original + extra
    fit = np.array([i for i, r in enumerate(rows['select'])
                    if int(core.digest('v80-meta:' + r['family'])[:10], 16) % 2 == 0])
    fit_mask = np.zeros(len(rows['select']), dtype=bool); fit_mask[fit] = True
    choose = np.flatnonzero(~fit_mask)
    features, embeddings, fast = {}, {}, {}
    # This loop deliberately excludes CAL until candidate selection is complete.
    fast['select'] = cached_fast(args.baseline, rows['select'], cache / 'select-fast.npy')
    mask = (runtime.base_scores(fast['select']) >= cutoff) | fit_mask
    features['select'], embeddings['select'] = cached_neural(old, rows['select'], fast['select'], mask, cache / 'select-embedding.npy')
    lexical = runtime.lexical_features([r['text'] for r in rows['select']])
    fit_rows = [rows['select'][i] for i in fit]
    labels = np.array([r['y'] for r in fit_rows]); weights = source_weights(fit_rows)
    print('Secondary fit', len(fit), 'choose', len(choose), flush=True)
    linear = LinearSVC(C=1., dual=False, max_iter=5000, random_state=81)
    linear.fit(embeddings['select'][fit], labels, sample_weight=weights)
    bundle['experts']['embedding-linear'] = dict(input='embedding', model=linear)
    rbf = make_pipeline(Nystroem(gamma=4., n_components=256, random_state=81),
                        LinearSVC(C=1., dual=False, max_iter=5000, random_state=81))
    rbf.fit(embeddings['select'][fit], labels, linearsvc__sample_weight=weights)
    bundle['experts']['embedding-rbf'] = dict(input='embedding', model=rbf)
    tree = HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, min_samples_leaf=50,
                                         l2_regularization=20., random_state=81)
    tree.fit(np.column_stack([features['select'], lexical])[fit], labels, sample_weight=weights)
    bundle['experts']['stack-tree'] = dict(input='meta', model=tree)
    original_train = read_rows(args.data / 'train.jsonl.gz')
    extra_train = read_rows(args.extra / 'train.jsonl.gz')
    if fingerprint(original_train) != old['original_manifest']['development_fingerprints']['train'] or fingerprint(extra_train) != old['extra_manifest']['fingerprints']['train']:
        raise ValueError('TRAIN anchor partition drift')
    groups = {}
    for row in original_train + extra_train:
        if row['source'] != 'Merged-public':
            groups.setdefault((row['source'], row['y']), []).append(row)
    anchors = []
    for _, group in sorted(groups.items()):
        anchors.extend(sorted(group, key=lambda r: core.digest('v81-anchor:' + r['identity']))[:1000])
    anchor_embedding = cache_embeddings(old['_encoder'], anchors, cache / 'train-anchors.npy')
    anchor_y = np.array([r['y'] for r in anchors])
    bundle['experts']['semantic-neighbors'] = dict(input='neighbors',
        ham=anchor_embedding[anchor_y == 0], spam=anchor_embedding[anchor_y == 1])
    print('TRAIN-only semantic anchors', len(anchors), flush=True)
    del original_train, extra_train, groups, anchors, anchor_embedding
    names = list(bundle['experts']) + ['v80-gate:' + n for n in sorted(old['gates'])] + ['v80-semantic', 'v80-semantic-phish', 'v79-word', 'v79-char', 'v79-blend']
    choose_rows = [rows['select'][i] for i in choose]
    original = runtime.base_scores(fast['select'][choose]) >= cutoff
    protected_fit = labels.astype(bool) & (runtime.base_scores(fast['select'][fit]) >= cutoff)
    candidates, best = [], None
    for name in names:
        scores = runtime.candidate_scores(bundle, name, features['select'], embeddings['select'], lexical)
        q = np.quantile(scores[fit][protected_fit], [.25, .75])
        reserve = float(max(q[1] - q[0], 1e-4) * RESERVE_IQR)
        for partition in ('global', 'dominance', 'margin'):
            thresholds = calibrate(fast['select'][choose], scores[choose], choose_rows,
                                   partition, cutoff, reserve)
            policy = dict(candidate=name, partition=partition, baseline_cutoff=cutoff,
                          reserve=reserve, thresholds=thresholds, enabled=bool(thresholds))
            proposed = runtime.apply_policy(fast['select'][choose], scores[choose], policy)
            metric = paired(original, proposed, choose_rows)
            if metric['baseline_spam_detections_lost'] or metric['false_positives_added']:
                raise AssertionError('Selection violated hard constraint')
            record = dict(policy=policy, selection=metric); candidates.append(record)
            key = (metric['false_positives_removed'], -len(thresholds))
            if best is None or key > best[0]:
                best = (key, record, scores)
    _, selected, _ = best
    policy = dict(selected['policy'])
    print('SELECT CHOSEN', json.dumps(selected), flush=True)
    fast['cal'] = cached_fast(args.baseline, rows['cal'], cache / 'cal-fast.npy')
    mask = runtime.base_scores(fast['cal']) >= cutoff
    features['cal'], embeddings['cal'] = cached_neural(old, rows['cal'], fast['cal'], mask, cache / 'cal-embedding.npy')
    cal_scores = runtime.candidate_scores(bundle, policy['candidate'], features['cal'], embeddings['cal'],
                                          runtime.lexical_features([r['text'] for r in rows['cal']]))
    policy['thresholds'] = calibrate(fast['cal'], cal_scores, rows['cal'], policy['partition'], cutoff,
                                     policy['reserve'], previous=policy['thresholds'])
    policy['enabled'] = bool(policy['thresholds'])
    proposed = runtime.apply_policy(fast['cal'], cal_scores, policy)
    cal = paired(mask, proposed, rows['cal'])
    if cal['baseline_spam_detections_lost'] or cal['false_positives_added']:
        raise AssertionError('Calibration violated hard constraint')
    select_scores = runtime.candidate_scores(bundle, policy['candidate'], features['select'], embeddings['select'], lexical)
    final_selection = paired(original, runtime.apply_policy(fast['select'][choose], select_scores[choose], policy), choose_rows)
    if final_selection['baseline_spam_detections_lost']:
        raise AssertionError('Calibration weakened the SELECT protection')
    guarded_evidence = dict(selection=final_selection, calibration=cal)
    # Require repeatable observed benefit on both development partitions.
    if final_selection['false_positives_removed'] < 3 or cal['false_positives_removed'] < 5:
        policy['enabled'] = False
        final_selection = paired(original, original, choose_rows)
        cal = paired(mask, mask, rows['cal'])
    joint_candidates = []
    if args.selection_mode == 'joint':
        # CAL participates in model choice here. It is development data, not
        # an independent estimate of the selected model's performance.
        select_values, cal_values, winner = {}, {}, None
        cal_lexical = runtime.lexical_features([r['text'] for r in rows['cal']])
        for record in candidates:
            candidate = record['policy']['candidate']
            if candidate not in select_values:
                select_values[candidate] = runtime.candidate_scores(bundle, candidate, features['select'], embeddings['select'], lexical)
                cal_values[candidate] = runtime.candidate_scores(bundle, candidate, features['cal'], embeddings['cal'], cal_lexical)
            cfg = dict(record['policy'])
            cfg['thresholds'] = calibrate(fast['cal'], cal_values[candidate], rows['cal'], cfg['partition'],
                                          cutoff, cfg['reserve'], previous=cfg['thresholds'])
            cfg['enabled'] = bool(cfg['thresholds'])
            sel = paired(original, runtime.apply_policy(fast['select'][choose], select_values[candidate][choose], cfg), choose_rows)
            c = paired(mask, runtime.apply_policy(fast['cal'], cal_values[candidate], cfg), rows['cal'])
            if sel['baseline_spam_detections_lost'] or c['baseline_spam_detections_lost']:
                raise AssertionError('Joint selection violated the no-loss constraint')
            joint_candidates.append(dict(policy=cfg, selection=sel, calibration=c))
            if sel['false_positives_removed'] + c['false_positives_removed'] > 0:
                # Rare FP corrections need not appear in both small sections.
                # Their benefit must still survive the locked paired regression.
                key = (sel['false_positives_removed'] + c['false_positives_removed'],
                       min(sel['false_positives_removed'], c['false_positives_removed']),
                       -len(cfg['thresholds']))
                if winner is None or key > winner[0]:
                    winner = (key, cfg, sel, c)
        if winner is not None:
            _, policy, final_selection, cal = winner
            guarded_evidence = dict(selection=final_selection, calibration=cal)
        print('JOINT DEVELOPMENT CHOICE', json.dumps(dict(policy=policy, selection=final_selection, calibration=cal)), flush=True)
    bundle['experts'] = {n: v for n, v in bundle['experts'].items() if n == policy['candidate']}
    bundle['policy'] = policy
    bundle['original_manifest'] = old['original_manifest']; bundle['extra_manifest'] = old['extra_manifest']
    bundle['development'] = dict(selection=final_selection, calibration=cal, candidates=candidates,
                                  fit_count=len(fit), choose_count=len(choose), reserve_iqr=RESERVE_IQR,
                                  min_spam_per_region=MIN_SPAM_PER_REGION,
                                  anchor_count=int(len(anchor_y)), guarded_candidate_evidence=guarded_evidence,
                                  disabled_for_insufficient_development_benefit=not policy['enabled'],
                                  selection_mode=args.selection_mode, joint_candidates=joint_candidates,
                                  calibration_used_for_candidate_choice=args.selection_mode == 'joint',
                                  calibration_is_independent_performance_estimate=False)
    serializable = {k: v for k, v in bundle.items() if not k.startswith('_')}
    joblib.dump(serializable, args.out / 'model.joblib', compress=3)
    locked = dict(model_sha256=file_sha(args.out / 'model.joblib'), policy=policy,
                  development=bundle['development'], regression_outcomes_used=False,
                  individual_baseline_tp_preservation_required=True)
    (args.out / 'policy-locked.json').write_text(json.dumps(locked, indent=2))
    print('LOCKED BEFORE REGRESSION', json.dumps(locked), flush=True)


if __name__ == '__main__':
    main()
