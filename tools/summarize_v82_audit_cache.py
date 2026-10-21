"""Explain the completed v82 audit using its score cache; no model or optimizer."""
import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from audit_v82_errors import describe, length_bucket, metric, paired, seen_train, sha


def fingerprint(rows):
    # Same full-content fingerprint as the checkpoint, without importing torch.
    digest = hashlib.sha256()
    for row in rows:
        data = json.dumps({k: row[k] for k in ('identity', 'family', 'source', 'y', 'text')},
                          ensure_ascii=False, sort_keys=True).encode()
        digest.update(len(data).to_bytes(8, 'big')); digest.update(data)
    return digest.hexdigest()


def log_loss(rows, scores):
    metric(rows, scores, .5)
    if not rows or any(not 0 <= float(s) <= 1 for s in scores):
        raise ValueError('Expected nonempty probability scores')
    values = [min(1 - 1e-7, max(1e-7, float(s))) for s in scores]
    return -sum(r['y'] * math.log(s) + (1 - r['y']) * math.log1p(-s)
                for r, s in zip(rows, values)) / len(rows)


def validate_alignment(rows, identities, labels, scores):
    expected = [r['identity'] for r in rows]
    if (len(set(expected)) != len(expected) or list(identities) != expected or
        list(labels) != [r['y'] for r in rows]):
        raise ValueError('Score identities, labels or order do not match frozen rows')
    log_loss(rows, scores)


def summary(rows, scores, cutoff):
    result = describe(rows, scores, cutoff)
    result.pop('highest_scoring_false_positives')
    result.pop('lowest_scoring_missed_spam')
    result['log_loss'] = log_loss(rows, scores)
    return result


def preservation_frontier(rows, old_scores, new_scores, old_cutoff):
    metric(rows, old_scores, old_cutoff); log_loss(rows, new_scores)
    previous_tp = [float(new_scores[i]) for i, r in enumerate(rows)
                   if r['y'] == 1 and old_scores[i] >= old_cutoff]
    if not previous_tp:
        raise ValueError('No baseline spam detections to preserve')
    # With score >= cutoff this is the largest scalar cutoff preserving every old TP.
    # It uses SELECT labels and predictions: diagnostic fitting, never a blind test.
    cutoff = min(previous_tp)
    return dict(threshold=cutoff,
                paired_select=paired(rows, old_scores, new_scores, old_cutoff, cutoff),
                selected_on_same_development=True, deployable_policy=False,
                is_a_v80_regression_check=False)


def grouped_pairs(rows, old_scores, new_scores, old_cutoff, new_cutoff, key):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[key(row)].append(i)
    result = {}
    for name, ix in sorted(groups.items()):
        value = paired([rows[i] for i in ix], [old_scores[i] for i in ix],
                       [new_scores[i] for i in ix], old_cutoff, new_cutoff)
        value.pop('first_lost_spam_identities')
        result[name] = value
    return result


def analyze(select, sample, old_scores, new_scores, train_scores, old_cutoff, new_cutoff):
    baseline = summary(select, old_scores, old_cutoff)
    current = summary(select, new_scores, new_cutoff)
    train = summary(sample, train_scores, new_cutoff)
    ham = current['at_locked_development_threshold']['ham']
    if not ham:
        raise ValueError('SELECT has no HAM')
    standardized = 0.
    for name, value in current['by_source'].items():
        if value['ham']:
            fpr = train['by_source'].get(name, {}).get('fpr')
            if fpr is None:
                raise ValueError('Cannot standardize a source absent from sampled TRAIN HAM')
            standardized += value['ham'] * fpr / ham
    return dict(baseline_select=baseline, current_select=current, current_seen_train=train,
                paired_by_source=grouped_pairs(select, old_scores, new_scores, old_cutoff,
                                               new_cutoff, lambda r: r['source']),
                paired_by_word_count=grouped_pairs(select, old_scores, new_scores, old_cutoff,
                                                   new_cutoff, lambda r: length_bucket(r['text'])),
                source_standardized_seen_train_fpr=standardized,
                observed_select_fpr=current['at_locked_development_threshold']['fpr'],
                source_standardization_is_causal_evidence=False,
                preserve_500_step_select_tp=preservation_frontier(select, old_scores, new_scores, old_cutoff))


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('.cache/v82-audit-cache-summary.json'))
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    reference = json.loads(Path(request['cached_reference_report']).read_text())
    source = args.cache / 'audit.json'; scores_path = args.cache / 'development-scores.npz'
    original_hashes = {str(p): sha(p) for p in (source, scores_path)}
    if json.loads(source.read_text()) != reference:
        raise ValueError('Downloaded audit does not equal the committed reference')
    if (reference['campaign'] != request['campaign'] or
        reference['global_optimizer_steps'] < request['expected_min_step'] or
        reference['baseline_step'] != request['baseline_step'] or
        not reference['checkpoint_files_unchanged'] or
        not reference['source_report_counts_reproduced'] or
        reference['optimizer_steps_executed'] != 0 or reference['cal_or_test_opened']):
        raise ValueError('Source audit failed its required provenance checks')
    baseline_report = json.loads(Path(request['cached_baseline_report']).read_text())
    current_report = json.loads(Path(request['cached_current_report']).read_text())
    if (baseline_report['checkpoint_sha256'] != reference['baseline_checkpoint_sha256'] or
        current_report['checkpoint_sha256'] != reference['current_checkpoint_sha256'] or
        current_report['global_optimizer_steps'] != reference['global_optimizer_steps'] or
        baseline_report['global_optimizer_steps'] != reference['baseline_step'] or
        current_report['development'] != reference['source_report']):
        raise ValueError('Daily reports do not match the scored checkpoints')
    train, select = [], []
    for directory, manifest in ((Path('.cache/v79/frozen'), Path('reports/v79-frozen.json')),
                                (Path('.cache/v80/extra'), Path('reports/v80-extra-frozen.json'))):
        if json.loads((directory / 'frozen.json').read_text()) != json.loads(manifest.read_text()):
            raise ValueError('Frozen manifest drift')
        for name, target in (('train', train), ('select', select)):
            with gzip.open(directory / (name + '.jsonl.gz'), 'rt') as stream:
                target.extend(json.loads(line) for line in stream)
    if dict(train=fingerprint(train), select=fingerprint(select)) != reference['corpus_fingerprints']:
        raise ValueError('Full corpus fingerprint drift')
    sample, population = seen_train(train, current_report, 82, request['train_seen_limit'])
    if population != reference['train_seen_population'] or fingerprint(sample) != reference['seen_train_sample_fingerprint']:
        raise ValueError('Observed TRAIN sample drift')
    with np.load(scores_path, allow_pickle=False) as cache:
        expected_keys = {'select_identity', 'select_y', 'current', 'baseline',
                         'train_identity', 'train_y', 'train_scores'}
        if set(cache.files) != expected_keys or any(cache[k].ndim != 1 for k in expected_keys):
            raise ValueError('Unexpected score cache schema')
        values = {k: cache[k].tolist() for k in expected_keys}
    validate_alignment(select, values['select_identity'], values['select_y'], values['current'])
    validate_alignment(select, values['select_identity'], values['select_y'], values['baseline'])
    validate_alignment(sample, values['train_identity'], values['train_y'], values['train_scores'])
    old_cutoff = reference['paired_select']['baseline_threshold']
    new_cutoff = reference['paired_select']['current_threshold']
    if paired(select, values['baseline'], values['current'], old_cutoff, new_cutoff) != reference['paired_select']:
        raise ValueError('Cached paired counts do not reproduce the original audit')
    result = analyze(select, sample, values['baseline'], values['current'], values['train_scores'],
                     old_cutoff, new_cutoff)
    for key, report in (('baseline_select', baseline_report), ('current_select', current_report)):
        observed = result[key]['at_locked_development_threshold']
        if any(observed[k] != report['development'][k] for k in ('n', 'spam', 'ham', 'tp', 'fp')):
            raise ValueError('Cached counts do not reproduce daily report')
        if not math.isclose(result[key]['log_loss'], report['development']['log_loss'], abs_tol=1e-9, rel_tol=0):
            raise ValueError('Cached log loss does not reproduce daily report')
    for key in ('at_locked_development_threshold', 'at_predefined_threshold_0_5',
                'by_source', 'by_word_count', 'by_character_window'):
        if result['current_seen_train'][key] != reference['seen_train_sample'][key]:
            raise ValueError('Cached TRAIN summary does not reproduce original audit')
        if result['current_select'][key] != reference['select'][key]:
            raise ValueError('Cached SELECT summary does not reproduce original audit')
    if any(sha(p) != expected for p, expected in original_hashes.items()):
        raise ValueError('Source cache modified')
    result.update(campaign=reference['campaign'], global_optimizer_steps=reference['global_optimizer_steps'],
                  source_audit_run_id=request['cached_audit_run_id'],
                  baseline_checkpoint_sha256=reference['baseline_checkpoint_sha256'],
                  current_checkpoint_sha256=reference['current_checkpoint_sha256'],
                  source_input_sha256=original_hashes, analysis_code_sha256=sha(__file__),
                  source_cache_unchanged=True, source_report_counts_reproduced=True,
                  source_report_log_losses_reproduced=True,
                  source_audit_checkpoint_files_unchanged=reference['checkpoint_files_unchanged'],
                  checkpoint_files_opened=False, optimizer_steps_executed=0, inference_steps_executed=0,
                  cal_or_test_opened=False, raw_message_text_saved=False,
                  independent_quality_test_performed=False, production_v80_unchanged=True,
                  labels_automatically_changed=False, quality_gain_demonstrated=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(cached_analysis_completed=True, step=result['global_optimizer_steps'],
                          preservation=result['preserve_500_step_select_tp'],
                          source_cache_unchanged=True, optimizer_steps_executed=0), indent=2))


if __name__ == '__main__':
    main()
