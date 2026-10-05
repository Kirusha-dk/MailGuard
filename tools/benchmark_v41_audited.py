#!/usr/bin/env python3
"""V40 model recipe, corrected Rspamd transport, exact frozen v40 data.

No tuning against test errors. The reused test is an engineering comparison,
not a new lockbox. Run only against the disposable benchmark Docker stack.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import benchmark_v40_fresh50k as v40
from rspamd_benchmark_audit import (
    check_learning_reply, load_frozen_v40, parse_scan, request_scan, scan_health,
)

b = v40.base.b
REPORTS = Path('reports')
HEALTH = {}
ORIGINAL_SCAN_MANY = b.scan_many
ORIGINAL_METRICS = v40.v19.metrics
DECISIONS = []
METRICS = []


def save_health():
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / 'v41-rspamd-health.json').write_text(json.dumps(HEALTH, indent=2))


def strict_scan(item):
    raw = item['path'].read_bytes()
    response = request_scan(b.RSPAMD + 'checkv2', raw)
    return {**item, 'raw': raw, **parse_scan(response)}


def audited_scan_many(items, label, workers=8):
    rows = ORIGINAL_SCAN_MANY(items, label, workers)
    HEALTH[label] = scan_health(rows)
    save_health()
    return rows


def audited_learn(items):
    counts = Counter()
    skip_markers = (
        'not enough tokens',
        'less tokens than required',
        'contains less tokens',
        'skip learning',
        'already learned',
        'already learnt',
    )
    failure_markers = (
        'success = false',
        '"success": false',
        "'success': false",
        'error =',
        '"error":',
    )

    for item in items:
        label = 'spam' if item['y'] else 'ham'
        try:
            raw_reply = b.post(
                b.CTRL + 'learn' + label,
                item['path'].read_bytes(),
                30,
            )
        except HTTPError as exc:
            body = exc.read().decode('utf-8', 'replace').lower()
            if exc.code in (400, 404) and any(x in body for x in skip_markers):
                counts[label + 'Skipped'] += 1
                counts['replyFormat:httpSkip'] += 1
            else:
                HEALTH['learning'] = dict(counts)
                HEALTH['learningFailure'] = {
                    'httpCode': int(exc.code),
                    'bodyPrefix': body[:500],
                }
                save_health()
                raise RuntimeError(
                    f'Bayes learning failed (HTTP {exc.code})'
                ) from exc
        else:
            text_reply = raw_reply.decode('utf-8', 'replace').strip()
            lower_reply = text_reply.lower()

            if any(x in lower_reply for x in skip_markers):
                counts[label + 'Skipped'] += 1
                counts['replyFormat:bodySkip'] += 1
            else:
                if any(x in lower_reply for x in failure_markers):
                    HEALTH['learning'] = dict(counts)
                    HEALTH['learningFailure'] = {
                        'httpCode': 200,
                        'bodyPrefix': text_reply[:500],
                    }
                    save_health()
                    raise RuntimeError(
                        'Bayes learning returned an explicit failure'
                    )

                # A 2xx controller reply is accepted even if Rspamd serializes it
                # as plain text/UCL instead of JSON. The real proof of learning is
                # the post-training stat + Bayes-symbol probe below.
                reply_format = 'empty2xx'
                if text_reply:
                    try:
                        reply = json.loads(text_reply)
                    except json.JSONDecodeError:
                        reply_format = 'plain2xx'
                    else:
                        if not isinstance(reply, dict):
                            raise RuntimeError(
                                'Bayes learning returned unexpected JSON type'
                            )
                        if reply.get('success') is False or reply.get('error'):
                            HEALTH['learningFailure'] = {
                                'httpCode': 200,
                                'bodyPrefix': text_reply[:500],
                            }
                            save_health()
                            raise RuntimeError(
                                'Bayes learning JSON reported failure'
                            )
                        reply_format = 'json2xx'

                counts[label + 'Learned'] += 1
                counts['replyFormat:' + reply_format] += 1

        total = (
            counts['hamLearned'] + counts['hamSkipped']
            + counts['spamLearned'] + counts['spamSkipped']
        )
        if total and total % 500 == 0:
            HEALTH['learning'] = dict(counts)
            save_health()
            print('v41 accepted learning', total, dict(counts), flush=True)

    HEALTH['learning'] = dict(counts)

    stat_raw = b.get(b.CTRL + 'stat')
    try:
        HEALTH['statAfterLearning'] = json.loads(stat_raw)
    except (json.JSONDecodeError, TypeError):
        if isinstance(stat_raw, bytes):
            stat_text = stat_raw.decode('utf-8', 'replace')
        else:
            stat_text = str(stat_raw)
        HEALTH['statAfterLearningRaw'] = stat_text[:10000]

    for label in ('ham', 'spam'):
        learned = counts[label + 'Learned']
        skipped = counts[label + 'Skipped']
        attempted = learned + skipped
        if learned < 200 or learned < .8 * max(1, attempted):
            save_health()
            raise RuntimeError(
                f'Insufficient accepted {label} learning: '
                f'learned={learned} skipped={skipped}'
            )

    # Strong end-to-end proof: after learning, a balanced probe must produce
    # both BAYES_HAM and BAYES_SPAM symbols. This verifies usable Bayes state
    # regardless of how /learnham and /learnspam serialize successful replies.
    probe = []
    for label in (0, 1):
        candidates = [x for x in items if x['y'] == label]
        probe.extend(
            candidates[::max(1, len(candidates) // 150)][:150]
        )

    rows = ORIGINAL_SCAN_MANY(
        probe,
        'v41-baseline-probe',
        workers=8,
    )
    HEALTH['probe'] = scan_health(rows)

    symbol_counts = Counter(
        name for row in rows for name, _ in row['symbols']
    )
    HEALTH['probe']['bayesHamHits'] = int(symbol_counts['BAYES_HAM'])
    HEALTH['probe']['bayesSpamHits'] = int(symbol_counts['BAYES_SPAM'])
    save_health()

    if not symbol_counts['BAYES_HAM'] or not symbol_counts['BAYES_SPAM']:
        raise RuntimeError(
            'Bayes learning was not usable: balanced probe lacks '
            'BAYES_HAM or BAYES_SPAM'
        )


def capture_guard(base_rspamd, score, ham_risk, agreement, gate):
    primary = ((score >= gate['primaryThreshold'])
               & (ham_risk <= gate['primaryMaxHamRisk'])
               & (agreement >= gate['primaryMinAgreement']))
    override = ((score >= gate['overrideThreshold'])
                & (ham_risk <= gate['overrideMaxHamRisk'])
                & (agreement >= gate['overrideMinAgreement']))
    standalone = primary | override
    combined = base_rspamd | standalone
    DECISIONS.append(dict(name=gate['name'], score=score.copy(), hamRisk=ham_risk.copy(),
                          agreement=agreement.copy(), primary=primary, override=override,
                          baseline=base_rspamd.copy(), standalone=standalone, combined=combined))
    return combined


def capture_metrics(rows, pred):
    result = ORIGINAL_METRICS(rows, pred)
    METRICS.append((rows, np.asarray(pred, dtype=bool).copy(), result))
    return result


def write_report():
    if len(METRICS) != 8 or len(DECISIONS) != 4:
        raise RuntimeError('Unexpected evaluation call count; refusing mislabeled report')
    rows = METRICS[-1][0]
    modes = {}
    for decision in DECISIONS:
        name = decision['name']
        modes[name] = {}
        for kind in ('standalone', 'combined'):
            pred = decision[kind]
            modes[name][kind] = ORIGINAL_METRICS(rows, pred)
        # Content is deliberately excluded. Hashes let errors be joined locally.
        with (REPORTS / f'v41-{name}-predictions.jsonl').open('w') as out:
            for i, row in enumerate(rows):
                record = {'id': hashlib.sha256(row['raw']).hexdigest(),
                          'file': row['path'].name, 'source': row['source'], 'label': int(row['y']),
                          'predictedSpam': bool(decision['combined'][i]),
                          'standaloneSpam': bool(decision['standalone'][i]),
                          'baselineSpam': bool(decision['baseline'][i]),
                          'primary': bool(decision['primary'][i]),
                          'override': bool(decision['override'][i]),
                          'score': float(decision['score'][i]),
                          'hamRisk': float(decision['hamRisk'][i]),
                          'agreement': int(decision['agreement'][i]),
                          'rspamdScore': row['rscore'], 'action': row['action'],
                          'symbols': row['symbols']}
                out.write(json.dumps(record) + '\n')
        grouped = {}
        for source in sorted({r['source'] for r in rows}):
            indices = [i for i, r in enumerate(rows) if r['source'] == source]
            grouped[source] = ORIGINAL_METRICS([rows[i] for i in indices], decision['combined'][indices])
        modes[name]['bySource'] = grouped
        y = np.asarray([bool(r['y']) for r in rows])
        modes[name]['falsePositiveOrigins'] = {
            'baseline': int(((~y) & decision['baseline']).sum()),
            'addedByMailGuard': int(((~y) & (~decision['baseline']) & decision['standalone']).sum())}
    report = json.loads((REPORTS / 'v24-crossfit.json').read_text())
    report['version'] = 'v41-audited-same-v40-50k'
    report['dataset']['split'] = 'exact immutable v40 cache; source-group OOF meta calibration'
    report['dataset']['fingerprints'] = HEALTH['dataset']
    report['method']['changesFromV40'] = [
        'Flags: pass_all; benchmark-only greylisting disabled; explicit autolearn=false',
        'strict scan/learning validation and fail-fast Bayes health checks',
        'per-message predictions and separate baseline/standalone/combined accounting']
    report['method']['individualTestErrorsInspected'] = True
    report['method']['testLabelsUsedForTrainingOrThresholds'] = False
    report['method']['gateBudgetAppliesTo'] = 'standalone MailGuard; combined can inherit Rspamd false positives'
    report['warning'] = ('Reused v40 test: engineering comparison only. Public historical rendered mail, '
                         'not current production SMTP traffic. Improvement is not guaranteed.')
    report['modes'] = modes
    report['v40Reference'] = {
        'run': 37205339677, 'target95': {'tp': 24029, 'fn': 971, 'fp': 36},
        'safe': {'tp': 18046, 'fn': 6954, 'fp': 3},
        'note': 'Original v40 had an unhealthy zero-recall Rspamd baseline.'}
    (REPORTS / 'v41-audited.json').write_text(json.dumps(report, indent=2))
    lines = ['# MailGuard v41: same v40 50k, audited Rspamd', '', report['warning'], '',
             '| Mode | Recall | FN | FP | FP rate |', '|---|---:|---:|---:|---:|']
    for name, metric in [('Rspamd + Bayes', report['rspamdBayes'])] + [
        (f'{name} {kind}', value[kind]) for name, value in modes.items()
        for kind in ('standalone', 'combined')]:
        lines.append(f"| {name} | {metric['recall']:.2%} | {metric['falseNegatives']} | "
                     f"{metric['falsePositives']} | {metric['fpr']:.3%} |")
    lines += ['', 'Standalone uses Rspamd features but can veto its decision. Combined is the OR '
              'of Rspamd and standalone, matching v40 policy. Validation FP budgets constrain '
              'standalone only; they do not guarantee a test FP rate.', '',
              'V40 target-95 reference: 96.116% recall, 971 FN, 36 FP. '
              'V40 safe reference: 72.184% recall, 6954 FN, 3 FP.', '',
              'See v41-rspamd-health.json and v41-*-predictions.jsonl for audit evidence.']
    text = '\n'.join(lines) + '\n'
    (REPORTS / 'v41-audited.md').write_text(text)
    print(text, flush=True)


def main():
    REPORTS.mkdir(exist_ok=True)
    data, audit = load_frozen_v40('.cache/v40-fresh50k')
    HEALTH['dataset'] = audit
    save_health()
    # Never reconstruct from a mutable online corpus on a cache miss.
    v40.prepare_v40 = lambda: data
    v40.OUT_MODELS = Path('models/v41')
    v40.write_report = write_report
    b.scan_one = strict_scan
    b.scan_many = audited_scan_many
    b.learn = audited_learn
    v40.v30.apply_balanced_guard = capture_guard
    v40.capture_metrics = capture_metrics
    v40.main()

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        HEALTH['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        save_health()
        raise
