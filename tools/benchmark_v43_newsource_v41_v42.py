#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import benchmark_v40_fresh50k as v40
import benchmark_v41_audited as v41
import train_v42_night_hybrid as v42
from rspamd_benchmark_audit import load_frozen_v40

REPORTS = Path('reports')
CACHE = Path('.cache/v43-newsource')
EML_DIR = CACHE / 'eml'
MODELS = Path('models/v43')
V42_CHECKPOINT = Path('.cache/v42-artifact/models/v42-night/byte-hybrid-best.pt')

DECISIONS = []
METRIC_CALLS = []


def clean_text(value):
    if value is None:
        return ''
    value = unicodedata.normalize('NFKC', str(value)).replace('\x00', ' ')
    return value.strip()


def one_line(value):
    return re.sub(r'[\r\n]+', ' ', clean_text(value)).strip()


def normalized_identity(subject, body):
    text = one_line(subject).casefold() + '\n' + clean_text(body).casefold()
    text = re.sub(r'\s+', ' ', text).strip()
    return hashlib.sha256(text.encode('utf-8', 'replace')).hexdigest()


def extract_identity(path):
    raw = path.read_bytes()
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    subject = str(msg.get('subject') or '')
    bodies = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get_content_disposition() == 'attachment':
                continue
            if part.get_content_type() not in ('text/plain', 'text/html'):
                continue
            try:
                bodies.append(part.get_content())
            except Exception:
                payload = part.get_payload(decode=True) or b''
                bodies.append(payload.decode('utf-8', 'replace'))
    else:
        try:
            bodies.append(msg.get_content())
        except Exception:
            payload = msg.get_payload(decode=True) or raw
            bodies.append(payload.decode('utf-8', 'replace'))
    return normalized_identity(subject, '\n'.join(map(str, bodies)))


def render_eml(subject, body, identity):
    msg = EmailMessage(policy=policy.SMTP)
    msg['From'] = 'sender@benchmark.invalid'
    msg['To'] = 'recipient@benchmark.invalid'
    msg['Subject'] = one_line(subject)[:900]
    msg['Message-ID'] = f'<{identity[:32]}@benchmark.invalid>'
    msg.set_content(clean_text(body), charset='utf-8')
    return msg.as_bytes()


def candidate(source, y, subject, body):
    subject = clean_text(subject)
    body = clean_text(body)
    if len(subject) + len(body) < 20:
        return None
    identity = normalized_identity(subject, body)
    return dict(source=source, y=int(y), subject=subject, body=body, identity=identity)


def unique_candidates(rows, blocked):
    out = {}
    for row in rows:
        if row is None or row['identity'] in blocked:
            continue
        out.setdefault(row['identity'], row)
    return [out[k] for k in sorted(out)]


def prepare_new_test(frozen):
    from datasets import load_dataset

    REPORTS.mkdir(exist_ok=True)
    EML_DIR.mkdir(parents=True, exist_ok=True)

    old_ids = set()
    old_total = sum(len(frozen[s]) for s in ('train', 'val', 'test'))
    done = 0
    for split in ('train', 'val', 'test'):
        for row in frozen[split]:
            old_ids.add(extract_identity(row['path']))
            done += 1
            if done % 10000 == 0:
                print('v43 old-content identity', done, '/', old_total, flush=True)

    joe_ds = load_dataset('renemel/joephishing_labeled_phishing', split='train')
    joe = []
    for row in joe_ds:
        subject = clean_text(row.get('Subject'))
        body = clean_text(row.get('Body'))
        if 'FOLDER INTERNAL DATA' in subject.upper() or 'not a real message' in body.lower():
            continue
        joe.append(candidate('JoePhishing-2021', 1, subject, body))
    joe = unique_candidates(joe, old_ids)

    biz_ds = load_dataset('wardacoder/business-email-dataset', split='train')
    business = []
    for row in biz_ds:
        output = clean_text(row.get('output'))
        match = re.search(r'(?im)^\s*subject\s*:\s*(.+?)\s*$', output)
        subject = match.group(1) if match else 'Business correspondence'
        business.append(candidate('BusinessSynthetic-2025', 0, subject, output))
    business = unique_candidates(business, old_ids)

    turkish_ds = load_dataset('anilguven/turkish_spam_email')
    turkish = []
    for split in turkish_ds:
        for row in turkish_ds[split]:
            turkish.append(candidate(
                'TurkishMail', int(row.get('labels') or 0), '', row.get('text')
            ))
    turkish = unique_candidates(turkish, old_ids)

    pair_n = min(1500, len(joe), len(business))
    if pair_n < 1000:
        raise RuntimeError(f'Not enough new-source English rows after dedupe: {pair_n}')
    selected = joe[:pair_n] + business[:pair_n] + turkish

    by_id = {}
    for row in selected:
        by_id.setdefault(row['identity'], row)
    selected = [by_id[k] for k in sorted(by_id)]

    rows = []
    manifest_rows = []
    for i, item in enumerate(selected):
        path = EML_DIR / f'test-{i:05d}.eml'
        path.write_bytes(render_eml(item['subject'], item['body'], item['identity']))
        rows.append(dict(path=path, y=item['y'], source=item['source']))
        manifest_rows.append(dict(
            file=path.name, y=item['y'], source=item['source'],
            contentSha256=item['identity'],
        ))

    counts = Counter((r['source'], r['y']) for r in rows)
    manifest = {
        'version': 43,
        'purpose': 'new-source evaluation only; labels not used for training or thresholds',
        'oldCorpusIdentitiesChecked': len(old_ids),
        'datasets': [
            'renemel/joephishing_labeled_phishing',
            'wardacoder/business-email-dataset',
            'anilguven/turkish_spam_email',
        ],
        'englishPairCapPerClass': pair_n,
        'counts': {f'{s}|{y}': n for (s, y), n in sorted(counts.items())},
        'test': manifest_rows,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('v43 new test', len(rows), 'spam', sum(r['y'] for r in rows),
          'ham', sum(not r['y'] for r in rows), 'counts', dict(counts), flush=True)
    return rows, manifest


def capture_guard(base_rspamd, score, ham_risk, agreement, gate):
    primary = ((score >= gate['primaryThreshold'])
               & (ham_risk <= gate['primaryMaxHamRisk'])
               & (agreement >= gate['primaryMinAgreement']))
    override = ((score >= gate['overrideThreshold'])
                & (ham_risk <= gate['overrideMaxHamRisk'])
                & (agreement >= gate['overrideMinAgreement']))
    standalone = primary | override
    combined = base_rspamd | standalone
    DECISIONS.append(dict(
        name=gate['name'], score=score.copy(), hamRisk=ham_risk.copy(),
        agreement=agreement.copy(), primary=primary, override=override,
        baseline=base_rspamd.copy(), standalone=standalone, combined=combined,
    ))
    return combined


def capture_metrics(rows, pred):
    result = v40._ORIGINAL_METRICS(rows, pred)
    METRIC_CALLS.append((rows, np.asarray(pred, dtype=bool).copy(), dict(result)))
    return result


def simple_metrics(rows, pred):
    y = np.asarray([bool(r['y']) for r in rows], dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int((y & pred).sum())
    fn = int((y & ~pred).sum())
    fp = int((~y & pred).sum())
    tn = int((~y & ~pred).sum())
    spam = tp + fn
    ham = fp + tn
    return {
        'count': len(rows), 'spam': spam, 'ham': ham,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'recall': tp / spam if spam else None,
        'precision': tp / (tp + fp) if (tp + fp) else None,
        'fpr': fp / ham if ham else None,
    }


def by_source(rows, pred):
    out = {}
    sources = sorted({r['source'] for r in rows})
    pred = np.asarray(pred, dtype=bool)
    for source in sources:
        idx = np.asarray([r['source'] == source for r in rows], dtype=bool)
        subrows = [r for r, keep in zip(rows, idx) if keep]
        out[source] = simple_metrics(subrows, pred[idx])
    return out


def fmt_pct(x, digits=2):
    return '—' if x is None else f'{100*x:.{digits}f}%'


def main():
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    frozen, frozen_audit = load_frozen_v40('.cache/v40-fresh50k')
    new_test, manifest = prepare_new_test(frozen)
    data = {'train': frozen['train'], 'val': frozen['val'], 'test': new_test}

    v41.HEALTH = {'v43FrozenV40': frozen_audit, 'v43NewSourceManifest': manifest}
    v41.REPORTS = REPORTS
    v40.prepare_v40 = lambda: data
    v40.EXPECTED_TEST = len(new_test)
    v40.OUT_MODELS = MODELS
    v40.write_report = lambda: None
    v40.v30.apply_balanced_guard = capture_guard
    v40.capture_metrics = capture_metrics

    v41.b.scan_one = v41.strict_scan
    v41.b.scan_many = v41.audited_scan_many
    v41.b.learn = v41.audited_learn

    v40.main()

    if len(DECISIONS) != 4 or not METRIC_CALLS:
        raise RuntimeError('Unexpected v41 evaluation shape in v43')
    target = next(d for d in DECISIONS if d['name'] == 'target-95')
    test_rows = METRIC_CALLS[-1][0]
    v41_pred = np.asarray(target['standalone'], dtype=bool)
    if len(test_rows) != len(v41_pred):
        raise RuntimeError('v41 prediction length mismatch')

    if not V42_CHECKPOINT.exists():
        raise FileNotFoundError(f'Missing v42 checkpoint: {V42_CHECKPOINT}')
    checkpoint = torch.load(V42_CHECKPOINT, map_location='cpu', weights_only=False)
    model = v42.ByteHybridNet()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    v42_ds = v42.RawMailDataset(new_test)
    v42_loader = DataLoader(v42_ds, batch_size=96, shuffle=False, num_workers=0)
    _, v42_prob = v42.score_model(model, v42_loader)
    threshold = float(checkpoint['meta']['valLowFp']['threshold'])
    v42_pred = np.asarray(v42_prob >= threshold, dtype=bool)

    rescue_pred = v41_pred | v42_pred
    y = np.asarray([bool(r['y']) for r in test_rows], dtype=bool)
    rescued_fn = int((y & ~v41_pred & v42_pred).sum())
    added_fp = int((~y & ~v41_pred & v42_pred).sum())

    modes = {
        'v41-target95-standalone': simple_metrics(test_rows, v41_pred),
        'v42-fixed-lowfp': simple_metrics(test_rows, v42_pred),
        'v41-plus-v42-rescue': simple_metrics(test_rows, rescue_pred),
    }
    source_metrics = {
        name: by_source(test_rows, pred)
        for name, pred in (
            ('v41-target95-standalone', v41_pred),
            ('v42-fixed-lowfp', v42_pred),
            ('v41-plus-v42-rescue', rescue_pred),
        )
    }

    report = {
        'version': 'v43-new-source-v41-plus-v42',
        'dataset': manifest,
        'v42Checkpoint': {
            'architecture': checkpoint.get('architecture'),
            'epoch': checkpoint['meta']['epoch'],
            'threshold': threshold,
            'validationLowFp': checkpoint['meta']['valLowFp'],
        },
        'method': {
            'v41Recipe': 'v40 recipe + v41 audited Rspamd transport; target-95 standalone',
            'v42Use': 'frozen epoch-3 checkpoint; fixed old-validation threshold; no new-source tuning',
            'combine': 'OR rescue: v41 standalone OR v42 fixed-threshold spam',
            'testLabelsUsedForTrainingOrThresholds': False,
            'contentOverlapGuard': 'normalized subject+body exact SHA256 against all frozen v40 train/val/test',
        },
        'modes': modes,
        'bySource': source_metrics,
        'rescue': {'recoveredV41FalseNegatives': rescued_fn, 'addedFalsePositives': added_fp},
    }
    (REPORTS / 'v43-newsource.json').write_text(json.dumps(report, indent=2))

    with (REPORTS / 'v43-newsource-predictions.jsonl').open('w') as out:
        for i, row in enumerate(test_rows):
            rec = {
                'id': hashlib.sha256(row['raw']).hexdigest(),
                'source': row['source'], 'label': int(row['y']),
                'v41Spam': bool(v41_pred[i]),
                'v42Probability': float(v42_prob[i]),
                'v42Spam': bool(v42_pred[i]),
                'rescueSpam': bool(rescue_pred[i]),
                'rspamdScore': float(row['rscore']),
                'rspamdAction': row['action'],
            }
            out.write(json.dumps(rec) + '\n')

    lines = [
        '# MailGuard v43: v41 + v42 on new source families', '',
        'No new-source labels were used for training, threshold selection, or tuning.', '',
        '| Mode | Recall | FN | FP | FP rate | Precision |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for name, metric in modes.items():
        lines.append(
            f"| {name} | {fmt_pct(metric['recall'])} | {metric['fn']} | "
            f"{metric['fp']} | {fmt_pct(metric['fpr'], 3)} | {fmt_pct(metric['precision'], 3)} |"
        )
    lines += [
        '',
        f"V42 fixed threshold from its old validation: **{threshold:.6f}**.",
        f"V42 rescued **{rescued_fn}** v41 false negatives and added **{added_fp}** new false positives.",
        '',
        'Per-source results for the combined rescue:', '',
        '| Source | N | Spam | Ham | Recall | FP | FP rate |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for source, metric in source_metrics['v41-plus-v42-rescue'].items():
        lines.append(
            f"| {source} | {metric['count']} | {metric['spam']} | {metric['ham']} | "
            f"{fmt_pct(metric['recall'])} | {metric['fp']} | {fmt_pct(metric['fpr'], 3)} |"
        )
    lines += [
        '',
        'JoePhishing is phishing-only; BusinessSynthetic is ham-only; TurkishMail contains both classes. '
        'The overall aggregate is therefore useful as an engineering stress test, while per-source rows '
        'should be read alongside it.',
    ]
    text = '\n'.join(lines) + '\n'
    (REPORTS / 'v43-newsource.md').write_text(text)
    print(text, flush=True)


if __name__ == '__main__':
    main()
