#!/usr/bin/env python3
"""Frozen v54 stress test: real labelled emails, conservative content deduplication."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import shutil
import sqlite3
import subprocess
import tempfile
import unicodedata
import zipfile
from collections import Counter
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

from train_v47_phishing_aware import canonical_fields, from_eml

CSDMC_REV = '8e9687170dc683a66ba9bae01839cc3eacb85e08'
CSDMC_URL = f'https://codeload.github.com/warmspringwinds/email_spam_filtering/zip/{CSDMC_REV}'
SEED = 'mailguard-unique500k-20261010:'
WARNING = ('Historical public mail, heavily spam-skewed. SpamArchive labels come from spam traps; '
           'CSDMC supplies labelled ham. This is a stress test, not a representative customer benchmark. '
           'FP rate uses the actual ham denominator. Exact body and normalized-template duplicates '
           'are excluded; semantic near-duplicates may still remain. No threshold tuning on these labels.')


def digest(text):
    return hashlib.sha256(text.encode('utf-8', 'replace')).hexdigest()


def content_keys(row):
    # Ignore envelope, sender, subject, tracking URLs, addresses and changing numbers.
    body = row['ngram_text'].split('\nBODY: ', 1)[-1]
    body = unicodedata.normalize('NFKC', body).casefold()
    body = re.sub('[\u200b-\u200f\ufeff]', '', body)
    body = re.sub(r'\s+', ' ', body).strip()
    template = re.sub(r'https?://\S+|www\.\S+', '<url>', body)
    template = re.sub(r'[\w.+-]+@[\w.-]+\.[\w-]+', '<email>', template)
    template = re.sub(r'\d+', '<number>', template)
    return ('identity:' + row['identity'], 'body:' + digest(body), 'template:' + digest(template)), body


class Pool:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=NORMAL')
        self.db.execute('CREATE TABLE samples(id INTEGER PRIMARY KEY, identity TEXT, bodykey TEXT, templatekey TEXT, text TEXT, y INTEGER, source TEXT, origin TEXT, rank TEXT, valid INTEGER DEFAULT 1)')
        self.db.execute('CREATE TABLE keys(key TEXT PRIMARY KEY, sample INTEGER, y INTEGER) WITHOUT ROWID')
        self.stats = Counter()

    def block(self, row):
        keys, _ = content_keys(row)
        self.db.executemany('INSERT OR IGNORE INTO keys VALUES (?,NULL,NULL)', [(k,) for k in keys])

    def add(self, row, y, source, origin):
        if type(y) is not int or y not in (0, 1):
            raise ValueError('Only verified binary labels accepted')
        self.stats['seen'] += 1
        keys, body = content_keys(row)
        if len(body) < 20:
            self.stats['shortBody'] += 1
            return
        previous = self.db.execute('SELECT sample,y FROM keys WHERE key IN (?,?,?)', keys).fetchall()
        if previous:
            ids = {r[0] for r in previous if r[0] is not None}
            blocked = any(r[0] is None for r in previous)
            conflict = any(r[1] is not None and r[1] != y for r in previous)
            # A variant bridging two retained rows invalidates both conservatively.
            if blocked or conflict or len(ids) > 1:
                self.db.executemany('UPDATE samples SET valid=0 WHERE id=?', [(i,) for i in ids])
                self.stats['conflictingOrBridged'] += 1
            sample = None if blocked else min(ids) if ids else None
            label = None if blocked else previous[0][1]
            self.db.executemany('INSERT OR IGNORE INTO keys VALUES (?,?,?)', [(k, sample, label) for k in keys])
            self.stats['oldContent' if blocked else 'duplicateCandidate'] += 1
            return
        cur = self.db.execute('INSERT INTO samples(identity,bodykey,templatekey,text,y,source,origin,rank) VALUES (?,?,?,?,?,?,?,?)',
                              (row['identity'], keys[1], keys[2], row['ngram_text'], y, source, origin, digest(SEED + row['identity'])))
        self.db.executemany('INSERT INTO keys VALUES (?,?,?)', [(k, cur.lastrowid, y) for k in keys])
        self.stats['retained'] += 1

    def count(self):
        return self.db.execute('SELECT COUNT(*) FROM samples WHERE valid=1').fetchone()[0]

    def close(self):
        self.db.commit()
        self.db.close()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    req = Request(url, headers={'User-Agent': 'MailGuard-public-research/1.0'})
    with urlopen(req, timeout=180) as src, dest.open('wb') as dst:
        shutil.copyfileobj(src, dst)
    return {'url': url, 'sha256': sha_file(dest), 'bytes': dest.stat().st_size}


def csdmc_label(value):
    # Original CSDMC README: 1721 ham=1; 779 spam=0.
    if value not in ('0', '1'):
        raise ValueError('Invalid CSDMC label')
    return 1 - int(value)


def collect_csdmc(pool, work):
    path = work / 'csdmc.zip'
    provenance = download(CSDMC_URL, path)
    with zipfile.ZipFile(path) as z:
        prefix = z.namelist()[0].split('/')[0] + '/'
        labels = list(csv.DictReader(io.StringIO(z.read(prefix + 'spam-mail.tr.label').decode())))
        counts = Counter(r['Prediction'] for r in labels)
        if counts != {'1': 1721, '0': 779} or len({r['Id'] for r in labels}) != 2500:
            raise RuntimeError(f'Unexpected CSDMC label inventory: {counts}')
        for r in labels:
            name = f'TR/TRAIN_{int(r["Id"])}.eml'
            pool.add(from_eml(z.read(prefix + name)), csdmc_label(r['Prediction']), 'CSDMC2010', name)
    path.unlink()
    pool.db.commit()
    return provenance


def has_mail_headers(raw):
    try:
        message = BytesParser(policy=policy.default).parsebytes(raw, headersonly=True)
        return bool(message.get("from") and any(message.get(k) for k in ("date", "subject", "received")))
    except Exception:
        return False


def collect_archive(pool, year, work):
    path = work / f'{year}.7z'
    provenance = download(f'https://untroubled.org/spam/{year}.7z', path)
    # Never execute downloaded content or allow archive paths to escape its directory.
    listing = subprocess.run(['7z', 'l', '-slt', str(path)], check=True, capture_output=True, text=True).stdout
    entries = listing.split('----------\n', 1)[-1]
    for entry in entries.split('\n\n'):
        if 'Symbolic Link =' in entry or 'Hard Link =' in entry:
            raise ValueError('Archive link rejected')
        for line in entry.splitlines():
            if line.startswith('Path = '):
                p = PurePosixPath(line[7:].replace('\\', '/'))
                if p.is_absolute() or '..' in p.parts:
                    raise ValueError('Unsafe archive path')
    dest = work / 'extracted'
    dest.mkdir()
    subprocess.run(['7z', 'x', '-y', '-bd', '-bso0', '-bsp0', f'-o{dest}', str(path)], check=True)
    read = 0
    for file in sorted(dest.rglob('*')):
        if file.is_symlink():
            raise ValueError('Archive symlink rejected')
        if not file.is_file():
            continue
        if file.stat().st_size > 8 * 1024 * 1024:
            pool.stats['largeFile'] += 1
            continue
        raw = file.read_bytes()
        if not has_mail_headers(raw):
            pool.stats['notRfc822Mail'] += 1
            continue
        row = from_eml(raw)
        pool.add(row, 1, f'SpamArchive-{year}', str(file.relative_to(dest)))
        read += 1
        if read % 10000 == 0:
            pool.db.commit()
            print(f'{year}: processed {read}, unique available {pool.count()}', flush=True)
    pool.db.commit()
    provenance['messagesRead'] = read
    shutil.rmtree(dest)
    path.unlink()
    return provenance


def block_previous(pool, cache):
    from datasets import load_dataset
    from rspamd_benchmark_audit import load_frozen_v40
    frozen, audit = load_frozen_v40(cache)
    for split in ('train', 'val', 'test'):
        for r in frozen[split]:
            pool.block(from_eml(r['path'].read_bytes()))
    fingerprints = {}
    # Block the ENTIRE prior source pools, not just rows selected for earlier tests.
    seven = load_dataset('JinqiangDing/seven-phishing-email-datasets', split='train')
    fingerprints['seven'] = seven._fingerprint
    for r in seven:
        pool.block(canonical_fields(r.get('subject') or '', r.get('text') or '', r.get('sender') or 'unknown@example.invalid'))
    for split, rows in load_dataset('SetFit/enron_spam').items():
        fingerprints['SetFit/' + split] = rows._fingerprint
        for r in rows:
            pool.block(canonical_fields('', r['text'], 'unknown@example.invalid'))
    pool.db.commit()
    print('Blocked previous training, validation and engineering sources', flush=True)
    return {'frozenAudit': audit, 'datasetFingerprints': fingerprints}


def collect(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'selected.jsonl.gz').exists():
        raise RuntimeError('Selection already frozen; use --phase score or a new output directory')
    db = out / 'pool.sqlite'
    if db.exists():
        raise RuntimeError('Partial pool exists; preserve it and use a new output directory')
    pool = Pool(db)
    provenance = []
    blocked = block_previous(pool, args.cache)
    try:
        with tempfile.TemporaryDirectory(prefix='unique-mail-') as temp:
            work = Path(temp)
            provenance.append(collect_csdmc(pool, work))
            for year in range(2025, 1999, -1):
                if pool.count() >= args.count:
                    break
                print(f'Downloading SpamArchive {year}', flush=True)
                provenance.append(collect_archive(pool, year, work))
                (out / 'collection-progress.json').write_text(json.dumps({'available': pool.count(), 'exclusions': dict(pool.stats), 'sources': provenance}, indent=2))
        available = pool.count()
        columns = 'identity,bodykey,templatekey,text,y,source,origin'
        cursor = pool.db.execute(f'SELECT {columns} FROM samples WHERE valid=1 ORDER BY rank LIMIT ?', (args.count,))
        count, labels, sources = 0, Counter(), Counter()
        with gzip.open(out / 'selected.jsonl.gz', 'wt', encoding='utf-8') as f, (out / 'manifest.csv').open('w', newline='') as mf:
            writer = csv.writer(mf)
            writer.writerow(['identity', 'bodykey', 'templatekey', 'label', 'source', 'origin'])
            for ident, bodykey, templatekey, text, y, source, origin in cursor:
                f.write(json.dumps({'identity': ident, 'ngram_text': text, 'y': y, 'source': source}, ensure_ascii=False) + '\n')
                writer.writerow([ident, bodykey, templatekey, y, source, origin])
                count += 1
                labels[y] += 1
                sources[source] += 1
        # Independently audit selected keys; no duplicated refill for a short pool.
        for key in ('identity', 'bodykey', 'templatekey'):
            unique = pool.db.execute(f'SELECT COUNT(DISTINCT {key}) FROM (SELECT {key} FROM samples WHERE valid=1 ORDER BY rank LIMIT ?)', (args.count,)).fetchone()[0]
            if unique != count:
                raise AssertionError(f'Duplicate selected {key}')
        report = {'requestedCount': args.count, 'selected': count, 'available': available,
                  'ham': labels[0], 'spam': labels[1], 'bySource': dict(sources), 'exclusions': dict(pool.stats),
                  'previousSources': blocked, 'sources': provenance, 'uniqueAuditPassed': True,
                  'selection': 'fixed seeded content hash; no labels or scores used to rank rows',
                  'selectedSha256': sha_file(out / 'selected.jsonl.gz'), 'warning': WARNING}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k not in ('previousSources', 'sources')}, indent=2), flush=True)
        if count < args.count:
            raise RuntimeError(f'Insufficient distinct mail: {count}/{args.count}; no duplicated refill')
        if not labels[0] or not labels[1]:
            raise RuntimeError('Both labelled classes required')
    finally:
        pool.close()
    # Preserve the selected corpus and provenance, not multi-GB dedup intermediates.
    db.unlink()
    for suffix in ('-wal', '-shm'):
        Path(str(db) + suffix).unlink(missing_ok=True)


def score(args):
    import numpy as np
    import torch
    from train_v47_phishing_aware import metrics_at
    from train_v54_margin import load_components, predict_margin_union
    out = Path(args.output)
    manifest = json.loads((out / 'manifest.json').read_text())
    if manifest['selected'] != args.count or not manifest['uniqueAuditPassed']:
        raise RuntimeError('Incomplete selection')
    if sha_file(out / 'selected.jsonl.gz') != manifest['selectedSha256']:
        raise RuntimeError('Frozen selection changed')
    models = list(Path(args.artifact_root).rglob('v54-full/model.joblib'))
    if len(models) != 1:
        raise RuntimeError(f'Expected one v54 full model, found {models}')
    model = models[0]
    hashes = {p.name: sha_file(p) for p in (model, model.parent / 'neural.pt')}
    artifact, neural = load_components(model.parent)
    if artifact.get('version') != 'v54-margin-rescue':
        raise RuntimeError('Unexpected model version')
    threshold = float(artifact['threshold'])
    torch.set_num_threads(4)
    ys, ps, sources, processed = [], [], [], 0
    with gzip.open(out / 'selected.jsonl.gz', 'rt', encoding='utf-8') as f, (out / 'predictions.csv').open('w', newline='') as pf:
        writer = csv.writer(pf)
        writer.writerow(['identity', 'source', 'label', 'score', 'predictedSpam'])
        batch = []
        def consume(rows):
            nonlocal processed
            scores = predict_margin_union(artifact, neural, [r['ngram_text'] for r in rows])
            if not np.isfinite(scores).all():
                raise ValueError('Non-finite model score')
            for r, p in zip(rows, scores):
                writer.writerow([r['identity'], r['source'], r['y'], float(p), int(p >= threshold)])
                ys.append(r['y']); ps.append(float(p)); sources.append(r['source'])
            processed += len(rows)
            pf.flush()
            if processed == 1000:
                smoke = metrics_at(np.asarray(ys), np.asarray(ps), threshold, sources)
                (out / 'smoke1000.json').write_text(json.dumps(smoke, indent=2))
                print('Smoke 1000:', json.dumps(smoke), flush=True)
            if processed % 10000 == 0 or processed == args.count:
                print(f'Frozen v54 scored {processed}/{args.count}', flush=True)
        for line in f:
            batch.append(json.loads(line))
            if len(batch) == 1000:
                consume(batch); batch = []
        if batch:
            consume(batch)
    if processed != args.count:
        raise AssertionError('Selected corpus count mismatch')
    if hashes != {p.name: sha_file(p) for p in (model, model.parent / 'neural.pt')}:
        raise AssertionError('Frozen model changed')
    metrics = metrics_at(np.asarray(ys), np.asarray(ps), threshold, sources)
    ham = metrics['fp'] + metrics['tn']
    report = {'count': processed, 'ham': ham, 'spam': metrics['tp'] + metrics['fn'],
              'modelHashes': hashes, 'threshold': threshold, 'thresholdUnchanged': True, 'modelRefit': False,
              'testLabelsUsedForTuning': False, 'metrics': metrics, 'minRecall': args.min_recall, 'maxFp': args.max_fp,
              'stressGoalPassed': metrics['recall'] >= args.min_recall and metrics['fp'] <= args.max_fp,
              'productionGoalValidated': False, 'warning': WARNING}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    md = (f'# Frozen v54: {processed:,} distinct emails\n\n{WARNING}\n\n'
          '| Emails | Ham | Spam | Spam recall | FP | FN | FP / ham |\n|---:|---:|---:|---:|---:|---:|---:|\n'
          f'| {processed} | {ham} | {report["spam"]} | {metrics["recall"]:.2%} | {metrics["fp"]} | {metrics["fn"]} | {metrics["fpr"]:.4%} |\n\n'
          f'Stress goal (recall >= {args.min_recall:.0%}, FP <= {args.max_fp}): {report["stressGoalPassed"]}. '
          'No production claim; model and threshold remained frozen.\n')
    (out / 'report.md').write_text(md)
    print(md, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=('collect', 'score'), required=True)
    p.add_argument('--count', type=int, default=500000)
    p.add_argument('--output', default='reports/v54-unique500k')
    p.add_argument('--cache', default='.cache/v40-fresh50k')
    p.add_argument('--artifact-root', default='.cache/v54-artifact')
    p.add_argument('--min-recall', type=float, default=.95)
    p.add_argument('--max-fp', type=int, default=50)
    args = p.parse_args()
    if args.count <= 0 or args.max_fp < 0:
        p.error('Invalid count or FP budget')
    (collect if args.phase == 'collect' else score)(args)


if __name__ == '__main__':
    main()
