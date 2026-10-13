#!/usr/bin/env python3
"""Freeze unused April development and May test mail; block all v59 rows."""
import csv
import gzip
import hashlib
import json
import re
import sqlite3
import tarfile
from collections import Counter
from pathlib import Path

import prepare_v59_new_mail as previous
from train_v55_multilingual import corpus_rows, file_sha, campaign_key
from evaluate_v54_unique500k import content_keys, download, has_mail_headers, Pool
from v59_mail import from_eml

OUT = Path('.cache/v60-data')
SPLITS = previous.SPLITS


def partition(family, source):
    if source.startswith('SpamArchive-2026-'):
        month = int(source.rsplit('-', 1)[1])
        if month == 5:
            return 'test'
        if month != 4:
            raise ValueError('v60 may download only previously unused April/May')
        n = int(family[-12:], 16) % 100
        return 'train' if n < 60 else 'select' if n < 80 else 'cal'
    if source.startswith('SpamAssassin-original-'):
        return 'test'
    raise ValueError(source)


def block(pool, row):
    keys = []
    for name in ('text', 'old_text'):
        if row.get(name):
            parsed = dict(identity=hashlib.sha256(row[name].encode()).hexdigest(), ngram_text=row[name])
            keys.extend(content_keys(parsed)[0])
            keys.append('family:' + campaign_key(row[name]))
    keys.extend(('identity:' + row['identity'], 'family:' + row['family']))
    if row.get('rawSha256'):
        keys.append('raw:' + row['rawSha256'])
    pool.db.executemany('INSERT OR IGNORE INTO keys VALUES (?,NULL,NULL)', [(k,) for k in keys])


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    pool = object.__new__(Pool)
    pool.db = sqlite3.connect('.cache/v59-blocking/blocked.sqlite')
    pool.stats = Counter()
    prior = Path('.cache/v59-data')
    provenance59 = json.loads((prior / 'provenance.json').read_text())
    count = 0
    for split in SPLITS:
        if file_sha(prior / f'{split}.jsonl.gz') != provenance59['splitHashes'][split]:
            raise ValueError('v59 data hash changed')
        for row in corpus_rows(prior / f'{split}.jsonl.gz'):
            block(pool, row); count += 1
    # Include the original Seven held-out split and charset aliases that were
    # added after the older blocking checkpoint. All recovered rows are blocked.
    from rspamd_benchmark_audit import load_frozen_v40
    from datasets import load_dataset
    from train_v47_phishing_aware import canonical_fields
    frozen, _ = load_frozen_v40('.cache/v40-fresh50k')
    for rows in frozen.values():
        for entry in rows:
            raw = entry['path'].read_bytes()
            parsed = from_eml(raw)
            pool.block(parsed)
            pool.db.execute('INSERT OR IGNORE INTO keys VALUES (?,NULL,NULL)', ('family:' + campaign_key(parsed['ngram_text']),))
            pool.db.execute('INSERT OR IGNORE INTO keys VALUES (?,NULL,NULL)', ('raw:' + hashlib.sha256(raw).hexdigest(),))
    for name, rows in load_dataset('JinqiangDing/seven-phishing-email-datasets').items():
        if name == 'train':
            continue
        for r in rows:
            parsed = canonical_fields(r.get('subject') or '', r.get('text') or '', r.get('sender') or 'unknown@example.invalid')
            pool.block(parsed)
            pool.db.execute('INSERT OR IGNORE INTO keys VALUES (?,NULL,NULL)', ('family:' + campaign_key(parsed['ngram_text']),))
    pool.db.commit()
    print('Blocked all prior data, including all v59 splits:', count, flush=True)
    previous.family_split = partition
    collector = previous.Collector(pool)
    work = OUT / 'downloads'; work.mkdir()
    archives = [previous.collect_month(collector, m, work) for m in (4, 5)]
    # Original labelled HAM is accepted only when its content/family is absent
    # from every prior source. Existing SpamAssassin copies are never called new.
    for stem, label in (('20030228_easy_ham', 0), ('20030228_easy_ham_2', 0), ('20030228_hard_ham', 0),
                        ('20030228_spam', 1), ('20050311_spam_2', 1)):
        archive = work / (stem + '.tar.bz2')
        source = download('https://spamassassin.apache.org/old/publiccorpus/' + archive.name, archive)
        read = 0
        with tarfile.open(archive, 'r:bz2') as tar:
            for member in tar:
                if not member.isfile() or member.size > 8 * 1024 * 1024:
                    continue
                raw = tar.extractfile(member).read()
                if has_mail_headers(raw):
                    collector.add(raw, label, 'SpamAssassin-original-' + stem, member.name); read += 1
        source['rawEmailsRead'] = read
        archives.append(source); archive.unlink()
        print(stem, read, 'new families accumulated', len(collector.rows), flush=True)
    rows_by_split = {}
    counts, fresh = {}, {}
    with (OUT / 'manifest.csv').open('w', newline='') as mf:
        writer = csv.writer(mf)
        writer.writerow(['identity', 'family', 'split', 'label', 'source', 'rawSha256', 'weakLabel'])
        for split in SPLITS:
            new = sorted((r for family, r in collector.rows.items() if family not in collector.conflicts and r['split'] == split), key=lambda r: r['identity'])
            # v59 TEST is reserved exclusively for regression, never fitting.
            rows = ([] if split == 'test' else list(corpus_rows(prior / f'{split}.jsonl.gz'))) + new
            rows_by_split[split] = rows
            with gzip.open(OUT / f'{split}.jsonl.gz', 'wt', encoding='utf-8') as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
                    writer.writerow([r['identity'], r['family'], split, r['y'], r['source'], r.get('rawSha256'), r.get('weakLabel', False)])
            counts[split] = dict(Counter(f'{r["source"]}:{r["y"]}' for r in rows))
            fresh[split] = dict(Counter(f'{r["source"]}:{r["y"]}' for r in new))
    for key in ('identity', 'family', 'rawSha256'):
        sets = [{r[key] for r in rows if r.get(key)} for rows in rows_by_split.values()]
        if any(sets[i] & sets[j] for i in range(4) for j in range(i + 1, 4)):
            raise AssertionError('Cross-split overlap: ' + key)
    evidence = dict(version=60, counts=counts, newCounts=fresh, sourceArchives=archives,
                    priorV59RowsBlocked=count, priorV59ManifestSha256=provenance59['manifestSha256'],
                    exclusions=dict(collector.stats), manifestSha256=file_sha(OUT / 'manifest.csv'),
                    splitHashes={s: file_sha(OUT / f'{s}.jsonl.gz') for s in SPLITS},
                    testPreviouslyScored=False, v59TestAddedToTraining=False, weakHamLabelsUsedOutsideTrain=False,
                    caveat='New May spam-trap families and unused original SpamAssassin messages, after prior-source blocking. HAM count may be small; separate reused Chinese and English regression results must not be presented as new. Historical public mail is not customer mail; semantic relatives may remain.')
    (OUT / 'provenance.json').write_text(json.dumps(evidence, indent=2))
    pool.db.close()
    print('LOCKED v60 counts', json.dumps(fresh), flush=True)


if __name__ == '__main__':
    main()
