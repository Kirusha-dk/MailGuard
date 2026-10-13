"""Freeze unused October spam, blocking every previous development/test family."""
import csv
import gzip
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
import prepare_v59_new_mail as collector_lib
from prepare_v60_new_mail import block
from train_v55_multilingual import corpus_rows, file_sha, campaign_key
from evaluate_v54_unique500k import Pool


def main():
    out = Path('.cache/v61-new'); out.mkdir(parents=True, exist_ok=False)
    pool = object.__new__(Pool); pool.db = sqlite3.connect('.cache/v61-blocking/blocked.sqlite'); pool.stats = Counter()
    count = 0
    for root, splits in [(Path('.cache/v60-data'), ('train','select','cal','test')), (Path('.cache/v59-data'), ('test',))]:
        provenance = json.loads((root/'provenance.json').read_text())
        for split in splits:
            if file_sha(root/f'{split}.jsonl.gz') != provenance['splitHashes'][split]:
                raise ValueError('Previous split hash changed')
            for row in corpus_rows(root/f'{split}.jsonl.gz'):
                block(pool,row); count += 1
    from rspamd_benchmark_audit import load_frozen_v40
    from v59_mail import from_eml
    frozen,_ = load_frozen_v40('.cache/v40-fresh50k')
    raw_count = 0
    for rows in frozen.values():
        for entry in rows:
            raw = entry['path'].read_bytes(); raw_count += 1
            pool.db.execute('INSERT OR IGNORE INTO keys VALUES (?,NULL,NULL)',('raw:'+hashlib.sha256(raw).hexdigest(),))
            try:
                raw.decode('utf-8','strict'); recover = b'\x1b$' in raw
            except UnicodeError:
                recover = True
            if recover:
                parsed = from_eml(raw); pool.block(parsed)
                pool.db.execute('INSERT OR IGNORE INTO keys VALUES (?,NULL,NULL)',('family:'+campaign_key(parsed['ngram_text']),))
    from datasets import load_dataset
    from train_v47_phishing_aware import canonical_fields
    extra = load_dataset('JinqiangDing/seven-phishing-email-datasets',split='test')
    for r in extra:
        parsed = canonical_fields(r.get('subject') or '',r.get('text') or '',r.get('sender') or 'unknown@example.invalid')
        pool.block(parsed)
        pool.db.execute('INSERT OR IGNORE INTO keys VALUES (?,NULL,NULL)',('family:'+campaign_key(parsed['ngram_text']),))
    pool.db.commit()
    print('All previous rows blocked',count,'raw',raw_count,flush=True)
    collector_lib.family_split = lambda family,source:'test'
    collector = collector_lib.Collector(pool)
    work = out/'downloads'; work.mkdir()
    source = collector_lib.collect_month(collector,10,work)
    rows = sorted([r for f,r in collector.rows.items() if f not in collector.conflicts],key=lambda r:r['identity'])
    if not rows: raise ValueError('No new October families')
    with gzip.open(out/'test.jsonl.gz','wt',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    with (out/'manifest.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['identity','family','source','label','rawSha256'])
        for r in rows: writer.writerow([r['identity'],r['family'],r['source'],r['y'],r['rawSha256']])
    report=dict(version=61,count=len(rows),ham=0,spam=len(rows),sourceArchive=source,
                freezeDate='2026-10-13',previousRowsBlocked=count,rawPreviousRowsBlocked=raw_count,
                wholeSevenTestRowsBlocked=len(extra),exclusions=dict(collector.stats),
                testSha256=file_sha(out/'test.jsonl.gz'),manifestSha256=file_sha(out/'manifest.csv'),
                testPreviouslyScored=False,
                caveat='Partial October snapshot, only previously unused spam-trap families. No new gold HAM; false positives must be measured separately on explicitly reused HAM tests. Semantic relatives may remain.')
    (out/'provenance.json').write_text(json.dumps(report,indent=2)); pool.db.close()
    print('LOCKED NEW OCTOBER TEST',json.dumps(report),flush=True)


if __name__=='__main__': main()
