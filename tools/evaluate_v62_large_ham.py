#!/usr/bin/env python3
"""Frozen v62 FP evaluation on every eligible distinct public gold HAM message."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sqlite3
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

from evaluate_v62_500k import MODEL_SHA, file_sha, model_file, rows, measure

SEVEN = 'JinqiangDing/seven-phishing-email-datasets'
CHINESE = 'https://cdn.aibydoing.com/aibydoing/files/trec06c.zip'
CAVEAT = ('Historical public gold labels, not customer mail. All TRAIN/SELECT/CAL identities and '
          'campaign families are excluded, including legacy-decoding aliases. Exact identity, '
          'body and normalized-template duplicates and contradictory labels are excluded. '
          'Sources have been inspected during engineering; additional messages are not an '
          'independent production benchmark. Thread relatives may remain. No fitting or '
          'threshold selection occurs in this evaluation; HAM-only results cannot establish spam recall.')


class HamPool:
    """Keep HAM once; process SPAM too so conflicting labels invalidate earlier HAM."""
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE keys(key TEXT PRIMARY KEY, sample INTEGER, label INTEGER, blocked INTEGER) WITHOUT ROWID')
        self.db.execute('CREATE TABLE samples(id INTEGER PRIMARY KEY, identity TEXT UNIQUE, family TEXT, text TEXT, source TEXT, origin TEXT, valid INTEGER DEFAULT 1)')
        self.stats = Counter()

    def block(self, keys):
        self.db.executemany('INSERT OR IGNORE INTO keys VALUES (?,NULL,NULL,1)', [(k,) for k in keys])

    def add(self, keys, development_aliases, label, identity, family, text, source, origin):
        if type(label) is not int or label not in (0, 1):
            raise ValueError('Binary gold labels required')
        self.stats['hamRead' if label == 0 else 'spamRead'] += 1
        keys = tuple(sorted(set(keys)))
        aliases = tuple(sorted(set(keys) | set(development_aliases)))
        previous = self.db.execute('SELECT sample,label,blocked FROM keys WHERE key IN ('+','.join('?' for _ in aliases)+')', aliases).fetchall()
        ids = {p[0] for p in previous if p[0] is not None}
        blocked = any(p[2] for p in previous)
        conflict = any(p[1] is not None and p[1] != label for p in previous)
        if previous:
            if blocked or conflict or len(ids) > 1:
                self.db.executemany('UPDATE samples SET valid=0 WHERE id=?', [(i,) for i in ids])
            key = 'developmentOverlap' if blocked else 'conflictingLabelOrBridge' if conflict or len(ids)>1 else 'duplicateContent'
            self.stats[key+(':ham' if label == 0 else ':spam')] += 1
            sample = None if blocked else min(ids) if ids else None
            stored_label = None if blocked else previous[0][1]
            inserted = aliases if blocked else keys
            self.db.executemany('INSERT OR IGNORE INTO keys VALUES (?,?,?,?)', [(k,sample,stored_label,int(blocked)) for k in inserted])
            return
        sample = None
        if label == 0:
            cur = self.db.execute('INSERT INTO samples(identity,family,text,source,origin) VALUES (?,?,?,?,?)',
                                  (identity,family,text,source,origin))
            sample = cur.lastrowid
        self.db.executemany('INSERT INTO keys VALUES (?,?,?,0)', [(k,sample,label) for k in keys])


def prepare(args):
    import joblib
    import numpy as np
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from train_v47_phishing_aware import canonical_fields, from_eml as legacy_parse
    from train_v55_multilingual import campaign_key, views
    from v61_semantic import training_pool
    from v59_mail import from_eml
    from evaluate_v54_unique500k import content_keys
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    model = model_file(args.model)
    if file_sha(model) != MODEL_SHA:
        raise ValueError('Frozen v62 SHA mismatch')
    artifact = joblib.load(model)
    data = Path('.cache/v60-data')
    provenance = json.loads((data/'provenance.json').read_text())
    if artifact['splitManifestSha256'] != provenance['manifestSha256']:
        raise ValueError('Development manifest differs from model')
    pool = HamPool(out/'pool.sqlite')
    split_counts = {}
    for split in ('train','select','cal'):
        path = data/f'{split}.jsonl.gz'
        if file_sha(path) != provenance['splitHashes'][split]:
            raise ValueError('Development split changed')
        batch = list(rows(path))
        split_counts[split] = len(batch)
        for r in batch:
            pool.block(['identity:'+r['identity'], 'family:'+r['family']])
            for field in ('text', 'old_text'):
                if field in r:
                    p = dict(identity=r['identity'], ngram_text=r[field])
                    pool.block([*content_keys(p)[0], 'family:'+campaign_key(r[field])])
        if split == 'train':
            train = training_pool(batch)
            ix = np.random.default_rng(62).permutation(len(train))
            digest = hashlib.sha256('\n'.join(train[i]['identity'] for i in ix).encode()).hexdigest()
            if len(train) != artifact['trainCount'] or digest != artifact['trainIdentitySha256']:
                raise ValueError('Actual training pool differs from model')
    del artifact, train, batch
    seen_ids, seen_families = set(), set()
    manifests = list(Path('.cache/v55').rglob('split-manifest.csv'))
    if len(manifests) != 1:
        raise ValueError('Expected original v55 split manifest')
    with manifests[0].open() as f:
        for r in csv.DictReader(f):
            if r['split'] == 'test' and r['label'] == '0':
                seen_ids.add(r['identity']); seen_families.add(r['family'])
    for path in (data/'test.jsonl.gz', Path('.cache/v59-data/test.jsonl.gz')):
        for r in rows(path):
            if r['y'] == 0:
                seen_ids.add(r['identity']); seen_families.add(r['family'])
    source_counts = defaultdict(Counter)
    def retain(parsed, label, source, origin, alias=None):
        family = campaign_key(parsed['ngram_text'])
        variants = [parsed] + ([alias] if alias else [])
        keys = [k for p in variants for k in content_keys(p)[0]]
        families = ['family:'+campaign_key(p['ngram_text']) for p in variants]
        pool.add(keys, families, label, parsed['identity'], family, parsed['ngram_text'], source, origin)
        source_counts[source]['ham' if label == 0 else 'spam'] += 1
    # Freeze a full immutable HF revision before reading labels or obtaining scores.
    revision = HfApi().dataset_info(SEVEN).sha
    seven_splits = load_dataset(SEVEN, revision=revision)
    fingerprints = {}
    for split,seven in sorted(seven_splits.items()):
        if set(('subject','text','sender','label','dataset_name')) - set(seven.column_names):
            raise ValueError('Seven dataset schema changed')
        fingerprints[split] = dict(count=len(seven),fingerprint=seven._fingerprint)
        for i,r in enumerate(seven):
            parsed = canonical_fields(r.get('subject') or '', r.get('text') or '', r.get('sender') or 'unknown@example.invalid')
            retain(parsed, r['label'], r['dataset_name'], f'{revision}:{split}:{i}')
            if (i+1)%25000 == 0:
                pool.db.commit(); print('SEVEN AUDIT', split, i+1, '/', len(seven), dict(pool.stats), flush=True)
    downloads = out/'downloads'
    downloads.mkdir()
    archive = downloads/'trec06c.zip'
    with urlopen(Request(CHINESE, headers={'User-Agent':'MailGuard-public-research/1.0'}), timeout=180) as src, archive.open('wb') as dst:
        import shutil
        shutil.copyfileobj(src, dst)
    with zipfile.ZipFile(archive) as z:
        indexes = [n for n in z.namelist() if n.endswith('/full/index')]
        if len(indexes) != 1:
            raise ValueError('Expected one original Chinese label index')
        name = indexes[0]
        index = z.read(name)
        labels = [line.split() for line in index.decode('ascii').splitlines() if line.strip()]
        if Counter(r[0] for r in labels) != {'ham':21766, 'spam':42854}:
            raise ValueError('Original Chinese gold label counts changed')
        prefix = name[:-len('full/index')]
        for i,(label,path) in enumerate(labels):
            path = PurePosixPath(path)
            if path.is_absolute() or path.parts[:2] != ('..','data') or '..' in path.parts[1:]:
                raise ValueError('Unsafe corpus index path')
            raw = z.read(prefix+str(PurePosixPath(*path.parts[1:])))
            parsed, legacy = from_eml(raw), legacy_parse(raw)
            # A severely garbled legacy alias must not falsely merge unrelated Chinese mail.
            if legacy['ngram_text'].count('\ufffd')/max(1,len(legacy['ngram_text'])) >= .005:
                legacy = None
            retain(parsed, int(label == 'spam'), 'TREC06-Chinese-original', str(path), legacy)
            if (i+1)%10000 == 0:
                pool.db.commit(); print('CHINESE AUDIT',i+1,'/',len(labels),flush=True)
    pool.db.commit()
    n = pool.db.execute('SELECT COUNT(*) FROM samples WHERE valid=1').fetchone()[0]
    if n < 20000:
        raise ValueError(f'Only {n} eligible unique gold HAM; no duplicate refill')
    count, sources, scripts, previously_scored = 0, Counter(), Counter(), Counter()
    from v61_semantic import script
    corpus = out/'ham.jsonl.gz'
    with gzip.open(corpus,'wt',encoding='utf-8',compresslevel=3) as f:
        for identity,family,text,source,origin in pool.db.execute('SELECT identity,family,text,source,origin FROM samples WHERE valid=1 ORDER BY identity'):
            focus, context = views(text)
            reused = identity in seen_ids or family in seen_families
            r = dict(identity=identity,family=family,source=source,origin=origin,y=0,focus=focus,context=context,previouslyScored=reused)
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
            count += 1; sources[source] += 1; scripts[script(r)] += 1; previously_scored['reused' if reused else 'additional'] += 1
    if count != n or file_sha(model) != MODEL_SHA:
        raise ValueError('Corpus count or frozen model changed')
    audit = dict(count=n,ham=n,spam=0,modelSha256=MODEL_SHA,hamSha256=file_sha(corpus),
                 bySource=dict(sources),byScript=dict(scripts),previouslyScored=dict(previously_scored),
                 developmentSplitCounts=split_counts,developmentManifestSha256=provenance['manifestSha256'],
                 exclusions=dict(pool.stats),sourceLabelCounts={k:dict(v) for k,v in source_counts.items()},
                 sources=[dict(dataset=SEVEN,revision=revision,splits=fingerprints),
                          dict(url=CHINESE,archiveSha256=file_sha(archive),labelIndexSha256=hashlib.sha256(index).hexdigest())],
                 codeCommit=os.environ.get('GITHUB_SHA'),githubRunId=os.environ.get('GITHUB_RUN_ID'),
                 thresholdsChanged=False,modelRetrained=False,customerGoalValidated=False,caveat=CAVEAT)
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    pool.db.close()
    print('HAM CORPUS FROZEN',n,dict(previously_scored),dict(sources),flush=True)


def score(args):
    import numpy as np
    from itertools import islice
    from v62_runtime import load_bundle,score_rows,predict_texts
    from train_v55_multilingual import views
    from v61_semantic import script
    out = Path(args.output)
    audit = json.loads((out/'audit.json').read_text())
    model, corpus = model_file(args.model), out/'ham.jsonl.gz'
    if file_sha(model) != MODEL_SHA or file_sha(corpus) != audit['hamSha256']:
        raise ValueError('Frozen model or corpus SHA mismatch')
    bundle = load_bundle(model.parent)
    profiles = ('conservative','high-recall')
    locked = json.dumps(bundle['profiles'],sort_keys=True)
    tally, count, unique = defaultdict(Counter), 0, set()
    started = time.monotonic()
    iterator = rows(corpus)
    with gzip.open(out/'predictions.csv.gz','wt',newline='') as f, (out/'false-positives.csv').open('w',newline='') as ef:
        writer, errors = csv.writer(f), csv.writer(ef)
        header = ['identity','family','source','script','previouslyScored','conservativeMargin','highRecallMargin']
        writer.writerow(header); errors.writerow(header)
        while batch := list(islice(iterator,128)):
            scores = {p:score_rows(bundle,batch,p) for p in profiles}
            if not count:
                for p in profiles:
                    if not np.allclose(scores[p],score_rows(bundle,batch,p,batch_size=7),rtol=1e-6,atol=1e-7):
                        raise ValueError('Runtime chunking parity failed')
                    texts = ['SUBJECT: Meeting\nBODY: Please confirm tomorrow.','SUBJECT: 会议\nBODY: 请确认会议时间谢谢']
                    direct = [dict(zip(('focus','context'),views(t))) for t in texts]
                    if not np.allclose(predict_texts(bundle,texts,p),score_rows(bundle,direct,p,batch_size=1),rtol=1e-6,atol=1e-7):
                        raise ValueError('Saved deployment parity failed')
            for i,r in enumerate(batch):
                if r['y'] != 0 or r['identity'] in unique:
                    raise ValueError('Not a unique gold HAM row')
                unique.add(r['identity'])
                language = script(r)
                group = 'previously-scored' if r['previouslyScored'] else 'additional'
                margins = [float(scores[p][i]) for p in profiles]
                if not all(math.isfinite(v) for v in margins):
                    raise ValueError('Nonfinite score')
                record = [r['identity'],r['family'],r['source'],language,r['previouslyScored'],*margins]
                writer.writerow(record)
                if any(v >= 0 for v in margins):
                    errors.writerow(record)
                for p,margin in zip(profiles,margins):
                    for name in ('all',group,'source/'+r['source'],'script/'+language):
                        tally[name,p]['fp' if margin >= 0 else 'tn'] += 1
            count += len(batch)
            if count%4096 == 0:
                print('HAM SCORED',count,'/',audit['count'],flush=True)
    if count != audit['count'] or file_sha(model) != MODEL_SHA or json.dumps(bundle['profiles'],sort_keys=True) != locked:
        raise ValueError('Incomplete score or mutated model/profile')
    result = {}
    for (name,p),c in sorted(tally.items()):
        result.setdefault(name,{})[p] = measure(c)
    report = dict(evaluation='v62-large-gold-ham',count=count,ham=count,spam=0,modelSha256=MODEL_SHA,
                  profiles=bundle['profiles'],results=result,audit=audit,seconds=time.monotonic()-started,
                  runtimeParityPassed=True,thresholdsChanged=False,modelRetrained=False,
                  customerGoalValidated=False,spamRecallValidated=False,caveat=CAVEAT)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    md = '# MailGuard v62: large gold HAM test\n\n'+CAVEAT+'\n\n| Group | Profile | HAM | FP | FPR | 95% FPR upper bound |\n|---|---|---:|---:|---:|---:|\n'
    for name,values in result.items():
        for p,m in values.items():
            md += f'| {name} | {p} | {m["ham"]} | {m["fp"]} | {m["hamFpr"]:.5%} | {m["hamFprWilson95Upper"]:.5%} |\n'
    (out/'report.md').write_text(md)
    print(md,flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase',choices=['prepare','score'],required=True)
    p.add_argument('--model',default='.cache/v62')
    p.add_argument('--output',default='reports/v62-large-ham')
    args = p.parse_args()
    (prepare if args.phase == 'prepare' else score)(args)


if __name__ == '__main__':
    main()
