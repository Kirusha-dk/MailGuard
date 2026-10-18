"""Freeze 100k distinct English emails before fitting or scoring v79.

SQLite keeps ingestion and global deduplication bounded in memory. Prior
reported test families are development-only. Ten outcome-blind hash blocks
are fixed before any classifier scores are computed.
"""
import argparse
import gzip
import hashlib
import json
import re
import sqlite3
import io
import zipfile
from collections import Counter
from pathlib import Path
import fasttext
import v77_mail as core
import v78_data as previous

SEED = 'v79-english-100000-ten-fixed-blocks-2026-10-18'
LID_SHA = '8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83'
MERGED_REV = '23029ecd499f6d2c564f5ddec4ac69964becce9c'
PARQUETS = {'0000.parquet': '62e2e357d5c440470dbfb1a26a3fdc969170f6ef6fd9f15134aa788b26b8b8fb',
            '0001.parquet': 'cfea57a51e1be0d93b9ae4fa6d864224eb5085665078f62d9c9a8f0d15a16b58'}
PLACEHOLDERS = re.compile(r'\b(?:numbertoken|urltoken|emailtoken|nametoken|orgtoken)\b')


def normalize(text):
    text = previous.normalize(text)
    for old, new in [('escapenumber', 'numbertoken'), ('escapeemail', 'emailtoken'), ('escapeurl', 'urltoken')]:
        text = re.sub(r'\b' + old + r'\b', new, text)
    return core.SPACE.sub(' ', text).strip()


def english_prediction(model, text):
    view = PLACEHOLDERS.sub(' ', text)
    view = core.SPACE.sub(' ', view[:4000] + ' ' + view[-1000:]).strip()
    result = model.f.predict(view + '\n', 1, 0., 'strict')
    return bool(result and result[0][1] == '__label__en' and result[0][0] >= .8), (result[0] if result else (0., 'unknown'))


def fingerprint(rows):
    return core.digest('\n'.join(sorted(r['identity'] + ':' + r['family'] for r in rows)))


class Pool:
    def __init__(self, path, lid):
        if Path(path).exists():
            raise ValueError('Pool already exists; do not overwrite a frozen corpus')
        self.db = sqlite3.connect(path)
        self.db.executescript('''
          PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
          CREATE TABLE rows(identity TEXT PRIMARY KEY,family TEXT,text TEXT,y INTEGER,source TEXT,kind TEXT,rank TEXT,lid_p REAL);
          CREATE TABLE conflicts(identity TEXT PRIMARY KEY);
          CREATE TABLE prior(family TEXT PRIMARY KEY);
        ''')
        self.lid = lid; self.stats = Counter(); self.crosscheck = Counter()

    def add(self, raw, y, source, prior=False, kind=None):
        self.stats[source + ':raw'] += 1
        if type(y) is not int or y not in (0, 1):
            raise ValueError('Unknown label')
        if not isinstance(raw, str) or not 30 <= len(raw) <= 250000:
            self.stats[source + ':length_excluded'] += 1; return
        text = normalize(raw)
        if len(text) < 30 or core.language(text) != 'en':
            self.stats[source + ':script_or_short_excluded'] += 1; return
        identity, family = previous.identifiers(text)
        if prior:
            self.db.execute('INSERT OR IGNORE INTO prior VALUES(?)', (family,))
        if self.db.execute('SELECT 1 FROM conflicts WHERE identity=?', (identity,)).fetchone():
            self.stats[source + ':conflicting_identity_excluded'] += 1; return
        old = self.db.execute('SELECT y,source FROM rows WHERE identity=?', (identity,)).fetchone()
        if old:
            self.stats[source + ':duplicate'] += 1
            if source == 'Merged-public' and old[1] != source:
                self.crosscheck[f'original_{old[0]}_merged_{y}'] += 1
            if old[0] != y:
                self.db.execute('INSERT OR IGNORE INTO conflicts VALUES(?)', (identity,))
                self.db.execute('DELETE FROM rows WHERE identity=?', (identity,))
                self.stats[source + ':label_conflict'] += 1
            return
        is_english, prediction = english_prediction(self.lid, text)
        if not is_english:
            self.stats[source + ':language_excluded'] += 1; return
        rank = core.digest(SEED + ':test:' + identity)
        self.db.execute('INSERT INTO rows VALUES(?,?,?,?,?,?,?,?)',
            (identity, family, text, y, source, kind or ('ham' if y == 0 else 'spam-unspecified'), rank, prediction[0]))
        self.stats[source + ':accepted'] += 1

    def checkpoint(self):
        self.db.commit()
        print('Indexed', sum(v for k,v in self.stats.items() if k.endswith(':raw')),
              'unique', self.db.execute('SELECT count(*) FROM rows').fetchone()[0], flush=True)


def export(pool, out, lid_bytes):
    db = pool.db
    disagreements = sum(v for k,v in pool.crosscheck.items() if k in ('original_0_merged_1','original_1_merged_0'))
    compared = sum(pool.crosscheck.values())
    if compared < 1000 or disagreements / compared > .01:
        raise ValueError(f'Merged source label crosscheck failed: {disagreements}/{compared}; do not train')
    db.executescript('CREATE INDEX family_index ON rows(family); CREATE INDEX rank_index ON rows(rank);')
    chosen = db.execute('''SELECT identity,family FROM rows r
       WHERE r.source<>'Merged-public' AND NOT EXISTS(SELECT 1 FROM prior p WHERE p.family=r.family)
       ORDER BY rank LIMIT 100000''').fetchall()
    if len(chosen) != 100000:
        raise ValueError(f'Only {len(chosen)} eligible new test identities; never duplicate to reach 100k')
    db.execute('CREATE TABLE test_ids(identity TEXT PRIMARY KEY)')
    db.execute('CREATE TABLE test_families(family TEXT PRIMARY KEY)')
    db.executemany('INSERT INTO test_ids VALUES(?)', ((i,) for i,f in chosen))
    db.executemany('INSERT OR IGNORE INTO test_families VALUES(?)', ((f,) for i,f in chosen))
    parts = {'train': [], 'select': [], 'cal': []}
    for identity, family, text, y, source, kind in db.execute('''SELECT identity,family,text,y,source,kind FROM rows r
          WHERE NOT EXISTS(SELECT 1 FROM test_families t WHERE t.family=r.family)'''):
        bucket = int(core.digest(SEED + ':dev:' + family)[:12],16)%100
        name = 'train' if bucket < 70 else 'select' if bucket < 80 else 'cal'
        parts[name].append(dict(identity=identity,family=family,text=text,y=y,source=source,kind=kind))
    train = []
    for source in sorted({r['source'] for r in parts['train']}):
        group = [r for r in parts['train'] if r['source'] == source]
        group.sort(key=lambda r: core.digest(SEED + ':train:' + r['identity']))
        train.extend(group[:40000])
    parts['train'] = sorted(train,key=lambda r:core.digest(SEED + ':shuffle:' + r['identity']))
    test = []
    for identity,family,text,y,source,kind in db.execute('''SELECT r.identity,r.family,r.text,r.y,r.source,r.kind FROM rows r
            JOIN test_ids t ON r.identity=t.identity ORDER BY r.rank'''):
        test.append(dict(identity=identity,family=family,text=text,y=y,source=source,kind=kind))
    groups = [{r['family'] for r in parts[k]} for k in ('train','select','cal')]
    test_families = {r['family'] for r in test}
    if any(a & test_families for a in groups) or any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Group leakage in final partitions')
    blocks = []
    for start in range(0,100000,10000):
        chunk=test[start:start+10000]
        blocks.append({'block':start//10000+1,'n':len(chunk),'fingerprint':fingerprint(chunk),
            'sources':dict(Counter(r['source'] for r in chunk)),
            'spam':sum(r['y'] for r in chunk),'ham':sum(1-r['y'] for r in chunk)})
    manifest = {'version':79,'seed':SEED,'test_count':100000,'blocks':blocks,
        'block_rule':'sort by SHA256(seed:test:canonical identity); consecutive 10000; no outcomes/labels in ordering',
        'test_fingerprint':fingerprint(test),'development_counts':{k:len(v) for k,v in parts.items()},
        'development_fingerprints':{k:fingerprint(v) for k,v in parts.items()},
        'development_source_class_counts':{k:dict(Counter(f"{r['source']}:{r['y']}" for r in v)) for k,v in parts.items()},
        'ingestion':dict(pool.stats),'merged_original_label_crosscheck':dict(pool.crosscheck),
        'merged_revision':MERGED_REV,'parquet_sha256':PARQUETS,'language_model_sha256':LID_SHA,
        'language_filter':'fastText lid.176.ftz: top language en, probability >=0.8, label-blind',
        'prior_reported_official_test_and_v78_test_families_development_only':True,
        'train_select_cal_test_prefix_groups_disjoint':True,'original_source_disjoint':False,
        'historical_public_data_not_customer_mail':True,'semantic_near_duplicates_not_fully_excluded':True,
        'merged_original_provenance_incomplete':True,
        'merged_unknown_origin_messages_development_only':True,
        'labelled_email_archive_sha256':'a51c47191cb32e0f62e56e4e265a20be2393d326a003001e0830601f6f444eb7',
        'note':'100k emails withheld from fresh v79 fitting, not a promise for every arbitrary 10k mail sequence.'}
    out.mkdir(parents=True,exist_ok=False)
    (out/'frozen.json').write_text(json.dumps(manifest,indent=2))
    for name, rows in {**parts,'test':test}.items():
        with gzip.open(out/(name+'.jsonl.gz'),'wt') as stream:
            for r in rows:stream.write(json.dumps(r)+'\n')
    (out/'lid.176.ftz').write_bytes(lid_bytes)
    print(json.dumps({'counts':manifest['development_counts'],'test':100000,
                     'blocks':blocks,'crosscheck':dict(pool.crosscheck)},indent=2),flush=True)


def main():
    import pyarrow.parquet as parquet
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',default='.cache/v79/raw')
    parser.add_argument('--archive',required=True)
    parser.add_argument('--out',default='.cache/v79/frozen')
    parser.add_argument('--pool',default='.cache/v79/pool.sqlite')
    args=parser.parse_args();raw=Path(args.raw)
    lid_bytes=(raw/'lid.176.ftz').read_bytes()
    if hashlib.sha256(lid_bytes).hexdigest()!=LID_SHA:raise ValueError('Language model drift')
    lid=fasttext.load_model(str(raw/'lid.176.ftz'));pool=Pool(args.pool,lid)
    with gzip.open('.cache/v78/data.json.gz','rt') as stream:old=json.load(stream)
    old_test={previous.identifiers(normalize(r['text']))[0] for r in old['test']};del old
    with gzip.open('.cache/v77/rows.json.gz','rt') as stream:
        for index,r in enumerate(previous.iter_array(stream),1):
            if r['lang']=='en':pool.add(r['text'],r['y'],r['source'],prior=r['official_split']=='test')
            if index%20000==0:pool.checkpoint()
    pool.checkpoint()
    gen,gen_audit=previous.genspam_rows(previous.source_blob('genspam.tar.gz',args.archive))
    for r in gen:pool.add(r['text'],r['y'],'GenSpam',prior=previous.identifiers(normalize(r['text']))[0] in old_test)
    del gen
    modern,_=previous.nazario_rows(previous.source_blob('nazario-2025.mbox',args.archive))
    for r in modern:pool.add(r['text'],r['y'],'Nazario-2025',kind='phish')
    with zipfile.ZipFile(args.archive) as outer:
        email_archive=outer.read('labelled-mail.zip')
    if hashlib.sha256(email_archive).hexdigest()!='a51c47191cb32e0f62e56e4e265a20be2393d326a003001e0830601f6f444eb7':
        raise ValueError('Labelled email archive drift')
    with zipfile.ZipFile(io.BytesIO(email_archive)) as archive:
        if len(archive.infolist())>25000:raise ValueError('Unexpected archive member count')
        for member in archive.infolist():
            if member.is_dir() or not member.filename.endswith('.eml') or member.file_size>250000:continue
            if '/dataset/1/' in member.filename:y=0
            elif '/dataset/2/' in member.filename:y=1
            else:continue
            pool.add(core.parse_eml(archive.read(member)),y,'Labelled-email-2010')
    del email_archive
    pool.checkpoint();scanned=0
    for name,expected in PARQUETS.items():
        if hashlib.file_digest((raw/name).open('rb'),'sha256').hexdigest()!=expected:raise ValueError('Parquet drift')
        source=parquet.ParquetFile(raw/name)
        for batch in source.iter_batches(batch_size=256,columns=['text','label']):
            for r in batch.to_pylist():
                label=r['label']
                if type(label) is not int or label not in (0,1,2):raise ValueError('Unexpected merged label')
                pool.add(r['text'],int(label!=0),'Merged-public',kind=('ham','phish','spam')[label])
                scanned+=1
            if scanned%25600==0:pool.checkpoint()
    if scanned!=365448:raise ValueError('Merged row count drift')
    pool.checkpoint();export(pool,Path(args.out),lid_bytes)


if __name__=='__main__':main()
