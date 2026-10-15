#!/usr/bin/env python3
"""v66 candidate source audit only: NO model training, scoring or TEST leakage."""
import argparse,csv,gzip,hashlib,io,json,sqlite3,tarfile
from pathlib import Path
from collections import Counter
from urllib.request import Request,urlopen

CW='75b488148d5560109d6fb0c6646d417d7271f66a'
TURK='082ace5'
def digest(data):return hashlib.sha256(data).hexdigest()
def file_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def fetch(url,maxsize):
    data=bytearray()
    with urlopen(Request(url,headers={'User-Agent':'MailGuard-v66-public-research/1.0'}),timeout=70) as r:
        while piece:=r.read(1024*1024):
            data.extend(piece)
            if len(data)>maxsize:raise ValueError('Unexpectedly large public corpus')
    return bytes(data)

def row(parsed,label,source,raw=None):
    from train_v55_multilingual import campaign_key,views
    from prepare_v64_transfer import fast_content_keys,view_keys
    keys,body=fast_content_keys(parsed)
    if len(body)<20 or not any(c.isalpha() for c in body):return None
    focus,context=views(parsed['ngram_text'])
    family=campaign_key(parsed['ngram_text'])
    keys=set([*keys,'family:'+family,*view_keys(dict(focus=focus,context=context))])
    if raw is not None:keys.add('raw:'+digest(raw))
    return dict(source=source,label=int(label),identity=parsed['identity'],family=family,aliases=sorted(keys))

def turkish():
    from train_v47_phishing_aware import canonical_fields
    rec=[];files=[];original=Counter()
    for name in ('train','eval'):
        url=f'https://huggingface.co/datasets/anilguven/turkish_spam_email/resolve/{TURK}/spam_{name}.csv'
        raw=fetch(url,2_000_000)
        files.append(dict(url=url,sha256=digest(raw),size=len(raw)))
        reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
        if not {'text','labels'}.issubset(reader.fieldnames or []):raise ValueError('Source schema changed')
        for line in reader:
            if line['labels'] not in ('0','1'):raise ValueError('Invalid source label')
            original[line['labels']]+=1
            text=line['text'].strip()
            if len(text)>250_000:raise ValueError('Oversized email text')
            value=row(canonical_fields('',text),int(line['labels']),'Turkish-stemmed')
            if value is not None:rec.append(value)
    return rec,dict(source='Turkish-stemmed',revision=TURK,files=files,publisherLabels=dict(original),
                    labelQuality='publisher supplied ham/spam, independent adjudication not verified',
                    limitations=['Turkish text already cleaned and stemmed','MIME headers absent','license unknown'])

def phishing():
    from v59_mail import from_eml
    url=f'https://codeload.github.com/cw-l/email-corpus/tar.gz/{CW}'
    raw=fetch(url,100_000_000);records=[];count=0
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        members=[m for m in archive if m.isfile() and m.name.endswith('.eml')]
        if len(members)>2500 or not members:raise ValueError('Unexpected phishing source inventory')
        for m in members:
            if '/submissions/' not in m.name or not m.name.startswith('email-corpus-') or m.size>2_000_000:
                raise ValueError('Unsafe mail archive entry')
            raw_eml=archive.extractfile(m).read();count+=1
            value=row(from_eml(raw_eml),1,'CW-L-phishing',raw_eml)
            if value is not None:records.append(value)
    return records,dict(source='CW-L-phishing',url=url,revision=CW,archiveSha256=digest(raw),
                        bytes=len(raw),rawMessages=count,
                        labelQuality='community contributed phishing/spam; labels not independently validated',
                        limitations=['No HAM source','PII redacted','URLs defanged','original MIME structure partially stripped'])

def dedup(records):
    owners={};reject=set();counts=Counter()
    for i,r in enumerate(records):
        for key in r['aliases']:
            if key in owners:
                reject.add(i);reject.add(owners[key])
                counts['conflictingLabel' if r['label']!=records[owners[key]]['label'] else 'duplicate']+=1
            else:owners[key]=i
    return [r for i,r in enumerate(records) if i not in reject],dict(counts)

def check_prior(candidates,dbfile,oldroot):
    if not dbfile.is_file():raise ValueError('Missing v64 exclusion database')
    db=sqlite3.connect('file:'+str(dbfile.resolve())+'?mode=ro',uri=True)
    blocked=set()
    try:
        allkeys=set(k for r in candidates for k in r['aliases'])
        for key in allkeys:
            if db.execute('SELECT 1 FROM keys WHERE key=?',(key,)).fetchone():blocked.add(key)
    finally:db.close()
    testkeys=set()
    from prepare_v64_transfer import view_keys
    # All 500000 previously evaluated test records, never just the TRAIN blocks.
    for shard in range(5):
        files=list(oldroot.rglob(f'input-{shard}.jsonl.gz'))
        if len(files)!=1:raise ValueError(f'Missing old test shard {shard}')
        count=0
        with gzip.open(files[0],'rt',encoding='utf-8') as f:
            for line in f:
                v=json.loads(line);count+=1
                for k in ('identity:'+v['identity'],'family:'+v['family'],*view_keys(v)):
                    if k in allkeys:testkeys.add(k)
        if count!=100000:raise ValueError('Incomplete old test shard')
        print('Checked prior 100000',shard,flush=True)
    survived=[];exclude=Counter()
    for r in candidates:
        if blocked.intersection(r['aliases']):exclude['priorDevelopOrEarlierBenchmarks']+=1
        elif testkeys.intersection(r['aliases']):exclude['v64FirstUseTest']+=1
        else:survived.append(r)
    return survived,dict(exclude),dict(blockedDatabaseSha256=file_hash(dbfile),
                                        priorFingerprintHits=len(blocked),
                                        priorTestFingerprintHits=len(testkeys),
                                        priorTestRowsChecked=500000)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prior',required=True);ap.add_argument('--old-input',required=True)
    ap.add_argument('--output',required=True);args=ap.parse_args()
    from prepare_v64_transfer import enable_equivalent_normalizers
    enable_equivalent_normalizers()
    candidates=[];source_audits=[]
    for loader in (turkish,phishing):
        entries,meta=loader();candidates.extend(entries);source_audits.append(meta)
        print('CANDIDATE SOURCE',meta['source'],len(entries),flush=True)
    unique,collisions=dedup(candidates)
    eligible,excluded,evidence=check_prior(unique,Path(args.prior),Path(args.old_input))
    counts=Counter(f"{r['source']}:{r['label']}" for r in eligible)
    classes=Counter(r['label'] for r in eligible)
    if classes[0]<1 or classes[1]<1:raise ValueError('Both classes absent after exclusions')
    result=dict(stage='v66 source inventory, not an independent score',sources=source_audits,
        totalParsed=len(candidates),uniqueBeforePriorExclusions=len(unique),ambiguousAliases=collisions,
        excludedPreviouslySeen=excluded,priorEvidence=evidence,retained=len(eligible),
        retainedBySourceClass=dict(counts),retainedHAM=classes[0],retainedSPAM=classes[1],
        noTrainingOrInference=True,noModelScoresRead=True,noRawMessagesSaved=True,
        independentEvaluationPerformed=False,
        restrictions='Not a gold production benchmark: publisher labels and Turkish preprocessing/unknown license limit interpretation. CW-L phishing-only. Semantic near-duplicates are not comprehensively excluded. No customer mail used. A trustworthy rare-FP claim requires >=30000 independent, validated HAM at zero FP.')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    md='# MailGuard v66 candidate source inventory (no predictions)\n\n'
    md+='Source revisions, download SHA256 and old TEST exclusion checks have been frozen.\n\n'
    md+='| Source | Eligible HAM | Eligible SPAM |\n|---|---:|---:|\n'
    for s in ('Turkish-stemmed','CW-L-phishing'):
        md+=f"| {s} | {counts[s+':0']} | {counts[s+':1']} |\n"
    md+=f"\nWithin-source collisions: {collisions}. Prior overlap exclusions: {excluded}.\n\n"
    md+='**Not independent performance validation.** '+result['restrictions']+'\n\n'
    md+='The audit publishes counts and hashes only, no messages, train data or predictions.\n'
    (out/'audit.md').write_text(md)
    print(md,flush=True)
if __name__=='__main__':main()
