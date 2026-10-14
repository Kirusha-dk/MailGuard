"""Prepare first-use transfer mail without importing a model or using its scores."""
import argparse
import csv
import gzip
import hashlib
import html
import json
import mailbox
import os
import re
import shutil
import sqlite3
import subprocess
import tarfile
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path,PurePosixPath

SEED='mailguard-v64-transfer-20261014:'
EXPECTED_GENS={'adapt_GEN.ems':300,'adapt_SPAM.ems':300,'test_GEN.ems':754,
               'test_SPAM.ems':797,'train_GEN.ems':8158,'train_SPAM.ems':30088}
SOURCES=('independent','2016','2015','2014','2013','2012','2011')
class EquivalentEmailNormalizer:
    frozen=re.compile(r'[\w.+-]+@[\w.-]+\.[\w-]+',re.U)
    bounded=re.compile(r'(?<![\w.+-])[\w.+-]+@[\w.-]+\.[\w-]+',re.U)

    def sub(self,replacement,text,count=0):
        # Adjacent addresses separated by '+' can begin immediately after a
        # preceding match. Preserve the original scanner in those strings.
        pattern=self.frozen if '+' in text and text.count('@')>1 else self.bounded
        return pattern.sub(replacement,text,count=count)

FAST_EMAIL=EquivalentEmailNormalizer()

def enable_equivalent_normalizers():
    # A valid match always starts at the beginning of a maximal local-part run.
    # This boundary eliminates quadratic retries in long text without changing
    # the leftmost matches produced by the frozen expression.
    import train_v55_multilingual as v55
    v55.EMAIL=FAST_EMAIL

def fast_content_keys(row):
    body=row['ngram_text'].split('\nBODY: ',1)[-1]
    body=unicodedata.normalize('NFKC',body).casefold()
    body=re.sub('[\u200b-\u200f\ufeff]','',body)
    body=re.sub(r'\s+',' ',body).strip()
    template=re.sub(r'https?://\S+|www\.\S+','<url>',body)
    template=FAST_EMAIL.sub('<email>',template)
    template=re.sub(r'\d+','<number>',template)
    return ('identity:'+row['identity'],'body:'+sha_text(body),'template:'+sha_text(template)),body

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def sha_text(text):return hashlib.sha256(text.encode('utf-8','replace')).hexdigest()

def read_rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)

def view_keys(row):
    common=os.path.commonprefix((row['focus'],row['context']))
    rest=row['context'][len(common):]
    if not common.startswith('subject ') or not rest.startswith('body '):
        raise ValueError('Expected canonical prepared views')
    # Ignore subject, envelope and hosts in this extra exclusion fingerprint.
    body=re.sub(r'\s+',' ',rest[5:]).strip()
    return ['viewbody:'+sha_text(body)]

def genspam_messages(text):
    for match in re.finditer(r'<MESSAGE\b[^>]*>(.*?)</MESSAGE>',text,re.S):
        message=match.group(1)
        def field(tag):
            block=re.search(r'<'+tag+r'\b[^>]*>(.*?)</'+tag+r'>',message,re.S)
            if not block:return ''
            fragments=re.findall(r'<TEXT_NORMAL\b[^>]*>(.*?)</TEXT_NORMAL>',block.group(1),re.S)
            return '\n'.join(html.unescape(s) for s in fragments)
        yield field('SUBJECT'),field('MESSAGE_BODY')

class UniquePool:
    """A conflicting label or a bridge invalidates every previously retained owner."""
    def __init__(self,path,blocked=()):
        self.db=sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=NORMAL')
        self.db.execute('CREATE TABLE samples(id INTEGER PRIMARY KEY, identity TEXT, family TEXT, y INTEGER, source TEXT, script TEXT, origin TEXT, focus TEXT, context TEXT, aliases TEXT, rank TEXT, valid INTEGER DEFAULT 1)')
        self.db.execute('CREATE TABLE keys(key TEXT PRIMARY KEY, sample INTEGER, y INTEGER) WITHOUT ROWID')
        self.blocked=set(blocked);self.stats=Counter()

    def add(self,row,aliases):
        if type(row['y']) is not int or row['y'] not in (0,1):raise ValueError('Binary source label required')
        aliases=sorted(set(aliases));y=row['y'];self.stats['read']+=1
        prior=self.db.execute('SELECT sample,y FROM keys WHERE key IN ('+','.join('?' for _ in aliases)+')',aliases).fetchall()
        owners={p[0] for p in prior if p[0] is not None}
        blocked=bool(self.blocked.intersection(aliases)) or any(p[0] is None for p in prior)
        conflict=any(p[1] is not None and p[1]!=y for p in prior)
        if blocked or prior:
            if blocked or conflict or len(owners)>1:
                self.db.executemany('UPDATE samples SET valid=0 WHERE id=?',[(i,) for i in owners])
            owner=None if blocked else min(owners) if owners else None
            stored=None if blocked else prior[0][1]
            self.db.executemany('INSERT OR IGNORE INTO keys VALUES (?,?,?)',[(k,owner,stored) for k in aliases])
            if blocked:self.blocked.update(aliases)
            self.stats['priorOverlap' if blocked else 'conflictOrBridge' if conflict or len(owners)>1 else 'duplicate']+=1
            return
        cur=self.db.execute('INSERT INTO samples(identity,family,y,source,script,origin,focus,context,aliases,rank) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (row['identity'],row['family'],y,row['source'],row['script'],row['origin'],row['focus'],row['context'],json.dumps(aliases),sha_text(SEED+row['identity'])))
        self.db.executemany('INSERT INTO keys VALUES (?,?,?)',[(k,cur.lastrowid,y) for k in aliases])
        self.stats['retained']+=1

    def count(self):return self.db.execute('SELECT COUNT(*) FROM samples WHERE valid=1').fetchone()[0]

    def records(self,where='valid=1',params=()):
        for values in self.db.execute('SELECT identity,family,y,source,script,origin,focus,context,aliases FROM samples WHERE '+where+' ORDER BY rank',params):
            r=dict(zip(('identity','family','y','source','script','origin','focus','context','aliases'),values));r['aliases']=json.loads(r['aliases']);yield r

    def close(self):self.db.commit();self.db.close()

def prepare_blocks(args):
    from train_v47_phishing_aware import canonical_fields
    from train_v55_multilingual import campaign_key,views
    enable_equivalent_normalizers()
    content_keys=fast_content_keys
    from datasets import load_dataset
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    old=Path('.cache/v64-prior/blocked.sqlite')
    if not old.exists():raise ValueError('Missing prior whole-source exclusion checkpoint')
    prior=sqlite3.connect(old.as_uri()+'?mode=ro',uri=True) if old.is_absolute() else sqlite3.connect('file:'+str(old)+'?mode=ro',uri=True)
    db=sqlite3.connect(out/'blocked.sqlite');db.execute('CREATE TABLE keys(key TEXT PRIMARY KEY) WITHOUT ROWID')
    cursor=prior.execute('SELECT key FROM keys');n=0
    while batch:=cursor.fetchmany(10000):db.executemany('INSERT OR IGNORE INTO keys VALUES (?)',batch);n+=len(batch)
    prior.close()
    if n<1000000:raise ValueError('Prior whole-source exclusion checkpoint unexpectedly small')
    evidence=dict(priorCache='mailguard-v59-prior-blocking-v1',priorDatabaseSha256=sha_file(old),priorKeys=n,
                  old500kCorpusSha256='5c0993c0258b854ce9cf2d5a31172d56692aaa351a533ca57c1ecc6e97c81f82',splits={},previousPrepared={},classifierLoaded=False)
    def block(r):
        keys=['identity:'+r['identity'],'family:'+r['family'],*view_keys(r)]
        if r.get('rawSha256'):keys.append('raw:'+r['rawSha256'])
        for field in ('text','old_text'):
            if r.get(field):
                p=dict(identity=r['identity'],ngram_text=r[field]);keys.extend(content_keys(p)[0]);keys.append('family:'+campaign_key(r[field]))
        db.executemany('INSERT OR IGNORE INTO keys VALUES (?)',[(k,) for k in set(keys)])
    for directory in ('.cache/v60-data','.cache/v59-data','.cache/v61-new'):
        root=Path(directory);p=json.loads((root/'provenance.json').read_text());checks=p.get('splitHashes',{'test':p.get('testSha256')})
        for split,digest in checks.items():
            path=root/(split+'.jsonl.gz')
            if sha_file(path)!=digest:raise ValueError('Prior split changed')
            count=0
            for r in read_rows(path):block(r);count+=1
            evidence['splits'][directory+'/'+split]=dict(count=count,sha256=digest)
            db.commit();print('BLOCKED SPLIT',directory,split,count,flush=True)
    audit500=json.loads(Path('reports/v62-500k-result.json').read_text())['audit']
    for shard,digest in audit500['shardSha256'].items():
        paths=list(Path('previous/spam').rglob('input-'+shard+'.jsonl.gz'))
        if len(paths)!=1 or sha_file(paths[0])!=digest:raise ValueError('Prior 500k input changed')
        count=0
        for r in read_rows(paths[0]):block(r);count+=1
        if count!=100000:raise ValueError('Incomplete prior shard')
        evidence['previousPrepared']['500k/'+shard]=dict(count=count,sha256=digest);db.commit()
    ham=Path('previous/ham');ha=json.loads((ham/'audit.json').read_text());path=ham/'ham.jsonl.gz'
    if sha_file(path)!=ha['hamSha256']:raise ValueError('Prior HAM input changed')
    count=0
    for r in read_rows(path):block(r);count+=1
    if count!=75271:raise ValueError('Incomplete prior HAM')
    evidence['previousPrepared']['ham']=dict(count=count,sha256=ha['hamSha256'])
    revision='deb4cc27dc2754c14cb157aace412727f77b41dd'
    counts={}
    for split,ds in load_dataset('JinqiangDing/seven-phishing-email-datasets',revision=revision).items():
        counts[split]=len(ds)
        for item in ds:
            p=canonical_fields(item.get('subject') or '',item.get('text') or '',item.get('sender') or 'unknown@example.invalid')
            f,c=views(p['ngram_text']);block(dict(p,family=campaign_key(p['ngram_text']),focus=f,context=c,text=p['ngram_text']))
    evidence['wholeSeven']=dict(revision=revision,counts=counts)
    db.commit();evidence['blockedKeys']=db.execute('SELECT COUNT(*) FROM keys').fetchone()[0];db.close()
    evidence['databaseSha256']=sha_file(out/'blocked.sqlite')
    (out/'audit.json').write_text(json.dumps(evidence,indent=2));print('PRIOR EXCLUSIONS FROZEN',json.dumps(evidence),flush=True)

def checked_blocks(root):
    root=Path(root);audit=json.loads((root/'audit.json').read_text())
    if sha_file(root/'blocked.sqlite')!=audit['databaseSha256']:raise ValueError('Exclusion checkpoint changed')
    db=sqlite3.connect('file:'+str(root/'blocked.sqlite')+'?mode=ro',uri=True)
    result={r[0] for r in db.execute('SELECT key FROM keys')};db.close();return result,audit

def collect(args):
    from train_v47_phishing_aware import canonical_fields,from_eml as legacy_parse
    from train_v55_multilingual import campaign_key,views
    from v59_mail import from_eml
    from v61_semantic import script
    from evaluate_v54_unique500k import download,has_mail_headers
    enable_equivalent_normalizers()
    content_keys=fast_content_keys
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    blocked,audit=checked_blocks(args.blocks);pool=UniquePool(out/'pool.sqlite',blocked);del blocked
    inventories=[]
    def retain(parsed,y,source,origin,raw=None):
        keys,body=content_keys(parsed)
        if len(body)<20 or not any(c.isalpha() for c in body):pool.stats['shortOrEmpty']+=1;return
        f,c=views(parsed['ngram_text']);family=campaign_key(parsed['ngram_text'])
        row=dict(identity=parsed['identity'],family=family,y=y,source=source,origin=origin,focus=f,context=c)
        row['script']=script(row);aliases=[*keys,'family:'+family,*view_keys(row)]
        if raw:
            aliases.append('raw:'+hashlib.sha256(raw).hexdigest());old=legacy_parse(raw)
            if old['ngram_text'].count('\ufffd')/max(1,len(old['ngram_text']))<.005:
                of,oc=views(old['ngram_text']);aliases.extend([*content_keys(old)[0],'family:'+campaign_key(old['ngram_text']),*view_keys(dict(focus=of,context=oc))])
        pool.add(row,aliases)
        if pool.stats['read']%10000==0:
            pool.db.commit();print('NEW SOURCE PROGRESS',args.source,pool.stats['read'],'unique',pool.count(),dict(pool.stats),flush=True)
    if args.source=='independent':
        root=Path(args.sources);inventory=json.loads((root/'source-inventory.json').read_text())
        sources={s['file']:s for s in inventory['sources'] if s['available']}
        for name,s in sources.items():
            if sha_file(root/name)!=s['sha256']:raise ValueError('Source snapshot changed')
        if 'genspam.tar.gz' not in sources:raise ValueError('New independent HAM source unavailable')
        with tarfile.open(root/'genspam.tar.gz','r:gz') as t:
            entries={Path(m.name).name:m for m in t if m.isfile()}
            if set(EXPECTED_GENS)-set(entries):raise ValueError('Missing GenSpam files')
            counts={}
            for name,n in EXPECTED_GENS.items():
                text=t.extractfile(entries[name]).read().decode('utf-8','replace');count=0
                for subject,body in genspam_messages(text):
                    retain(canonical_fields(subject,body),int('_SPAM.' in name),'GenSpam',name+':'+str(count));count+=1
                if count!=n:raise ValueError('Original GenSpam inventory changed: '+name+': '+str(count))
                counts[name]=count;print('GENSPAM INVENTORY',name,count,flush=True)
            inventories.append(dict(sources['genspam.tar.gz'],messageCounts=counts,
                publishedTrainSpamCount=30099,archiveTrainSpamCount=30088,inventoryDifferenceDocumented=True,
                labels='Original genuine/spam annotation',anonymized=True,attribution='Ben Medlock, CEAS 2006'))
        with zipfile.ZipFile(root/'labelled-mail.zip') as z:
            labels=Counter()
            for name in z.namelist():
                label=0 if '/dataset/1/' in name else 1 if '/dataset/2/' in name else None
                if label is None or name.endswith('/'):continue
                raw=z.read(name);retain(from_eml(raw),label,'Published-ASCII-mail',name,raw);labels[label]+=1
            if labels!={0:10776,1:8752}:raise ValueError('Published ASCII mail inventory changed')
        inventories.append(sources['labelled-mail.zip'])
        box=mailbox.mbox(root/'nazario-2025.mbox',create=False);count=0
        for i,key in enumerate(box.iterkeys()):
            raw=box.get_bytes(key);retain(from_eml(raw),1,'Nazario-2025',str(i),raw);count+=1
        box.close()
        if count!=sources['nazario-2025.mbox']['mboxMessages']:raise ValueError('Nazario mbox inventory changed')
        inventories.append(sources['nazario-2025.mbox'])
    else:
        year=args.source;work=out/'download';work.mkdir();archive=work/(year+'.7z')
        s=download('https://untroubled.org/spam/'+year+'.7z',archive)
        listing=subprocess.run(['7z','l','-slt',str(archive)],check=True,capture_output=True,text=True).stdout
        for entry in listing.split('----------\n',1)[-1].split('\n\n'):
            if 'Symbolic Link =' in entry or 'Hard Link =' in entry:raise ValueError('Archive link rejected')
            for line in entry.splitlines():
                if line.startswith('Path = '):
                    p=PurePosixPath(line[7:].replace('\\','/'))
                    if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
        dest=work/'mail';subprocess.run(['7z','x','-y','-bd','-bso0','-bsp0','-o'+str(dest),str(archive)],check=True)
        count=0
        for p in sorted(dest.rglob('*')):
            if p.is_symlink():raise ValueError('Archive symlink rejected')
            if not p.is_file() or p.stat().st_size>8*1024*1024:continue
            raw=p.read_bytes()
            if has_mail_headers(raw):retain(from_eml(raw),1,'SpamArchive-'+year,str(p.relative_to(dest)),raw);count+=1
        s.update(rawEmailsRead=count,labels='Spam trap; not individually adjudicated');inventories.append(s);shutil.rmtree(work)
    pool.db.commit();count=pool.count();file=out/'candidates.jsonl.gz'
    with gzip.open(file,'wt',encoding='utf-8',compresslevel=3) as f:
        for r in pool.records():f.write(json.dumps(r,ensure_ascii=False)+'\n')
    result=dict(source=args.source,count=count,exclusions=dict(pool.stats),sources=inventories,
                candidatesSha256=sha_file(file),priorDatabaseSha256=audit['databaseSha256'],classifierLoaded=False,scoreBasedSelection=False)
    (out/'audit.json').write_text(json.dumps(result,indent=2));pool.close()
    for p in out.glob('pool.sqlite*'):p.unlink()
    print('NEW CANDIDATES FROZEN',json.dumps(result),flush=True)

def merge(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    blocked,prior=checked_blocks(args.blocks);pool=UniquePool(out/'pool.sqlite',blocked);del blocked
    audits=[];inputroot=Path(args.input)
    for source in SOURCES:
        roots=[p.parent for p in inputroot.rglob('audit.json') if json.loads(p.read_text()).get('source')==source]
        if len(roots)!=1:raise ValueError('Missing or duplicate source '+source)
        root=roots[0];audit=json.loads((root/'audit.json').read_text());path=root/'candidates.jsonl.gz'
        if audit['priorDatabaseSha256']!=prior['databaseSha256'] or sha_file(path)!=audit['candidatesSha256']:raise ValueError('Candidate or exclusions changed')
        count=0
        for r in read_rows(path):pool.add(r,r['aliases']);count+=1
        if count!=audit['count']:raise ValueError('Incomplete candidate source')
        pool.db.commit();audits.append(audit);print('MERGED SOURCE',source,'eligible',pool.count(),flush=True)
    available=pool.count()
    if available<args.count:raise ValueError(f'Only {available}/{args.count} new unique messages; no duplicate refill')
    pool.db.execute('CREATE TABLE selected(id INTEGER PRIMARY KEY)')
    pool.db.execute("INSERT INTO selected SELECT id FROM samples WHERE valid=1 AND source NOT LIKE 'SpamArchive-%'")
    independent=pool.db.execute('SELECT COUNT(*) FROM selected').fetchone()[0]
    if independent>=args.count:raise ValueError('Independent cohort unexpectedly exceeds total')
    pool.db.execute("INSERT INTO selected SELECT id FROM samples WHERE valid=1 AND source LIKE 'SpamArchive-%' ORDER BY rank LIMIT ?",(args.count-independent,))
    pool.db.commit()
    count=0;classes=Counter();sources=Counter();scripts=Counter();keys=set();counts=Counter()
    handles=[gzip.open(out/f'input-{i}.jsonl.gz','wt',encoding='utf-8',compresslevel=3) for i in range(5)]
    try:
        for r in pool.records('id IN (SELECT id FROM selected)'):
            if any(k in keys or k in pool.blocked for k in r['aliases']):raise ValueError('Final selected duplicate or prior overlap')
            keys.update(r['aliases']);r.pop('aliases');r['previouslyScored']=False
            handles[count%5].write(json.dumps(r,ensure_ascii=False)+'\n');counts[count%5]+=1;count+=1
            classes[r['y']]+=1;sources[r['source']]+=1;scripts[r['script']]+=1
    finally:
        for h in handles:h.close()
    if count!=args.count or not classes[0] or not classes[1]:raise ValueError('Incomplete selection or missing class')
    audit=dict(count=count,ham=classes[0],spam=classes[1],available=available,nonArchiveSourceRows=independent,
        bySource=dict(sources),byScript=dict(scripts),shardCounts=dict(counts),shardSha256={str(i):sha_file(out/f'input-{i}.jsonl.gz') for i in range(5)},
        priorExclusions=prior,candidateAudits=audits,globalExclusions=dict(pool.stats),
        selection='All eligible non-archive labelled-source mail; bulk archives fill remaining slots by fixed content hash; global hash order; no model scores used',
        zeroDetectedPriorOverlap=True,identityBodyTemplateFamilyViewDedupPassed=True,testPreviouslyScored=False,testLabelsUsedForTraining=False,
        targetRecall=.95,targetHamFpr=.0001,codeCommit=os.environ.get('GITHUB_SHA'),githubRunId=os.environ.get('GITHUB_RUN_ID'),
        caveat='First-use historical public mail, not customer traffic. GenSpam is anonymized; published ASCII mail has removed headers; Nazario is a single manually labelled inbox. Bulk spam traps are weak spam labels and come from a previously used provider in different years. Exact fingerprints and campaign prefixes exclude known relatives, but semantic near-duplicates may remain. HAM sample size may not establish extremely rare production FP. No synthetic messages or duplicated refill are used.')
    (out/'audit.json').write_text(json.dumps(audit,indent=2));pool.close()
    for p in out.glob('pool.sqlite*'):p.unlink()
    print('FIRST-USE TEST LOCKED BEFORE SCORING',json.dumps({k:v for k,v in audit.items() if k not in ('priorExclusions','candidateAudits')}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('blocks','collect','merge'),required=True)
    p.add_argument('--source',choices=SOURCES);p.add_argument('--blocks',default='prior-blocks');p.add_argument('--sources',default='new-sources')
    p.add_argument('--input');p.add_argument('--count',type=int,default=500000);p.add_argument('--output',required=True)
    args=p.parse_args();{'blocks':prepare_blocks,'collect':collect,'merge':merge}[args.phase](args)

if __name__=='__main__':main()
