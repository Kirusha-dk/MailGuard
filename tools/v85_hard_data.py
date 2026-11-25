"""Pinned public native English email pilot; no scoring, gradients, or raw-mail publication."""
import argparse, collections, hashlib, io, json, re, zipfile
from pathlib import Path
import requests, fasttext
from v77_mail import parse_eml, clean
from v78_data import identifiers
from v79_data import normalize, english_prediction, LID_SHA
from v84_replay_data import read_rows, write_rows, fingerprint, windows, sha, EmailSubstitution
SOURCES=[
 dict(name='CSDMC2010-mirror-ham',filename='csdmc-ham.zip',url='https://raw.githubusercontent.com/zrz1996/Spam-Email-Classifier-DataSet/36b5435af66a037a5cd474874950ca38c837e953/ham.zip',sha256='ae4e7e46387fa98ba3e4dbb2b11618fe63e0d2fd10966ac85707413372ce98c8',label=0),
 dict(name='CSDMC2010-mirror-spam',filename='csdmc-spam.zip',url='https://raw.githubusercontent.com/zrz1996/Spam-Email-Classifier-DataSet/36b5435af66a037a5cd474874950ca38c837e953/spam.zip',sha256='9d2d5a4f6d85768f53cde4ae7eba365929c8818a43d5ec085cff25815f434247',label=1),
 dict(name='IWSPA-AP-2018-training',filename='iwspa.zip',url='https://raw.githubusercontent.com/BarathiGanesh-HB/IWSPA-AP/d854f44ec2f26d8110bed9450869ac97e19db754/data/IWSPA-AP-traindata.zip',sha256='a1c65c95ac42d723fbebd1ee4f633824e4883c27d0973be4738b34f2e2fdd931')]
SEED='v85-en-hard-20261116-v1'
def freeze(raw,old,output,lidpath):
 import v77_mail
 v77_mail.ADDRESS=EmailSubstitution(v77_mail.ADDRESS)
 assert sha(lidpath)==LID_SHA
 lid=fasttext.load_model(str(lidpath));stats=collections.Counter();pool={};conflicts=set()
 raw.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
 for source in SOURCES:
  path=raw/source['filename']
  if not path.exists():
   r=requests.get(source['url'],timeout=120);r.raise_for_status();path.write_bytes(r.content)
  if sha(path)!=source['sha256']:raise ValueError('Pinned source drift')
  with zipfile.ZipFile(path) as z:
   for name in sorted(z.namelist()):
    if '__MACOSX' in name or not name.endswith('.txt'):continue
    if source['name'].startswith('CSDMC'):
     if not re.fullmatch(r'(ham|spam)/TRAIN_\d{5}\.eml\.txt',name):continue
     y=source['label'];b=z.read(name)
     # Mirror prepended subject directly to a generated mbox envelope.
     if b'From nobody ' in b:
      subject,rest=b.split(b'From nobody ',1)
      if b'\n' not in rest:stats['malformed']+=1;continue
      text=normalize(subject.decode('utf8','replace')+' '+parse_eml(rest.split(b'\n',1)[1]))
      stats['csdmc_mime_envelope_view']+=1
     else:
      text=normalize(clean('',b.decode('utf8','replace')))
      stats['csdmc_plain_subject_body_view']+=1
     src='CSDMC2010'
    else:
     if '/legit/' in name:y=0
     elif '/phish/' in name:y=1
     else:continue
     text=normalize(clean('',z.read(name).decode('utf8','replace')));src='IWSPA-AP-2018'
    stats[f'input:{src}:{y}']+=1
    if len(text)<30 or len(text.split())<10:stats['short']+=1;continue
    if not english_prediction(lid,text)[0]:stats['non_english']+=1;continue
    identity,family=identifiers(text)
    row=dict(identity=identity,family=family,text=text,y=y,source=src,hard=False,parent_exposures=0)
    if identity in pool:
     stats['duplicate']+=1
     if pool[identity]['y']!=y:conflicts.add(family)
    else:pool[identity]=row
 labels=collections.defaultdict(set)
 for r in pool.values():labels[r['family']].add(r['y'])
 conflicts|={f for f,ys in labels.items() if len(ys)>1}
 # Index only candidate windows, then stream the fixed old corpus; no model scores.
 win=collections.defaultdict(set);families=collections.defaultdict(set)
 for r in pool.values():
  families[r['family']].add(r['identity'])
  for w in windows(r['text']):win[w].add(r['family'])
 rejected=set(conflicts);old_frozen=json.loads((old/'manifest.json').read_text())
 for partition in ('train','select'):
  rows=list(read_rows(old/(partition+'.jsonl.gz')))
  if fingerprint(rows)!=old_frozen[partition+'_fingerprint']:raise ValueError('Old exclusion corpus drift')
  for r in rows:
   if r['identity'] in pool:rejected.add(pool[r['identity']]['family'])
   if r['family'] in families:rejected.add(r['family'])
   for w in windows(r['text']):rejected.update(win.get(w,()))
  stats['old_'+partition+'_rows']=len(rows)
 kept=[r for r in pool.values() if r['family'] not in rejected]
 stats['conflicting_families']=len(conflicts);stats['overlap_or_conflict_rows_removed']=len(pool)-len(kept)
 # Connected quote/prefix components stay together before any model sees scores.
 parent={r['identity']:r['identity'] for r in kept}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 first={}
 for r in kept:
  for key in [('family',r['family'])]+[('window',w) for w in windows(r['text'])]:
   if key in first:union(r['identity'],first[key])
   else:first[key]=r['identity']
 partitions={k:[] for k in ('train','select','test')}
 for r in sorted(kept,key=lambda r:r['identity']):
  group=find(r['identity']);r['split_group']=group
  bucket=int(hashlib.sha256((SEED+':'+group).encode()).hexdigest(),16)%100
  partitions['train' if bucket<70 else 'select' if bucket<85 else 'test'].append(r)
 counts={k:dict(n=len(rows),spam=sum(r['y'] for r in rows),ham=sum(1-r['y'] for r in rows)) for k,rows in partitions.items()}
 ready=all(min(v['spam'],v['ham'])>= (50 if k=='train' else 30) for k,v in counts.items())
 manifest=dict(version=85,campaign='v85-en-hard-20261116',sources=SOURCES,counts=counts,by_source_class={k:dict(sorted(collections.Counter(r['source']+':'+str(r['y']) for r in rows).items())) for k,rows in partitions.items()},fingerprints={k:fingerprint(v) for k,v in partitions.items()},stats=dict(stats),language_model_sha256=LID_SHA,exclusion_v84_frozen=old_frozen['dataset_version'],old_manifest_sha256=sha(old/'manifest.json'),old_train_and_select_excluded_by_identity_family_and20word_windows=True,split_seed=SEED,split_before_model_scores=True,quote_components_disjoint=True,training_ready=ready,customer_mail_used=False,raw_mail_in_git=False,scope='CSDMC spam/ham plus IWSPA phishing/legitimate email; not proof of universal spam detection',rights='CSDMC mirror MIT applies to repository software; upstream mail-text rights not separately established. IWSPA mirror no standard license found; research pilot only, no raw redistribution. Preserve source attribution and uncertainty.',csdmc_count_note='Pinned mirror actually contains2949 HAM/1378 SPAM; README says2929 HAM. Counts inherited from explicit directory labels, discrepancy preserved.')
 for k,rows in partitions.items():write_rows(output/(k+'.jsonl.gz'),rows)
 (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
 return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,default=Path('.cache/v85/raw'));p.add_argument('--old',type=Path,default=Path('.cache/v84/corpus'));p.add_argument('--output',type=Path,default=Path('.cache/v85/corpus'));p.add_argument('--lid',type=Path,default=Path('.cache/v79/frozen/lid.176.ftz'));a=p.parse_args();freeze(a.raw,a.old,a.output,a.lid)
