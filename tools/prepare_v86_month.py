"""Freeze the approved 100k new email core plus inherited-budget HAM support."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import fasttext
import requests
import torch

import v77_mail
from v77_mail import clean, parse_eml
from v78_data import identifiers
from v79_data import normalize, english_prediction, LID_SHA
from v84_replay_data import EmailSubstitution, fingerprint, read_rows, write_rows, sha
from v86_exclude_prior import exclude
from v86_month import PARENT_SHA, PARENT_STEP, CAMPAIGN


def rebuild(args):
    """Public sources only; normalized text stays in the runner cache."""
    root = args.work; root.mkdir(parents=True, exist_ok=True)
    archive = root/'archives'
    if not (archive/'audit.json').exists():
        subprocess.run(['python','-u','tools/v86_archive_audit.py','--raw',str(root/'raw'),
                        '--output',str(archive),'--lid',str(args.lid)],check=True)
    if not Path('.cache/v85/corpus/manifest.json').exists():
        subprocess.run(['python','-u','tools/v85_hard_data.py','--old',str(args.old)],check=True)
    v85 = json.loads(Path('research/v85/frozen.json').read_text())
    v84 = json.loads(Path('research/v84/frozen.json').read_text())
    prior = [('v84',args.old,{k:v84[k+'_fingerprint'] for k in ('train','select')}),
             ('v85',Path('.cache/v85/corpus'),v85['fingerprints'])]
    if sha(args.lid) != LID_SHA: raise ValueError('Language model drift')
    lid = fasttext.load_model(str(args.lid)); v77_mail.ADDRESS=EmailSubstitution(v77_mail.ADDRESS)
    repo=root/'trec06';revision='0c16fa9f7d6c24927a34f2dea3712152595374e8'
    if not (repo/'.git').exists():
        subprocess.run(['git','init',str(repo)],check=True,stdout=subprocess.DEVNULL)
        subprocess.run(['git','-C',str(repo),'remote','add','origin','https://huggingface.co/datasets/lepuer/trec06p'],check=True)
        subprocess.run(['git','-C',str(repo),'fetch','--depth=1','origin',revision],check=True)
    labels=subprocess.check_output(['git','-C',str(repo),'show',revision+':labels'])
    if hashlib.sha256(labels).hexdigest()!='146f8df1ef20b623b083941eef56b29eb8d7e034aa8552a19f479d71094170ac':raise ValueError('TREC labels drift')
    truth={path.removeprefix('../'):int(label=='spam') for line in labels.decode().splitlines() for label,path in [line.split()]}
    trec=[];seen=set()
    proc=subprocess.Popen(['git','-C',str(repo),'archive',revision,'data'],stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:
        for member in tar:
            if not member.isfile():continue
            text=normalize(parse_eml(tar.extractfile(member).read()))
            if len(text)<30 or len(text.split())<10 or not english_prediction(lid,text)[0]:continue
            identity,family=identifiers(text)
            if identity in seen:continue
            seen.add(identity);trec.append(dict(identity=identity,family=family,text=text,y=truth[member.name],source='TREC-06-original-mirror',origin=member.name))
    if proc.wait():raise ValueError('TREC archive failed')
    write_rows(root/'trec.jsonl.gz',trec)
    if not (root/'trec-new/audit.json').exists():exclude([root/'trec.jsonl.gz'],prior,root/'trec-new')
    curated=[];seen=set();csv.field_size_limit(10*1024*1024)
    for source in json.loads(Path('research/v86/curated-ham-source-audit.json').read_text())['files']:
        raw=requests.get(source['url'],timeout=120);raw.raise_for_status();raw=raw.content
        if hashlib.sha256(raw).hexdigest()!=source['sha256']:raise ValueError('Curated source drift')
        name=source['url'].rsplit('/',1)[1]
        for row in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
            if row['label']!='0':continue
            text=normalize(clean(row.get('subject',''),row.get('body','')))
            if len(text)<30 or len(text.split())<10 or not english_prediction(lid,text)[0]:continue
            identity,family=identifiers(text)
            if identity in seen:continue
            seen.add(identity);curated.append(dict(identity=identity,family=family,text=text,y=0,source='Zenodo-8339691-'+name))
    families={r['family'] for r in curated};identities={r['identity']:r['family'] for r in curated};bad=set()
    for _,directory,parts in prior:
        for part in parts:
            for row in read_rows(directory/(part+'.jsonl.gz')):
                if row['identity'] in identities:bad.add(identities[row['identity']])
                if row['family'] in families:bad.add(row['family'])
    write_rows(root/'curated.jsonl.gz',[r for r in curated if r['family'] not in bad])
    if not (root/'novel/audit.json').exists():exclude([archive/'candidates.jsonl.gz',root/'trec-new/candidates.jsonl.gz',root/'curated.jsonl.gz'],prior,root/'novel')
    rows=list(read_rows(root/'novel/candidates.jsonl.gz'))
    expected=json.loads(Path('research/v86/candidate-cohort.json').read_text())
    if fingerprint(rows)!=expected['parent_candidate_pool_fingerprint']:raise ValueError('Novel pool differs from reviewed source cohort')
    ham=[r for r in rows if r['y']==0];spam=[r for r in rows if r['y']==1]
    seed=expected['selection_seed'];spam.sort(key=lambda r:hashlib.sha256((seed+':'+r['identity']).encode()).hexdigest())
    core=sorted(ham+spam[:100000-len(ham)],key=lambda r:r['identity'])
    if fingerprint(core)!=expected['canonical_fingerprint']:raise ValueError('New100k cohort drift')
    return core


def main():
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,default=Path('.cache/v84/corpus'))
    p.add_argument('--parent',type=Path,default=Path('.cache/v86/parent/checkpoint.pt'))
    p.add_argument('--output',type=Path,default=Path('.cache/v86/corpus'))
    p.add_argument('--work',type=Path,default=Path('.cache/v86/work'))
    p.add_argument('--lid',type=Path,default=Path('.cache/v79/frozen/lid.176.ftz'))
    p.add_argument('--reviewed-core',type=Path);p.add_argument('--write-lock',action='store_true');args=p.parse_args()
    if sha(args.parent)!=PARENT_SHA:raise ValueError('Required reviewed parent missing')
    parent=torch.load(args.parent,weights_only=True);progress=parent['progress']
    if progress['step']!=PARENT_STEP or progress['active_session'] is not None:raise ValueError('Parent must be completed46000')
    old=list(read_rows(args.old/'train.jsonl.gz'));select=list(read_rows(args.old/'select.jsonl.gz'))
    if fingerprint(old)!=parent['metadata']['corpus']['train_fingerprint'] or fingerprint(select)!=parent['metadata']['corpus']['select_fingerprint']:raise ValueError('Parent source data drift')
    core=list(read_rows(args.reviewed_core)) if args.reviewed_core else rebuild(args)
    core_lock=json.loads(Path('research/v86/candidate-cohort.json').read_text())
    if len(core)!=100000 or fingerprint(core)!=core_lock['canonical_fingerprint']:raise ValueError('Exactly reviewed100000 required')
    support=[]
    for i,row in enumerate(old):
        count=int(progress['exposures'][i])
        if row['y']==0 and count<3:
            support.append(dict(row,original_v84_parent_exposures=row['parent_exposures'],parent_exposures=count,support_role='previously_seen_ham'))
    support.sort(key=lambda r:r['identity'])
    support_lock=json.loads(Path('research/v86/proposed-ham-support.json').read_text())
    if len(support)!=92890 or fingerprint(support)!=support_lock['canonical_fingerprint']:raise ValueError('Approved92890 support pool drift')
    train=[dict(r,hard=False,parent_exposures=0,cohort='new') for r in core]+[dict(r,hard=False,cohort='ham-support') for r in support]
    train.sort(key=lambda r:r['identity'])
    if len({r['identity'] for r in train})!=192890 or {r['family'] for r in train}&{r['family'] for r in select}:raise ValueError('Canonical duplication or SELECT leakage')
    manifest=dict(version=86,dataset_version='v86-en-month-20261118-v1',campaign=CAMPAIGN,train_rows=len(train),new_core_rows=100000,new_core_spam=99902,new_core_ham=98,support_rows=92890,support_remaining_presentations=sum(3-r['parent_exposures'] for r in support),train_spam=sum(r['y'] for r in train),train_ham=sum(1-r['y'] for r in train),select_rows=len(select),test_rows=0,train_fingerprint=fingerprint(train),select_fingerprint=fingerprint(select),parent_train_fingerprint=parent['metadata']['corpus']['train_fingerprint'],parent_checkpoint_sha256=PARENT_SHA,parent_step=PARENT_STEP,select_is_development=True,new_independent_test_available=False,training_ready=True,user_approved_old_ham_support='2026-11-18: Можно можно',max_lifetime_row_exposures=3,source_catalog_sha256=sha(Path('research/v86/archive-sources.json')),new_core_original_fingerprint=core_lock['canonical_fingerprint'],support_original_fingerprint=support_lock['canonical_fingerprint'],class_source_counts=dict(Counter(r['source']+':'+str(r['y']) for r in train)),customer_mail_used=False,source_label_limits='Inherited labels; spam-trap positives weak; legacy Merged-public provenance incomplete; native emails only',raw_mail_in_git=False)
    lock=Path('research/v86/frozen.json')
    if args.write_lock:lock.write_text(json.dumps(manifest,indent=2)+'\n')
    elif not lock.exists() or json.loads(lock.read_text())!=manifest:raise ValueError('Frozen v86 corpus drift')
    args.output.mkdir(parents=True,exist_ok=True)
    write_rows(args.output/'train.jsonl.gz',train);write_rows(args.output/'select.jsonl.gz',select)
    write_rows(args.output/'test.jsonl.gz',[])
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
