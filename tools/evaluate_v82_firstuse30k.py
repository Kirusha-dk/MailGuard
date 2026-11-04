"""Freeze and score genuinely first-use English spam; never fit any parameters.

The new year2012 was absent from both earlier500k benchmarks. Prior content
keys and all later v64 test rows are still blocked, as are all v82 partitions.
Spam-trap labels cannot measure false positives on legitimate mail.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import urllib.request

COUNT=30000
SEED='v82-firstuse30k-english-2012-20261104'
LOCK='1566901303dde37cbbeec93e22cc5d513e3bc2248dff79e6a0f171215f6a197f'
CUTOFF=0.03910822421312332
ARCHIVE_SHA='65bdb680fe95f3d2f52d24d99af7b7d6490c03fc5508313e8f5edefb4672de41'
SOURCE='https://untroubled.org/spam/2012.7z'
PRIOR_SHA='17d0418fc6088176adc03ac623a8f0349da983c2b756295b82e47e9faded2011'


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read_rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)


def dump(p,value):
    Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def current_index(checkpoint):
    from v82_neural import fingerprint
    import torch
    cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if cp['progress']['step']!=8000 or cp['progress']['active_session'] is not None:
        raise ValueError('Expected completed step8000 state')
    ids=set();families=set();parts={'train':[],'select':[]};sources={}
    for root,reference in ((Path('.cache/v79/frozen'),Path('reports/v79-frozen.json')),
                           (Path('.cache/v80/extra'),Path('reports/v80-extra-frozen.json'))):
        manifest_path=root/('frozen.json')
        if json.loads(manifest_path.read_text())!=json.loads(reference.read_text()):
            raise ValueError('Original frozen corpus drift')
        for part in ('train','select','cal','test','fresh-test'):
            p=root/(part+'.jsonl.gz')
            if not p.exists():continue
            sources[str(p)]=sha(p)
            for r in read_rows(p):
                ids.add(r['identity']);families.add(r['family'])
                if part in parts:parts[part].append(r)
    if any(cp['metadata'][k]!=fingerprint(v) for k,v in parts.items()):
        raise ValueError('Checkpoint TRAIN/SELECT content fingerprints mismatch')
    return ids,families,sources


def old_index():
    from prepare_v64_transfer import checked_blocks,view_keys
    audit=json.loads(Path('reports/v64-transfer-result.json').read_text())['audit']
    for report in ('reports/v62-500k-result.json','reports/v64-transfer-result.json'):
        if 'SpamArchive-2012' in json.loads(Path(report).read_text())['audit']['bySource']:
            raise ValueError('Selected year already appeared in an old benchmark')
    blocked,evidence=checked_blocks('prior-blocks')
    if evidence['databaseSha256']!=PRIOR_SHA:
        raise ValueError('Wrong legacy exclusion database')
    old_counts={}
    for shard,expected in audit['shardSha256'].items():
        files=list(Path('old-tests').rglob('input-'+shard+'.jsonl.gz'))
        if len(files)!=1 or sha(files[0])!=expected:raise ValueError('Old500k shard drift')
        count=0
        for r in read_rows(files[0]):
            blocked.update(('identity:'+r['identity'],'family:'+r['family'],*view_keys(r)))
            count+=1
        if count!=100000:raise ValueError('Incomplete prior evaluated shard')
        old_counts[shard]=count
        print(json.dumps(dict(prior_shard=shard,rows=count)),flush=True)
    return blocked,dict(legacy_database_sha256=PRIOR_SHA,legacy_blocked_keys=evidence['blockedKeys'],
        extra_prior_evaluated_rows=sum(old_counts.values()),extra_prior_shards=old_counts)


def old_keys(raw):
    from prepare_v64_transfer import fast_content_keys,view_keys
    from train_v55_multilingual import views,campaign_key
    from v59_mail import from_eml
    from train_v47_phishing_aware import from_eml as legacy_parse
    keys={'raw:'+hashlib.sha256(raw).hexdigest()}
    for parser in (from_eml,legacy_parse):
        p=parser(raw);f,c=views(p['ngram_text'])
        keys.update(fast_content_keys(p)[0]);keys.add('family:'+campaign_key(p['ngram_text']))
        keys.update(view_keys(dict(focus=f,context=c)))
    return keys


def freeze(args):
    import fasttext
    from v77_mail import parse_eml,language
    from v79_data import normalize,english_prediction,LID_SHA
    from v78_data import identifiers
    from v82_neural import fingerprint
    from prepare_v64_transfer import enable_equivalent_normalizers
    from evaluate_v54_unique500k import has_mail_headers
    enable_equivalent_normalizers()
    if sha(args.state)!=LOCK:raise ValueError('Checkpoint lock failed')
    source_report=json.loads((args.state.parent/'report.json').read_text())
    if source_report['global_optimizer_steps']!=8000 or source_report['development']['threshold']!=CUTOFF:
        raise ValueError('Predeclared step/threshold differs from original report')
    args.out.mkdir(parents=True,exist_ok=False)
    ids,families,current_sources=current_index(args.state)
    blocked,prior_evidence=old_index()
    lid_path=Path('.cache/v79/frozen/lid.176.ftz')
    if sha(lid_path)!=LID_SHA:raise ValueError('Language model drift')
    lid=fasttext.load_model(str(lid_path))
    archive=args.work/'2012.7z';args.work.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        with urllib.request.urlopen(SOURCE,timeout=120) as response,archive.open('wb') as f:
            shutil.copyfileobj(response,f)
    if sha(archive)!=ARCHIVE_SHA:raise ValueError('New source archive changed')
    listing=subprocess.run(['7z','l','-slt',str(archive)],check=True,capture_output=True,text=True).stdout
    for entry in listing.split('----------\n',1)[-1].split('\n\n'):
        if 'Symbolic Link =' in entry or 'Hard Link =' in entry:raise ValueError('Archive link rejected')
        for line in entry.splitlines():
            if line.startswith('Path = '):
                p=PurePosixPath(line[7:].replace('\\','/'))
                if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
    dest=args.work/'mail'
    subprocess.run(['7z','x','-y','-bd','-bso0','-bsp0','-o'+str(dest),str(archive)],check=True)
    files=[p for p in dest.rglob('*') if p.is_file()]
    files.sort(key=lambda p:hashlib.sha256((SEED+':'+str(p.relative_to(dest))).encode()).hexdigest())
    counts=Counter();rows=[];selected_keys=set();selected_ids=set();selected_families=set();origins=[]
    for p in files:
        counts['files_considered']+=1
        if p.is_symlink():raise ValueError('Extracted symlink')
        if not 30<=p.stat().st_size<=250000:counts['raw_length_excluded']+=1;continue
        raw=p.read_bytes()
        if not has_mail_headers(raw):counts['non_email_excluded']+=1;continue
        try:text=normalize(parse_eml(raw))
        except Exception:counts['mime_parse_excluded']+=1;continue
        if not 30<=len(text)<=250000 or language(text)!='en' or not english_prediction(lid,text)[0]:
            counts['non_english_or_short_excluded']+=1;continue
        ident,family=identifiers(text)
        if ident in ids or family in families:counts['v82_or_old_regression_overlap_excluded']+=1;continue
        if ident in selected_ids or family in selected_families:counts['new_cohort_duplicate_family_excluded']+=1;continue
        try:keys=old_keys(raw)
        except Exception:counts['legacy_parse_excluded']+=1;continue
        if blocked.intersection(keys):counts['previous_project_exposure_excluded']+=1;continue
        if selected_keys.intersection(keys):counts['new_cohort_legacy_duplicate_excluded']+=1;continue
        selected_keys.update(keys);selected_ids.add(ident);selected_families.add(family)
        rows.append(dict(identity=ident,family=family,text=text,y=1,source='SpamArchive-2012',kind='spam-trap'))
        origins.append(dict(origin=str(p.relative_to(dest)),raw_sha256=hashlib.sha256(raw).hexdigest(),identity=ident,family=family))
        if len(rows)%1000==0:print(json.dumps(dict(eligible=len(rows),counts=dict(counts))),flush=True)
        if len(rows)==COUNT:break
    lock=dict(count=len(rows),requested=COUNT,spam=len(rows),ham=0,checkpoint_sha256=LOCK,
        global_optimizer_steps=8000,threshold=CUTOFF,source_url=SOURCE,source_archive_sha256=ARCHIVE_SHA,
        language_filter='pinned fastText en>=0.8; English-script gate',
        sample_rule='Source filenames ordered by SHA256(seed:relative_path); first30000 eligible distinct identities and families; no classifier scores',
        seed=SEED,cohort_fingerprint=fingerprint(rows),exclusions=dict(counts),
        current_partition_hashes=current_sources,prior_exclusion_evidence=prior_evidence,
        new_source_year_relative_to_old500k=True,new_publisher=False,
        exact_current_training_overlap=0,current_select_cal_test_overlap=0,
        previous_scored500k_overlap=0,legacy_exclusion_key_overlap=0,
        semantic_near_duplicates_fully_excluded=False,source_labels='Spam trap, not individually adjudicated',
        false_positive_rate_evaluable=False,first_use_freeze_before_model_inference=True)
    dump(args.out/'input-lock.json',lock);dump(args.out/'origins.json',origins)
    if len(rows)!=COUNT:raise ValueError(f'Only {len(rows)}/{COUNT} eligible first-use English mail; no refill with old data')
    with gzip.open(args.work/'selected.jsonl.gz','wt') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    dump(args.out/'prepared-file.json',dict(selected_sha256=sha(args.work/'selected.jsonl.gz')))
    shutil.rmtree(dest)
    print(json.dumps(lock,indent=2),flush=True)


def score(args):
    import numpy as np
    from compare_v82_v80 import initialize_scorer,score_chunk
    from audit_v82_errors import metric
    from v82_neural import fingerprint
    lock=json.loads((args.out/'input-lock.json').read_text())
    if lock['count']!=COUNT or lock['checkpoint_sha256']!=LOCK or lock['threshold']!=CUTOFF:
        raise ValueError('Pre-inference lock drift')
    path=args.work/'selected.jsonl.gz'
    if sha(path)!=json.loads((args.out/'prepared-file.json').read_text())['selected_sha256']:
        raise ValueError('Frozen input bytes changed')
    rows=list(read_rows(path))
    if fingerprint(rows)!=lock['cohort_fingerprint']:raise ValueError('Frozen cohort drift')
    backbone=Path('.cache/v82/backbone');m=json.loads((backbone/'manifest.json').read_text())
    for name,expected in m['files'].items():
        if sha(backbone/name)!=expected:raise ValueError('Frozen backbone drift')
    initialize_scorer(backbone,args.state,LOCK);scores=[]
    for start in range(0,len(rows),512):
        scores.extend(score_chunk([r['text'] for r in rows[start:start+512]]))
        print(json.dumps(dict(scored=len(scores),total=COUNT)),flush=True)
    if sha(args.state)!=LOCK:raise ValueError('Checkpoint changed during inference')
    result=dict(input_lock=lock,metrics=metric(rows,scores,CUTOFF),optimizer_steps_executed=0,
        checkpoint_unchanged=True,parameters_rng_and_gradients_unchanged=True,threshold_refitted=False,
        false_positive_rate_evaluable=False,accuracy_on_general_mail=None,
        first_use_against_tracked_prior_datasets=True,independent_new_publisher_test=False,
        limitations=['Spam-only cohort: recall can be measured, HAM false positives cannot',
        'Historical2012 real mail from a previously used spam-trap provider',
        'Trap labels are weak labels; exact/template exclusions do not prove complete semantic independence'],
        accepted_for_production=False,customer_mail_used=False)
    dump(args.out/'report.json',result)
    np.savez_compressed(args.out/'scores.npz',identity=np.array([r['identity'] for r in rows]),scores=np.array(scores))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['freeze','score'],required=True)
    p.add_argument('--state',type=Path,default=Path('fresh-state/checkpoint.pt'))
    p.add_argument('--work',type=Path,default=Path('.cache/firstuse30k'))
    p.add_argument('--out',type=Path,default=Path('fresh30k-output'))
    args=p.parse_args();(freeze if args.phase=='freeze' else score)(args)
