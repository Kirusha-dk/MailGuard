#!/usr/bin/env python3
"""v65 source-rescue experiment: deterministic source-aware TRAIN/CAL/HOLDOUT.

This experiment REUSES the previously evaluated v64 transfer corpus. Only
within-corpus adaptation is assessed; it is NOT first-use or customer quality.
Labels of all HOLDOUT rows are never used for model fitting or calibration.
"""
import argparse
import csv
import gzip
import hashlib
import heapq
import json
from collections import Counter,defaultdict
from itertools import islice
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

PRED_RUN = 38636123356
INPUT_RUN = 38635836571
YEARS_100K = ('SpamArchive-2013','SpamArchive-2015')
COLUMNS = ('v64ConservativeMargin','v64HighRecallMargin')
PROFILES = ('v64-conservative','v65-conservative-rescue','v64-high-recall')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def split_family(row):
    """Partition by campaign family, independently of scores and labels."""
    s=row['source']
    n=int(hashlib.sha256(('v65-family-v1:'+row['family']).encode()).hexdigest()[:12],16)%100
    if s in ('GenSpam','Published-ASCII-mail'):
        return 'train' if n<60 else 'cal' if n<80 else 'holdout'
    if s=='SpamArchive-2016':
        return 'train' if n<12 else 'cal' if n<17 else 'holdout'
    return 'holdout'


def text_view(row):
    return (row['focus'][:650]+' '+row['context'][:1600]).strip()


def source_weight_counts(train):
    c=Counter((r['source'],r['y']) for r in train)
    return np.array([min(1.,5000/c[(r['source'],r['y'])])*(1.5 if r['y']==0 else 1.)
                     for r in train],dtype=float)


def vectors():
    return {
        'word':HashingVectorizer(ngram_range=(1,2),n_features=2**18,
                                 alternate_sign=False,norm='l2',dtype=np.float32),
        'char':HashingVectorizer(analyzer='char',ngram_range=(3,5),n_features=2**19,
                                 alternate_sign=False,norm='l2',dtype=np.float32)
    }


def fit_experts(train,epochs=4):
    if len({r['y'] for r in train})!=2:raise ValueError('TRAIN needs both classes')
    reps=vectors();models={}
    rng=np.random.default_rng(65)
    y=np.asarray([r['y'] for r in train],dtype=int)
    weights=source_weight_counts(train)
    for name,vectorizer in reps.items():
        model=SGDClassifier(loss='log_loss',alpha=2e-5,average=True,random_state=65,
                            learning_rate='optimal')
        for epoch in range(epochs):
            order=rng.permutation(len(train))
            for start in range(0,len(order),192):
                ix=order[start:start+192]
                x=vectorizer.transform([text_view(train[i]) for i in ix])
                model.partial_fit(x,y[ix],classes=np.array([0,1]),sample_weight=weights[ix])
            print('V65 TRAIN',name,'epoch',epoch+1,'/',epochs,flush=True)
        models[name]=model
    return reps,models


def new_scores(reps,models,batch):
    x=[text_view(r) for r in batch]
    return sum(.5*models[k].decision_function(reps[k].transform(x))
               for k in ('word','char'))


def score_active(reps,models,batch):
    result=np.full(len(batch),float('-inf'))
    ix=[i for i,r in enumerate(batch) if r['script']!='cjk' and float(r['v64ConservativeMargin'])<0]
    if ix:
        result[ix]=new_scores(reps,models,[batch[i] for i in ix])
    return result


def fpr_zero_cutoff(rows,scores):
    """Require zero CAL FP from the additional specialist; no TEST peeking."""
    ham=np.asarray([float(s) for r,s in zip(rows,scores)
                    if r['y']==0 and r['script']!='cjk'
                    and float(r['v64ConservativeMargin'])<0])
    if len(ham)<500:raise ValueError('Too little CAL HAM to define even a screen')
    return float(np.nextafter(ham.max(),float('inf')))


def decisions(row,new_score,cutoff):
    old=float(row['v64ConservativeMargin'])>=0
    rescued=row['script']!='cjk' and float(new_score)>=cutoff
    return (old,old or rescued,float(row['v64HighRecallMargin'])>=0)


def tally_one(acc,row,preds):
    for i,p in enumerate(PROFILES):
        truth=int(row['y']);pred=bool(preds[i])
        acc[p]['tp' if truth and pred else 'fn' if truth else 'fp' if pred else 'tn']+=1


def metrics(counter):
    tp,fn,fp,tn=(counter[k] for k in ('tp','fn','fp','tn'))
    return dict(n=tp+fn+fp+tn,spam=tp+fn,ham=fp+tn,tp=tp,fn=fn,fp=fp,tn=tn,
                recall=tp/(tp+fn) if tp+fn else None,
                fpr=fp/(fp+tn) if fp+tn else None)


def pair_shards(inputs,predictions,audit):
    for shard in range(5):
        inp=list(Path(inputs).rglob(f'input-{shard}.jsonl.gz'))
        pred=list(Path(predictions).rglob(f'predictions-{shard}.csv.gz'))
        meta=list(Path(predictions).rglob(f'shard-{shard}.json'))
        if (len(inp),len(pred),len(meta))!=(1,1,1):
            raise ValueError(f'Incomplete or ambiguous shard {shard}')
        m=json.loads(meta[0].read_text())
        if m['inputSha256']!=audit['shardSha256'][str(shard)] or m['predictionsSha256']!=digest(pred[0]):
            raise ValueError('Frozen input/prediction checksum mismatch')
        if not m['first1000ParityPassed'] or m['modelsSha256']['v64']!='ecd8bb221594538698188fd3270daab840a0a43f672c1fcc6af272bc380190e5':
            raise ValueError('Not the verified v64 margin artifact')
        if digest(inp[0])!=m['inputSha256']:raise ValueError('Modified source input')
        with gzip.open(inp[0],'rt',encoding='utf-8') as a,gzip.open(pred[0],'rt',newline='') as b:
            for i,(raw,score) in enumerate(zip(a,csv.DictReader(b))):
                row=json.loads(raw)
                if (row['identity'],row['family'],int(row['y']),row['source'])!=(
                      score['identity'],score['family'],int(score['label']),score['source']):
                    raise ValueError(f'Input/score row misalignment in shard {shard}')
                for key in COLUMNS:row[key]=float(score[key])
                yield row
            if i+1 != 100000:raise ValueError('Incomplete shard')


def retain_fixed_100k(heap,row,selected):
    """Keep the 100000 smallest predeclared identity hashes for each year."""
    rank=int(hashlib.sha256(('v65-100k-v1:'+row['identity']).encode()).hexdigest(),16)
    value=(-rank,(int(row['y']),tuple(selected)))
    if len(heap)<100000:heapq.heappush(heap,value)
    elif value[0]>heap[0][0]:heapq.heapreplace(heap,value)


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--inputs',default='prepared')
    a.add_argument('--predictions',default='combined')
    a.add_argument('--output',default='reports/v65-domain-rescue')
    a.add_argument('--epochs',type=int,default=4)
    args=a.parse_args()
    audit=json.loads((Path(args.inputs)/'audit.json').read_text())
    if audit['count']!=500000 or not audit['zeroDetectedPriorOverlap']:
        raise ValueError('Wrong prior experiment')
    # First pass: no prediction margins or HOLDOUT labels influence split.
    train,cal=[],[]
    counts=Counter();family_splits={};seen=set()
    for r in pair_shards(args.inputs,args.predictions,audit):
        if r['identity'] in seen:raise ValueError('Identity duplicate')
        seen.add(r['identity'])
        split=split_family(r)
        prior=family_splits.setdefault(r['family'],split)
        if prior!=split:raise ValueError('Campaign family crosses TRAIN/CAL/HOLDOUT')
        if split=='holdout':
            counts[(split,r['source'],'label-unread')]+=1
        else:
            counts[(split,r['source'],r['y'])]+=1
        if split=='train' and r['script']!='cjk':train.append(r)
        if split=='cal':cal.append(r)
    if len(seen)!=500000 or sum(k[0]=='holdout' and v for k,v in counts.items())<100000:
        raise ValueError('Not enough held-out messages')
    if sum(r['y']==0 for r in cal)<500:raise ValueError('Inadequate CAL HAM')
    print('V65 FROZEN PARTITION',dict((str(k),v) for k,v in counts.items()),flush=True)
    reps,models=fit_experts(train,args.epochs)
    cal_new=np.concatenate([score_active(reps,models,cal[i:i+128])
                             for i in range(0,len(cal),128)])
    cutoff=fpr_zero_cutoff(cal,cal_new)
    verify=defaultdict(Counter)
    for r,s in zip(cal,cal_new):tally_one(verify,r,decisions(r,s,cutoff))
    if verify['v65-conservative-rescue']['fp']>verify['v64-conservative']['fp']:
        raise ValueError('CAL safety gate failed')
    print('V65 FROZEN CUTOFF',cutoff,'CAL',dict((p,metrics(c)) for p,c in verify.items()),flush=True)
    # Crucial: save immutable model BEFORE reading even one HOLDOUT label.
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    artifact=out/'model.joblib'
    joblib.dump(dict(version='v65-retrospective-adaptation',models=models,
                     cutoff=cutoff,parameters={'epochs':args.epochs,'split':'v65-family-v1'},
                     profile='v64 conservative OR Latin/other specialist'),artifact,compress=3)
    locked=digest(artifact)
    print('V65 MODEL LOCKED',locked,flush=True)
    # Second pass: evaluate only untouched family HOLDOUT and fixed 100k years.
    groups=defaultdict(lambda:defaultdict(Counter))
    heaps={year:[] for year in YEARS_100K}
    holdout_count=0
    iterator=(r for r in pair_shards(args.inputs,args.predictions,audit)
              if split_family(r)=='holdout')
    while batch:=list(islice(iterator,128)):
        new=score_active(reps,models,batch)
        for r,s in zip(batch,new):
            selected=decisions(r,s,cutoff)
            names=['all-holdout','source/'+r['source'],'script/'+r['script']]
            if r['source'] in ('GenSpam','Published-ASCII-mail','Nazario-2025'):
                names.append('independent-supplier-holdout')
            for name in names:tally_one(groups[name],r,selected)
            if r['source'] in heaps:retain_fixed_100k(heaps[r['source']],r,selected)
            holdout_count+=1
        if holdout_count%25000<128:print('V65 HOLDOUT',holdout_count,flush=True)
    if digest(artifact)!=locked:raise ValueError('Model unexpectedly changed after HOLDOUT')
    for year,heap in heaps.items():
        if len(heap)!=100000:raise ValueError('Not 100000 messages: '+year)
        for _,(y,preds) in heap:
            tally_one(groups['fixed-100k/'+year],{'y':y},preds)
    report={name:{p:metrics(c) for p,c in table.items()} for name,table in sorted(groups.items())}
    summary=dict(version=65,kind='RETROSPECTIVE cross-source adaptation experiment',
                 inputRunId=INPUT_RUN,predictionsRunId=PRED_RUN,holdoutCount=holdout_count,
                 trainCount=len(train),calCount=len(cal),splitCounts={str(k):v for k,v in counts.items()},
                 cutoff=cutoff,modelSha256=locked,calibration={p:metrics(c) for p,c in verify.items()},
                 results=report,modelLockedBeforeHoldout=True,testLabelsUsedForTraining=False,
                 optimisticBecausePriorV64ScoresAlreadyInspected=True,
                 confirmedAny100kGeneralization=False,confirmedCustomerQuality=False,
                 prefitHoldoutLabelsUninspected=True,
                 limitation='Previously assessed v64 transfer corpus was used for v65 adaptation. Training and CAL use disjoint campaign families from v65 HOLDOUT, but the source pool was already reported. Archive trap spam labels are weak, two 100k cohorts contain no HAM, and the independent supplier HAM sample is insufficient to prove extremely low production FPR.')
    (out/'result.json').write_text(json.dumps(summary,indent=2))
    md='# MailGuard v65: source adaptation, retrospective experiment\n\n'+summary['limitation']+'\n\n'
    md+='TRAIN '+str(len(train))+'; CAL '+str(len(cal))+'; held-out '+str(holdout_count)+'. Frozen before holdout: '+locked+'.\n\n'
    md+='| Group | Model | N | Spam | Recall | FP/HAM |\n|---|---|---:|---:|---:|---:|\n'
    for name,models_table in report.items():
        if not (name in ('all-holdout','independent-supplier-holdout') or name.startswith(('source/','script/','fixed-100k/'))):continue
        for p,m in models_table.items():
            rec='n/a' if m['recall'] is None else f"{m['recall']:.2%}"
            md+=f"| {name} | {p} | {m['n']} | {m['spam']} | {rec} | {m['fp']} / {m['ham']} |\n"
    (out/'result.md').write_text(md)
    print(md,flush=True)


if __name__=='__main__':main()
