#!/usr/bin/env python3
"""MailGuard v68: Russian/English research model and fixed 10,000-message evaluation.

RU: CC0-licensed Telegram messages (NOT email); EN: previously used seven
English mail collections (RETROSPECTIVE). No customer data, no future source
TEST reuse. DATASET *train* splits only, family-disjoint deterministic partitions,
CAL-only cutoffs, and saved model before reading evaluation labels.

The 10k outcome is an in-domain grouped research holdout, not a promise for
arbitrary unknown suppliers or the customer's real mail.
"""
import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter,defaultdict
from pathlib import Path

RU_ID='alt-gnome/telegram-spam'
EN_ID='JinqiangDing/seven-phishing-email-datasets'
SOURCES={'ru':RU_ID,'en':EN_ID}
SEED='v68-ru-en-2026-10-15'
TEST_PER_LANGUAGE=5000
TRAIN_CAP={'ru':18000,'en':70000}
FEATURE_DIMENSIONS={'word':2**17,'char':2**18}
SPLIT_BUCKETS={'ru':(55,70),'en':(75,90)}
FPR_BUDGET=.0001
WARNING=('This is a retrospective engineering benchmark. RU is Telegram chat, NOT emails; '
         'EN mail corpus includes sources already examined in earlier MailGuard versions. '
         'The 10000-item sample is grouped within two providers, NOT arbitrary independent suppliers. '
         'Neither production 90% recall nor 0.01% FPR is validated. Do not use its TEST to tune thresholds.')

URL=re.compile(r'(?i)https?://\S+|www\.\S+')
EMAIL=re.compile(r'(?iu)[\w.+-]+@[\w.-]+\.[\w.-]+')
DIGITS=re.compile(r'\d+',re.UNICODE)
SPACE=re.compile(r'\s+',re.UNICODE)
def digest(value):
    return hashlib.sha256(value.encode('utf-8','replace')).hexdigest()

def clean(value):
    text=unicodedata.normalize('NFKC',str(value or '')).casefold()
    text=URL.sub(' urltoken ',text)
    text=EMAIL.sub(' emailtoken ',text)
    text=DIGITS.sub(' numbertoken ',text)
    return SPACE.sub(' ',text).strip()

def language_fraction(text):
    chars=[x for x in text if x.isalpha()]
    if not chars:return (0.,0.)
    cyr=sum('\u0400'<=x<='\u052f' for x in chars)
    lat=sum('a'<=x.lower()<='z' for x in chars)
    return cyr/len(chars),lat/len(chars)

def make_row(source,label,text):
    if label not in (0,1) or len(text)>150000:raise ValueError('Invalid publisher labels/size')
    cyr,lat=language_fraction(text)
    if (source=='ru' and cyr<.48) or (source=='en' and lat<.55):
        return None
    canonical=clean(text)[:6000]
    if len(canonical)<30:return None
    # Substituted URLs, emails and digits suppress template overfitting. Protect
    # the holdout from exact and same-prefix variants of campaigns.
    identity=digest(source+'|'+canonical)
    family=digest(source+'|'+canonical[:160])
    cutoff=int(digest(SEED+'|split|'+family)[:12],16)%100
    train_end,cal_end=SPLIT_BUCKETS[source]
    split='train' if cutoff<train_end else 'cal' if cutoff<cal_end else 'test'
    return dict(source=source,y=label,text=canonical,identity=identity,
                family=family,split=split)

def ingest(items,source):
    """Exact dedup; a conflicting-label identity is excluded rather than relabelled."""
    seen={};blocked=set();stats=Counter()
    for label,text in items:
        row=make_row(source,int(label),text)
        if row is None:
            stats['not_english_or_russian_or_too_short']+=1
            continue
        key=row['identity']
        if key in seen:
            stats['exact_duplicate']+=1
            if row['y']!=seen[key]['y']:blocked.add(key);stats['conflicting_labels']+=1
        else:seen[key]=row
    return [r for k,r in seen.items() if k not in blocked],dict(stats)

def partition(rows):
    """Freeze source-first and family group splits before fitting anything."""
    family_split={};identity_to_family={}
    out={s:{split:[] for split in ('train','cal','test')} for s in SOURCES}
    for row in rows:
        fs=family_split.setdefault(row['family'],row['split'])
        if fs!=row['split']:raise ValueError('Family crossed TRAIN/CAL/TEST')
        ident=row['identity'];prev=identity_to_family.setdefault(ident,row['family'])
        if prev!=row['family']:raise ValueError('Repeated identity belongs to two families')
        out[row['source']][row['split']].append(row)
    for source,splits in out.items():
        for split,cap in (('train',TRAIN_CAP[source]),('test',TEST_PER_LANGUAGE)):
            selected=sorted(splits[split],key=lambda r:digest(SEED+'|select|'+r['identity']))
            if split=='test':
                if len(selected)<cap:
                    raise ValueError(f'Not {cap} label-blind TEST rows for {source}: {len(selected)}')
                splits[split]=selected[:cap]
            elif len(selected)>cap:
                splits[split]=selected[:cap]
        if len(splits['cal'])<250:raise ValueError('Insufficient CAL for '+source)
    return out

def sample_fingerprint(splits):
    out={}
    for source,buckets in splits.items():
        out[source]={}
        for split,items in buckets.items():
            # Labels only enter cryptographic prefit provenance, not fitting or tuning for TEST.
            rows=[(r['identity'],r['family']) for r in items]
            out[source][split]=dict(
                n=len(items),
                identityFamilySha256=digest(json.dumps(sorted(rows),separators=(',',':'))))
    return out

def vectors(texts):
    import numpy as np
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import HashingVectorizer
    word=HashingVectorizer(ngram_range=(1,2),n_features=FEATURE_DIMENSIONS['word'],
                           norm='l2',alternate_sign=False,dtype=np.float32)
    char=HashingVectorizer(analyzer='char',ngram_range=(3,5),
                           n_features=FEATURE_DIMENSIONS['char'],
                           norm='l2',alternate_sign=False,dtype=np.float32)
    return hstack([word.transform(texts),char.transform(texts)],format='csr')

def fit(rows,epochs=2):
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    from numpy.random import default_rng
    if not rows or {r['y'] for r in rows}!={0,1}:
        raise ValueError('TRAIN must have both classes')
    model=SGDClassifier(loss='log_loss',alpha=2e-5,random_state=68,
                        average=True,learning_rate='optimal')
    weights=np.array([2. if r['y']==0 else 1. for r in rows],dtype=float)
    ys=np.array([r['y'] for r in rows],dtype=int)
    rng=default_rng(68)
    classes=np.array([0,1])
    for epoch in range(epochs):
        order=rng.permutation(len(rows))
        for offset in range(0,len(order),256):
            ix=order[offset:offset+256]
            data=vectors([rows[i]['text'] for i in ix])
            model.partial_fit(data,ys[ix],classes=classes,sample_weight=weights[ix])
        print('v68 train',len(rows),'epoch',epoch+1,flush=True)
    return model

def score(model,rows):
    import numpy as np
    if not rows:return np.empty(0)
    return np.concatenate([
        model.decision_function(vectors([r['text'] for r in rows[i:i+256]]))
        for i in range(0,len(rows),256)])

def thresholds(scores,labels):
    import numpy as np
    ham=np.sort(np.asarray([float(v) for v,y in zip(scores,labels) if y==0]))[::-1]
    if len(ham)<300:raise ValueError('Too few labelled HAM for a safe CAL threshold')
    return {
        # Identical conservative threshold regardless of TEST outcomes.
        'conservative':float(np.nextafter(ham[0],float('inf'))),
        # Diagnose 0.1% CAL-FPR variant separately; NOT the 0.01% target.
        'research-high-recall':float(np.nextafter(ham[min(len(ham)-1,int(len(ham)*.001))],
                                                 float('inf')))
    }

def confusion(scores,labels,cutoff):
    from audit_v65_transfer_validity import wilson_upper
    tp=fn=fp=tn=0
    for v,y in zip(scores,labels):
        yes=float(v)>=cutoff
        if y==1:
            if yes:tp+=1
            else:fn+=1
        elif y==0:
            if yes:fp+=1
            else:tn+=1
        else:raise ValueError('Invalid label')
    return dict(n=tp+fn+fp+tn,spam=tp+fn,ham=fp+tn,
                tp=tp,fn=fn,fp=fp,tn=tn,
                recall=tp/(tp+fn) if tp+fn else None,
                fpr=fp/(fp+tn) if fp+tn else None,
                fpr95upper=wilson_upper(fp,fp+tn))

def acquire():
    from huggingface_hub import HfApi
    from datasets import load_dataset
    revisions={};rows=[];counts={}
    api=HfApi()
    for lang,name in SOURCES.items():
        revision=api.dataset_info(name).sha
        if not revision or len(revision)!=40:raise ValueError('No immutable HF commit: '+name)
        revisions[lang]=revision
        dataset=load_dataset(name,split='train',revision=revision)
        if lang=='ru':
            if not {'text','label'}<=set(dataset.column_names):raise ValueError('RU schema changed')
            iterator=((int(v['label']),str(v['text'] or '')) for v in dataset)
        else:
            if not {'text','subject','label','dataset_name'}<=set(dataset.column_names):raise ValueError('EN schema changed')
            iterator=((int(v['label']),f"subject {v['subject'] or ''} body {v['text'] or ''}")
                      for v in dataset)
        selected,stats=ingest(iterator,lang)
        counts[lang]=dict(publisherRows=len(dataset),eligible=len(selected),exclusions=stats)
        rows.extend(selected)
        print('v68 indexed',lang,counts[lang],'revision',revision,flush=True)
    return rows,revisions,counts

def main():
    import joblib
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='reports/v68-ru-en-experiment')
    p.add_argument('--epochs',type=int,default=2,choices=(1,2,3))
    args=p.parse_args()
    rows,revisions,input_counts=acquire()
    splits=partition(rows)
    frozen=sample_fingerprint(splits)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    # Freeze identities and exact upstream commits without exposing message text.
    before=dict(sourceRevisions=revisions,partitions=frozen,inputCounts=input_counts,
                sampler='fixed 5000 RU and 5000 EN by sha256(identity), before scoring',
                labelBlindWithinGroup=True,sourceDisjoint=False,
                testPreviouslyUsedElsewhere=True,warning=WARNING)
    (out/'frozen.json').write_text(json.dumps(before,indent=2))
    print('v68 frozen before training',json.dumps(frozen),flush=True)
    del rows
    models={};cuts={};cal_scores={}
    for lang in ('en','ru'):
        model=fit(splits[lang]['train'],epochs=args.epochs)
        cs=score(model,splits[lang]['cal'])
        cal_y=[r['y'] for r in splits[lang]['cal']]
        cut=thresholds(cs,cal_y)
        # All cutoffs are calculated only from CAL HAM, never from TEST.
        models[lang]=model;cuts[lang]=cut
        cal_scores[lang]={name:confusion(cs,cal_y,t) for name,t in cut.items()}
        print('v68 calibrated',lang,cal_scores[lang],flush=True)
    locked=out/'model.joblib'
    joblib.dump(dict(version='v68-ru-en-development-only',models=models,
                     cutoffs=cuts,sourceRevisions=revisions,partitionHashes=frozen,
                     noCustomerMail=True,epochs=args.epochs),locked,compress=3)
    model_sha=hashlib.sha256(locked.read_bytes()).hexdigest()
    print('V68 MODEL LOCKED BEFORE TEST',model_sha,flush=True)
    # Only now access any TEST label for measurement.
    results={};overall={name:Counter() for name in ('conservative','research-high-recall')}
    for lang in ('ru','en'):
        group=splits[lang]['test']
        sc=score(models[lang],group)
        labels=[r['y'] for r in group]
        results[lang]={}
        for profile,cut in cuts[lang].items():
            metric=confusion(sc,labels,cut)
            results[lang][profile]=metric
            for key in ('tp','fn','fp','tn'):overall[profile][key]+=metric[key]
    final={}
    for profile,c in overall.items():
        # Metrics from actual TP FN FP TN counts, never average cohort recalls.
        from audit_v65_transfer_validity import wilson_upper
        tp,fn,fp,tn=[c[k] for k in ('tp','fn','fp','tn')]
        final[profile]=dict(n=tp+fn+fp+tn,spam=tp+fn,ham=fp+tn,
                            tp=tp,fn=fn,fp=fp,tn=tn,
                            recall=tp/(tp+fn) if tp+fn else None,
                            fpr=fp/(fp+tn) if fp+tn else None,
                            fpr95upper=wilson_upper(fp,fp+tn),
                            minLanguageRecall=min(results[s][profile]['recall'] for s in ('ru','en')))
        if final[profile]['n']!=2*TEST_PER_LANGUAGE:
            raise ValueError('Not 10000 TEST messages')
    if hashlib.sha256(locked.read_bytes()).hexdigest()!=model_sha:
        raise ValueError('Model changed after TEST')
    payload=dict(version=68,task='Russian + English only',
                 evaluationKind='IN-DOMAIN retrospective/partial first-use; NOT new customer mail',
                 producerRevisions=revisions,
                 notes=WARNING,modelSha256=model_sha,
                 testWasHeldFromThisEstimator=True,testFamilyOverlap=0,
                 testAndCalThresholdLeakage=False,
                 counts=input_counts,freeze=frozen,
                 cal=cal_scores,testByLanguage=results,testAll=final,
                 claims90OnAny10000Validated=False,productionFPGoalValidated=False,
                 modelSavedBeforeTestLabelsRead=True)
    (out/'result.json').write_text(json.dumps(payload,indent=2))
    lines=['# MailGuard v68 — Russian/English only (research baseline)','',WARNING,'',
           'Model SHA256: '+model_sha,'',
           '| Language | Profile | N | SPAM | Recall | FP / HAM | Wilson 95% FPR upper |',
           '|---|---|---:|---:|---:|---:|---:|']
    for language,d in [*results.items(),('ALL',final)]:
        for profile,m in d.items():
            lines.append(f"| {language} | {profile} | {m['n']} | {m['spam']} | "
                         f"{m['recall']:.2%} | {m['fp']} / {m['ham']} | {m['fpr95upper']:.3%} |")
    lines.extend(['','This is 10000 **grouped samples within known source families** (5000 each language), '
                  'not a random independent stream or an arbitrary source acceptance test.',
                  'Russian samples are Telegram chat, not emails; English samples include historically trained source pools.',
                  '90% target is NOT considered achieved unless independently tested on fresh source-disjoint 10k mail cohorts and with adequate independent HAM.',
                  'All model and score artifacts must be treated as research-only.',
                  ''])
    (out/'result.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
