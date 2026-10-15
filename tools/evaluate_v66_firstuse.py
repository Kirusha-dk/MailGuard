#!/usr/bin/env python3
"""v66 first-use frozen v64/v65 inference, independent of label-driven fitting.

Separate freeze/score GitHub jobs. Freeze persists fingerprints, not mail or scores.
Model binaries, script cutoffs and specialist threshold MUST match prior runs.
"""
import argparse,hashlib,json,math,os,sys
from collections import Counter,defaultdict
from pathlib import Path
from itertools import islice

V64_SHA='ecd8bb221594538698188fd3270daab840a0a43f672c1fcc6af272bc380190e5'
V62_SHA='380991f74ae7ddc4d26cb36fd952ff7a1b6a24a1c8985becf28cb14d2f493aaf'
V65_SHA='c672d33a6f81ccc5649c84997ee3cdd38d6ce3a8707e76c13a8d552778063c29'
SOURCE_AUDIT='reports/v66-candidate-audit.json'
PROFILES=('v64-conservative','v64-high-recall','v65-conservative-rescue')


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def fingerprint(rows):
    """Whole filtered cohort locked before models, without writing sensitive mail."""
    lines=[]
    for r in rows:
        from hashlib import sha256
        def hashed(k):return sha256(r[k].encode('utf-8','replace')).hexdigest()
        lines.append((r['source'],int(r['label']),r['identity'],r['family'],
                      hashed('focus'),hashed('context')))
    lines.sort()
    return hashlib.sha256(json.dumps(lines,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def validate_source_audit(source_meta,old_meta):
    if len(source_meta)!=len(old_meta['sources']):
        raise ValueError('New publisher list changed')
    for x,y in zip(source_meta,old_meta['sources']):
        if x!=y:raise ValueError('New publisher content or provenance changed')


def reconstruct(prior,oldroot,expected_audit,scoring=False):
    """Uses the EXACT v66 screening logic, including all 500k old test rows."""
    import prepare_v66_candidate_audit as v66
    from train_v55_multilingual import views
    from v61_semantic import script
    old=v66.row
    def capture(parsed,label,source,raw=None):
        info=old(parsed,label,source,raw)
        if info is None:return None
        focus,context=views(parsed['ngram_text'])
        info.update(focus=focus,context=context,script=script(dict(context=context)))
        return info
    v66.row=capture
    try:
        from prepare_v64_transfer import enable_equivalent_normalizers
        enable_equivalent_normalizers()
        records=[];source_meta=[]
        for loader in (v66.turkish,v66.phishing):
            items,meta=loader()
            records.extend(items);source_meta.append(meta)
    finally:
        v66.row=old
    validate_source_audit(source_meta,expected_audit)
    unique,collisions=v66.dedup(records)
    eligible,excluded,evidence=v66.check_prior(unique,Path(prior),Path(oldroot))
    klass=Counter(r['label'] for r in eligible)
    classes=dict(Counter(f"{r['source']}:{r['label']}" for r in eligible))
    if len(records)!=expected_audit['totalParsed'] or len(unique)!=expected_audit['uniqueBeforePriorExclusions']:
        raise ValueError('Unexpected source counts')
    if (collisions!=expected_audit['ambiguousAliases'] or excluded!=expected_audit['excludedPreviouslySeen'] or
        evidence!=expected_audit['priorEvidence'] or len(eligible)!=expected_audit['retained'] or
        klass[0]!=expected_audit['retainedHAM'] or klass[1]!=expected_audit['retainedSPAM'] or
        classes!=expected_audit['retainedBySourceClass']):
        raise ValueError('Missing or unexpected v66 screening results')
    if not expected_audit['noTrainingOrInference']:
        raise ValueError('Prior source inventory was not model-blind')
    return eligible


def freeze(args):
    old=json.loads(Path(SOURCE_AUDIT).read_text())
    rows=reconstruct(args.prior,args.old_input,old)
    manifest=dict(stage='pre-inference-v66-fixed-first-use-cohort',
                  count=len(rows),spam=sum(r['label'] for r in rows),
                  ham=sum(1-r['label'] for r in rows),
                  fingerprintSha256=fingerprint(rows),
                  sourceAuditSha256=sha_file(SOURCE_AUDIT),
                  previousV64TrainSha256=V64_SHA,previousV62TrainSha256=V62_SHA,
                  previousV65ModelSha256=V65_SHA,
                  sourceLabelInspectionForModelSelection=False,
                  modelOrThresholdChanged=False,modelLoaded=False,
                  expectedVersions=list(PROFILES),
                  semanticDuplicateScreeningComplete=False,
                  screeningCaveats=old['restrictions'])
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('V66 FROZEN BEFORE MODEL INFERENCE',json.dumps(manifest),flush=True)


def tally_one(table,label,pred):
    kind=('tp' if pred else 'fn') if label else ('fp' if pred else 'tn')
    table[kind]+=1


def counts(c):
    tp,fn,fp,tn=(c[k] for k in ('tp','fn','fp','tn'))
    return dict(n=tp+fn+fp+tn,spam=tp+fn,ham=fp+tn,tp=tp,fn=fn,
                fp=fp,tn=tn,recall=tp/(tp+fn) if tp+fn else None,
                fpr=fp/(fp+tn) if fp+tn else None)


def wilson(fp,n):
    if n==0:return None
    z=1.959963984540054
    p=fp/n;v=z*z
    return (p+v/(2*n)+z*math.sqrt((p*(1-p)+v/(4*n))/n))/(1+v/n)


def score(args):
    import joblib,numpy as np
    from v64_runtime import load_bundle,score_rows
    from v65_domain_rescue import vectors,score_active
    manifest=json.loads(Path(args.manifest).read_text())
    old=json.loads(Path(SOURCE_AUDIT).read_text())
    if manifest.get('stage')!='pre-inference-v66-fixed-first-use-cohort':
        raise ValueError('Missing prior frozen cohort')
    if manifest['sourceAuditSha256']!=sha_file(SOURCE_AUDIT):
        raise ValueError('v66 source audit changed after freeze')
    if manifest['previousV64TrainSha256']!=V64_SHA or manifest['previousV65ModelSha256']!=V65_SHA:
        raise ValueError('Changed model reference in frozen manifest')
    root=Path(args.model64)
    model64=root/'model.joblib'
    parent=root/'v62/model.joblib'
    specialist=Path(args.model65)/'model.joblib'
    for name,path,expected in (('v64',model64,V64_SHA),('v62',parent,V62_SHA),('v65',specialist,V65_SHA)):
        if sha_file(path)!=expected:raise ValueError('Frozen '+name+' model SHA changed')
    bundle=load_bundle(root)
    expert=joblib.load(specialist)
    if expert['version']!='v65-retrospective-adaptation' or expert['profile']!='v64 conservative OR Latin/other specialist':
        raise ValueError('Wrong v65 specialist checkpoint')
    if not math.isfinite(expert['cutoff']):
        raise ValueError('Invalid frozen v65 cutoff')
    print('V66 FROZEN MODELS AND CUTOFF VERIFIED before replay',flush=True)
    rows=reconstruct(args.prior,args.old_input,old,scoring=True)
    if (len(rows)!=manifest['count'] or sum(r['label'] for r in rows)!=manifest['spam'] or
            fingerprint(rows)!=manifest['fingerprintSha256']):
        raise ValueError('Input cohort differs from FIRST-USE frozen manifest')
    results=defaultdict(lambda:defaultdict(Counter));reps=vectors()
    for start in range(0,len(rows),64):
        batch=rows[start:start+64]
        margins=score_rows(bundle,batch)
        for i,r in enumerate(batch):r['v64ConservativeMargin']=float(margins['conservative'][i])
        score_new=score_active(reps,expert['models'],batch)
        for i,r in enumerate(batch):
            conservative=float(margins['conservative'][i])>=0
            higher=float(margins['high-recall'][i])>=0
            add=r['script']!='cjk' and float(score_new[i])>=expert['cutoff']
            selected=(conservative,higher,conservative or add)
            groups=('all','source/'+r['source'],'script/'+r['script'])
            for group in groups:
                for name,p in zip(PROFILES,selected):
                    tally_one(results[group][name],int(r['label']),p)
        print('V66 FROZEN inference',min(start+64,len(rows)),'/',len(rows),flush=True)
    metrics={group:{name:counts(c) for name,c in bucket.items()} for group,bucket in results.items()}
    for g,table in metrics.items():
        for name,m in table.items():
            m['fprWilson95Upper']=wilson(m['fp'],m['ham'])
            if m['n']==0:raise ValueError('Empty metric group')
    for name,m in metrics['all'].items():
        if m['n']!=len(rows) or m['ham']!=manifest['ham'] or m['spam']!=manifest['spam']:
            raise ValueError('Incomplete inference or corrupt labels')
    if sha_file(model64)!=V64_SHA or sha_file(specialist)!=V65_SHA:
        raise ValueError('Model mutated during scoring')
    result=dict(version=66,stage='FIRST-USE labelled publisher mail; source-quality limited',
                parentSourceAuditSha256=manifest['sourceAuditSha256'],
                cohortFingerprintSha256=manifest['fingerprintSha256'],
                frozenModelSha256={'v64':V64_SHA,'v62':V62_SHA,'v65':V65_SHA},
                frozenV65Cutoff=expert['cutoff'],trainedOnV66=False,thresholdsAdjustedOnV66=False,
                v66LabelsUsedForSelection=False,previouslyScoredByV64OrV65=False,
                independentOfPreexistingTrainingSourcesAtExactFingerprintLevel=True,
                semanticOverlapNotRuledOut=True,
                sourceLabelCaveats=old['restrictions'],
                resultByGroup=metrics,
                extremeLowFpGoalValidated=False,customerTrafficGoalValidated=False,
                note='v66 data publishers supply labels; Turkish-stemmed mail lacks headers and has unclear license. Community phishing samples are altered and spam-only. 489 HAM are inadequate for 0.01% FPR claim. Repeated future evaluation on v66 is retrospective.')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    lines=['# MailGuard v66 frozen first-use test','',
           '**No model fitting or threshold calibration on v66.**',
           'The 1307 screened records were frozen in a different workflow job before downloading models.',
           'Data-quality limitations: '+old['restrictions'],'',
           '| Group | Model | N | Spam | Recall | FP/HAM | 95% FPR upper |',
           '|---|---|---:|---:|---:|---:|---:|']
    for group in ('all','source/Turkish-stemmed','source/CW-L-phishing','script/latin','script/other','script/cjk'):
        if group not in metrics:continue
        for name,m in metrics[group].items():
            recall=f"{m['recall']:.2%}" if m['recall'] is not None else 'n/a'
            upper=f"{m['fprWilson95Upper']:.2%}" if m['fprWilson95Upper'] is not None else 'n/a'
            lines.append(f"| {group} | {name} | {m['n']} | {m['spam']} | {recall} | {m['fp']} / {m['ham']} | {upper} |")
    lines+=['','These 1307 messages are **not** a random sample of mail and do **not** prove FPR of 0.01%, any-100k generalization, or customer performance.',
            'Even zero FP among 489 legitimate messages would give a wide FPR uncertainty interval.',
            'Saved predictions only as aggregate metrics; no raw message text, identifiers or scores published.',
            '']
    md='\n'.join(lines)
    (out/'result.md').write_text(md)
    print(md,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['freeze','score'],required=True)
    p.add_argument('--prior',default='prior/blocked.sqlite')
    p.add_argument('--old-input',default='previous')
    p.add_argument('--output',required=True)
    p.add_argument('--manifest',default='frozen/manifest.json')
    p.add_argument('--model64',default='models/v64')
    p.add_argument('--model65',default='models/v65')
    args=p.parse_args()
    if args.phase=='freeze':freeze(args)
    else:score(args)
if __name__=='__main__':main()
