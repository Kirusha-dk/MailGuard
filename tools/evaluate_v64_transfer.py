"""First-use transfer comparison of two locked models on exactly the same mail."""
import argparse
import csv
import gzip
import json
import math
import os
import time
from collections import Counter,defaultdict
from itertools import islice
from pathlib import Path
from evaluate_v62_500k import file_sha,rows,measure
from evaluate_v64_frozen import model_root

SHA64='ecd8bb221594538698188fd3270daab840a0a43f672c1fcc6af272bc380190e5'
SHA62='380991f74ae7ddc4d26cb36fd952ff7a1b6a24a1c8985becf28cb14d2f493aaf'
PROFILES=('conservative','high-recall')
COLUMNS={'v62':('v62ConservativeMargin','v62HighRecallMargin'),
         'v64':('v64ConservativeMargin','v64HighRecallMargin')}

def shared_scores(bundle,batch):
    import numpy as np
    from v64_runtime import BRANCHES,character_evidence,word_features,transform,structure
    from v61_semantic import script
    artifact,parent=bundle
    chars=character_evidence(parent,batch)
    columns={(scope,name):i for i,(scope,name) in enumerate((s,n) for s in sorted(parent['cores']) for n in sorted(parent['cores'][s]))}
    languages=[script(r) for r in batch]
    result={}
    for p,cfg in parent['profiles'].items():
        result['v62/'+p]=np.asarray([chars[i,columns[cfg['scripts'][lang]['scope'],cfg['scripts'][lang]['model']]]-cfg['scripts'][lang]['cutoff']
                                    for i,lang in enumerate(languages)],dtype=np.float64)
    margins=[chars]
    for branch in BRANCHES:
        x=transform(word_features(batch,branch),artifact['idf'][branch],artifact['ratio'][branch])
        margins.append(np.column_stack([artifact['cores'][branch][k].decision_function(x) for k in sorted(artifact['cores'][branch])]))
    margins=np.column_stack(margins);margins=np.sign(margins)*np.log1p(np.abs(margins))
    x=np.round(np.column_stack([margins,structure(batch)]),6)
    if not np.isfinite(x).all():raise ValueError('Nonfinite evidence')
    for p,cfg in artifact['profiles'].items():
        result['v64/'+p]=artifact['gates'][cfg['selected']].predict_proba(x)[:,1]-np.asarray([cfg['scriptCutoffs'][s] for s in languages])
    return result

def add_record(tally,changes,r):
    y=int(r['label'])
    if y not in (0,1):raise ValueError('Invalid source label')
    groups=('all',r['cohort'],'source/'+r['source'],'script/'+r['script'],r['cohort']+'/script/'+r['script'])
    for version,cols in COLUMNS.items():
        for p,column in zip(PROFILES,cols):
            margin=float(r[column])
            if not math.isfinite(margin):raise ValueError('Nonfinite decision')
            positive=margin>=0;key=('tp' if positive else 'fn') if y else ('fp' if positive else 'tn')
            for g in groups:tally[g,version,p][key]+=1
    for p,old,new in zip(PROFILES,COLUMNS['v62'],COLUMNS['v64']):
        a,b=float(r[old])>=0,float(r[new])>=0
        key=('fixedFN' if b else 'newFN') if y else ('newFP' if b else 'fixedFP')
        if a!=b:
            changes['all',p][key]+=1;changes[r['cohort'],p][key]+=1

def score(args):
    import numpy as np
    from v64_runtime import load_bundle,score_rows as score64
    from v62_runtime import score_rows as score62
    from v61_semantic import script
    root=model_root(args.model);audit=json.loads(Path(args.audit).read_text())
    if audit['count']!=500000 or not audit['zeroDetectedPriorOverlap'] or audit['testPreviouslyScored']:
        raise ValueError('Not the frozen first-use test')
    if file_sha(args.input)!=audit['shardSha256'][str(args.shard)]:raise ValueError('Frozen new input changed')
    if file_sha(root/'model.joblib')!=SHA64 or file_sha(root/'v62/model.joblib')!=SHA62:raise ValueError('Model changed before TEST')
    bundle=load_bundle(root);artifact,parent=bundle
    if file_sha('tools/v64_runtime.py')!=artifact['runtimeSha256']:raise ValueError('v64 runtime changed')
    config={'v62':parent['profiles'],'v64':artifact['profiles']}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    probe=list(islice(rows(args.input),1000))
    shared=shared_scores(bundle,probe)
    standard64=score64(bundle,probe,batch_size=251)
    for p in PROFILES:
        for v,standard in (('v64',standard64[p]),('v62',score62(parent,probe,profile=p,batch_size=251))):
            candidate=shared[v+'/'+p]
            if not np.allclose(candidate,standard,rtol=1e-6,atol=1e-7) or np.any((candidate>=0)!=(standard>=0)):
                raise ValueError('Shared first 1000 differs from frozen '+v+' runtime')
    print('FIRST USE 1000 FROZEN RUNTIME PARITY PASSED',args.shard,flush=True)
    counts=Counter();tally=defaultdict(Counter);changes=defaultdict(Counter);seen=set()
    header=['identity','family','source','script','label','cohort',*COLUMNS['v62'],*COLUMNS['v64']]
    output=out/f'predictions-{args.shard}.csv.gz';start=time.perf_counter()
    with gzip.open(output,'wt',newline='') as f:
        writer=csv.writer(f);writer.writerow(header);iterator=rows(args.input)
        while batch:=list(islice(iterator,128)):
            scores=shared_scores(bundle,batch)
            for i,r in enumerate(batch):
                if r['identity'] in seen or r.get('previouslyScored') is not False:raise ValueError('Duplicate or reused row')
                seen.add(r['identity']);lang=script(r)
                if lang!=r['script']:raise ValueError('Input script fingerprint changed')
                cohort='different-year-same-provider' if r['source'].startswith('SpamArchive-') else 'new-mail-known-domain' if r['source']=='Published-ASCII-mail' else 'unseen-source'
                values=[r['identity'],r['family'],r['source'],lang,r['y'],cohort,
                        *[float(scores[v+'/'+p][i]) for v in COLUMNS for p in PROFILES]]
                writer.writerow(values);add_record(tally,changes,dict(zip(header,values)));counts['count']+=1
                counts['ham' if r['y']==0 else 'spam']+=1
            if counts['count']%8192==0:print('NEW TRANSFER SCORE',args.shard,counts['count'],flush=True)
    if counts['count']!=100000 or file_sha(root/'model.joblib')!=SHA64 or file_sha(root/'v62/model.joblib')!=SHA62:
        raise ValueError('Incomplete shard or changed model')
    metadata=dict(shard=args.shard,counts=dict(counts),modelsSha256={'v62':SHA62,'v64':SHA64},profiles=config,
                  predictionsSha256=file_sha(output),inputSha256=file_sha(args.input),auditSha256=file_sha(args.audit),
                  runtimeSha256=artifact['runtimeSha256'],first1000ParityPassed=True,
                  probeByScript=dict(Counter(script(r) for r in probe)),probeBySource=dict(Counter(r['source'] for r in probe)),
                  frozenModelCodeCommit=artifact['codeCommit'],evaluationCodeCommit=os.environ.get('GITHUB_SHA'),
                  evaluationRunId=os.environ.get('GITHUB_RUN_ID'),inferenceAndCsvSeconds=time.perf_counter()-start)
    (out/f'shard-{args.shard}.json').write_text(json.dumps(metadata,indent=2))
    print('NEW TRANSFER SHARD COMPLETE',args.shard,json.dumps(metadata['counts']),flush=True)

def aggregate(args):
    audit=json.loads(Path(args.audit).read_text());inp,out=Path(args.input),Path(args.output)
    out.mkdir(parents=True,exist_ok=False);tally=defaultdict(Counter);changes=defaultdict(Counter)
    shards=[];identities=set();families=set();config=None
    for shard in range(5):
        ms=list(inp.rglob(f'shard-{shard}.json'));fs=list(inp.rglob(f'predictions-{shard}.csv.gz'))
        if len(ms)!=1 or len(fs)!=1:raise ValueError('Missing or duplicate shard')
        m=json.loads(ms[0].read_text())
        if m['modelsSha256']!={'v62':SHA62,'v64':SHA64} or m['auditSha256']!=file_sha(args.audit) or m['inputSha256']!=audit['shardSha256'][str(shard)] or not m['first1000ParityPassed']:
            raise ValueError('Wrong model, inputs or unvalidated inference')
        if config is None:config=m['profiles']
        if m['profiles']!=config or m['predictionsSha256']!=file_sha(fs[0]):raise ValueError('Threshold or predictions changed')
        counts=Counter()
        with gzip.open(fs[0],'rt',newline='') as f:
            for r in csv.DictReader(f):
                if r['identity'] in identities or r['family'] in families:raise ValueError('Global duplicate identity or family')
                identities.add(r['identity']);families.add(r['family']);add_record(tally,changes,r)
                counts['count']+=1;counts['ham' if r['label']=='0' else 'spam']+=1
        if dict(counts)!=m['counts'] or counts['count']!=100000:raise ValueError('Incomplete shard')
        shards.append(m)
    results={}
    for (g,v,p),c in sorted(tally.items()):results.setdefault(g,{}).setdefault(v,{})[p]=measure(c)
    if len(identities)!=500000 or results['all']['v64']['conservative']['ham']!=audit['ham'] or results['all']['v64']['conservative']['spam']!=audit['spam']:
        raise ValueError('Frozen sample size changed')
    goals={}
    for v in COLUMNS:
        goals[v]={}
        for p in PROFILES:
            m=results['all'][v][p]
            source_recalls=[z[v][p]['recall'] for g,z in results.items() if g.startswith('source/') and z[v][p]['spam']>=50]
            script_recalls=[z[v][p]['recall'] for g,z in results.items() if g.startswith('script/') and z[v][p]['spam']>=100]
            goals[v][p]=dict(observedRecallAndFprGoalMet=m['recall']>=.95 and m['hamFpr']<=.0001,
                            minimumSourceRecall=min(source_recalls),minimumScriptRecall=min(script_recalls),
                            broadTransferGoalMet=m['recall']>=.95 and m['hamFpr']<=.0001 and min(source_recalls+script_recalls)>=.95,
                            rareFp95UpperBoundSupportsTarget=m['hamFprWilson95Upper']<=.0001,customerGoalValidated=False)
    paired={}
    for (g,p),c in sorted(changes.items()):paired.setdefault(g,{})[p]=dict(c)
    report=dict(version=64,benchmark='first-use-public-transfer',modelsSha256={'v62':SHA62,'v64':SHA64},audit=audit,
        profiles=config,results=results,pairedChanges=paired,goals=goals,shards=shards,
        modelAndThresholdChanged=False,testLabelsUsedForTraining=False,
        firstUseRelativeToPriorBenchmarks=True,productionGoalValidated=False,
        evaluationCodeCommit=os.environ.get('GITHUB_SHA'),githubRunId=os.environ.get('GITHUB_RUN_ID'),
        caveat=audit['caveat'])
    (out/'report.json').write_text(json.dumps(report,indent=2))
    md='# MailGuard v64: first-use public transfer test\n\n'+audit['caveat']+'\n\n'
    md+=f"Exactly {audit['count']} new selected messages: {audit['spam']} SPAM, {audit['ham']} HAM. Both models and thresholds were fixed before collection. The same messages are evaluated by both versions.\n\n"
    md+='| Cohort | Version | Profile | Count | Spam recall | FP / HAM | FPR | Wilson 95% FPR upper |\n|---|---|---|---:|---:|---:|---:|---:|\n'
    for g in ('all','unseen-source','new-mail-known-domain','different-year-same-provider'):
        if g not in results:continue
        for v,z in results[g].items():
            for p,m in z.items():
                recall='n/a' if m['recall'] is None else f'{m["recall"]:.4%}'
                fpr='n/a' if m['hamFpr'] is None else f'{m["hamFpr"]:.4%}'
                upper='n/a' if m['hamFprWilson95Upper'] is None else f'{m["hamFprWilson95Upper"]:.4%}'
                md+=f'| {g} | {v} | {p} | {m["count"]} | {recall} | {m["fp"]} / {m["ham"]} | {fpr} | {upper} |\n'
    md+='\n## Source and script transfer\n\n| Group | Version | Profile | Spam | Recall | FP / HAM |\n|---|---|---|---:|---:|---:|\n'
    for g,z in results.items():
        if not (g.startswith('source/') or g.startswith('script/')):continue
        for v,profiles in z.items():
            for p,m in profiles.items():
                recall='n/a' if m['recall'] is None else f'{m["recall"]:.4%}'
                md+=f'| {g} | {v} | {p} | {m["spam"]} | {recall} | {m["fp"]} / {m["ham"]} |\n'
    md+='\nHAM here is newly selected after exclusion, not the old 75271-message replay. Source labels are preserved. A high aggregate score on spam-heavy archives cannot by itself establish transfer to customer traffic.\n'
    (out/'report.md').write_text(md);print(md,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('score','aggregate'),required=True)
    p.add_argument('--model',default='.cache/v64');p.add_argument('--shard',type=int,choices=range(5))
    p.add_argument('--input',required=True);p.add_argument('--audit',default='prepared/audit.json');p.add_argument('--output',required=True)
    args=p.parse_args();(score if args.phase=='score' else aggregate)(args)

if __name__=='__main__':main()
