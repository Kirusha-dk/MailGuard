"""Frozen-model result on the ready new labelled-source cohort, not a 500k result."""
import csv,gzip,json,os,time
from collections import Counter,defaultdict
from itertools import islice
from pathlib import Path
import numpy as np
from evaluate_v62_500k import file_sha,rows,measure
from evaluate_v64_frozen import model_root
from evaluate_v64_transfer import SHA64,SHA62,PROFILES,COLUMNS,shared_scores,add_record
from v64_runtime import load_bundle,score_rows as score64
from v62_runtime import score_rows as score62

def main():
    inp=Path('new-cohort');audit=json.loads((inp/'audit.json').read_text());path=inp/'candidates.jsonl.gz'
    if audit['source']!='independent' or file_sha(path)!=audit['candidatesSha256']:raise ValueError('Wrong new cohort')
    root=model_root('.cache/v64')
    if file_sha(root/'model.joblib')!=SHA64 or file_sha(root/'v62/model.joblib')!=SHA62:raise ValueError('Changed model')
    bundle=load_bundle(root);artifact,parent=bundle
    if file_sha('tools/v64_runtime.py')!=artifact['runtimeSha256']:raise ValueError('Changed v64 runtime')
    first=list(islice(rows(path),1000));shared=shared_scores(bundle,first);standard=score64(bundle,first,batch_size=251)
    for p in PROFILES:
        for v,reference in (('v64',standard[p]),('v62',score62(parent,first,profile=p,batch_size=251))):
            values=shared[v+'/'+p]
            if not np.allclose(values,reference,rtol=1e-6,atol=1e-7) or np.any((values>=0)!=(reference>=0)):
                raise ValueError('First 1000 differs from frozen runtime')
    out=Path('reports/v64-transfer-preview');out.mkdir(parents=True,exist_ok=False)
    tally=defaultdict(Counter);changes=defaultdict(Counter);identities=set();families=set();classes=Counter();n=0
    header=['identity','family','source','script','label','cohort',*COLUMNS['v62'],*COLUMNS['v64']]
    start=time.perf_counter()
    with gzip.open(out/'predictions.csv.gz','wt',newline='') as f:
        writer=csv.writer(f);writer.writerow(header);iterator=rows(path)
        while batch:=list(islice(iterator,128)):
            values=shared_scores(bundle,batch)
            for i,r in enumerate(batch):
                if r['identity'] in identities or r['family'] in families:raise ValueError('Duplicate new mail')
                identities.add(r['identity']);families.add(r['family']);classes[r['y']]+=1;n+=1
                cohort='new-mail-known-domain' if r['source']=='Published-ASCII-mail' else 'unseen-source'
                record=[r['identity'],r['family'],r['source'],r['script'],r['y'],cohort,
                        *[float(values[v+'/'+p][i]) for v in COLUMNS for p in PROFILES]]
                writer.writerow(record);add_record(tally,changes,dict(zip(header,record)))
            if n%4096==0:print('NEW LABELLED SOURCE PREVIEW',n,'/',audit['count'],flush=True)
    if n!=audit['count'] or file_sha(root/'model.joblib')!=SHA64 or file_sha(root/'v62/model.joblib')!=SHA62:raise ValueError('Count or model changed')
    results={}
    for (g,v,p),c in sorted(tally.items()):results.setdefault(g,{}).setdefault(v,{})[p]=measure(c)
    report=dict(benchmark='first-use-labelled-source-preview',count=n,ham=classes[0],spam=classes[1],
        modelsSha256={'v62':SHA62,'v64':SHA64},results=results,audit=audit,
        first1000FrozenRuntimeParityPassed=True,modelAndThresholdChanged=False,testLabelsUsedForTraining=False,
        notA500kResult=True,productionGoalValidated=False,seconds=time.perf_counter()-start,
        codeCommit=os.environ.get('GITHUB_SHA'),githubRunId=os.environ.get('GITHUB_RUN_ID'),
        caveat='Previously unused labelled public-source cohort after prior-source exclusions. GenSpam is anonymized; published ASCII mail is from a known Enron-like domain with removed headers; Nazario is one manually classified inbox. This cohort will also be included in the pending larger comparison with the same frozen models; it is not a second independent test. Exclusions include short bodies and duplicates. Source labels are preserved; results do not establish customer quality.')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    md='# MailGuard: ready new labelled-source cohort\n\n'+report['caveat']+f'\n\nActual count: {n}; SPAM: {classes[1]}; HAM: {classes[0]}. This is not the 500000-message result.\n\n'
    md+='| Group | Version | Profile | Spam recall | FP / HAM | FPR | Wilson 95% upper |\n|---|---|---|---:|---:|---:|---:|\n'
    for g,z in results.items():
        if not (g in ('all','unseen-source','new-mail-known-domain') or g.startswith('source/')):continue
        for v,profiles in z.items():
            for p,m in profiles.items():
                recall='n/a' if m['recall'] is None else f'{m["recall"]:.4%}'
                rate='n/a' if m['hamFpr'] is None else f'{m["hamFpr"]:.4%}'
                upper='n/a' if m['hamFprWilson95Upper'] is None else f'{m["hamFprWilson95Upper"]:.4%}'
                md+=f'| {g} | {v} | {p} | {recall} | {m["fp"]} / {m["ham"]} | {rate} | {upper} |\n'
    (out/'report.md').write_text(md);print(md,flush=True)

if __name__=='__main__':main()
