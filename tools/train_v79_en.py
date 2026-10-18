"""Train fresh experts; lock policies on SELECT/CAL; score ten frozen blocks."""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import v77_mail as core
import train_v78_en as train
import v79_mail as runtime


def read_rows(path):
    with gzip.open(path,'rt') as stream:
        return [json.loads(line) for line in stream]


def group_cutoff(values, rows, budget):
    truth = np.asarray([r['y'] for r in rows])
    threshold = train.cutoff(values,truth,budget)
    for source in sorted({r['source'] for r in rows}):
        ix = np.array([i for i,r in enumerate(rows) if r['source']==source],dtype=int)
        if np.sum(truth[ix]==0) >= 500:
            threshold=max(threshold,train.cutoff(np.asarray(values)[ix],truth[ix],budget))
    return threshold


def choose(scores, rows, budget):
    truth = np.asarray([r['y'] for r in rows]);best=None
    for name,values in scores.items():
        threshold=group_cutoff(values,rows,budget)
        metric=core.measure(values,truth,threshold)
        source_recalls=[]
        for source in sorted({r['source'] for r in rows}):
            ix=np.array([i for i,r in enumerate(rows) if r['source']==source and r['y']==1],dtype=int)
            if len(ix)>=25:source_recalls.append(float(np.mean(values[ix]>=threshold)))
        key=(min(source_recalls),metric['recall'],-metric['fp'])
        if best is None or key>best[0]:best=(key,{'candidate':name,'select_cutoff':threshold,'selection':metric,
            'worst_select_source_recall':key[0],'calibration_fpr_budget':budget})
    return best[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='.cache/v79/frozen')
    parser.add_argument('--out',default='reports/v79')
    args=parser.parse_args();data=Path(args.data);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((data/'frozen.json').read_text())
    if manifest['test_count']!=100000 or not manifest['merged_unknown_origin_messages_development_only']:
        raise ValueError('Unexpected frozen test scope')
    if any('Merged-public' in block['sources'] for block in manifest['blocks']):
        raise ValueError('Unknown-provenance messages cannot enter the email TEST')
    parts={name:read_rows(data/(name+'.jsonl.gz')) for name in ('train','select','cal')}
    from v79_data import fingerprint, LID_SHA
    for name, rows in parts.items():
        if len(rows)!=manifest['development_counts'][name] or fingerprint(rows)!=manifest['development_fingerprints'][name]:
            raise ValueError('Frozen development partition drift: '+name)
    if hashlib.sha256((data/'lid.176.ftz').read_bytes()).hexdigest()!=LID_SHA:
        raise ValueError('Language identifier drift')
    bundle={'version':runtime.VERSION,'experts':{},'policies':{},'manifest':manifest,
            'language_model_bytes':(data/'lid.176.ftz').read_bytes()}
    (out/'frozen.json').write_text(json.dumps(manifest,indent=2))
    for kind in ('word','char','tfidf'):
        bundle['experts'][kind]=train.fit_expert(parts['train'],kind)
        joblib.dump(bundle['experts'][kind],out/(kind+'.joblib'),compress=3)
    select=runtime.candidates(bundle,[r['text'] for r in parts['select']])
    cal=runtime.candidates(bundle,[r['text'] for r in parts['cal']])
    truth=[r['y'] for r in parts['cal']]
    for profile,budget in runtime.PROFILES.items():
        cfg=choose(select,parts['select'],budget)
        cfg['cutoff']=max(cfg['select_cutoff'],group_cutoff(cal[cfg['candidate']],parts['cal'],budget))
        cfg['calibration']=core.measure(cal[cfg['candidate']],truth,cfg['cutoff'])
        bundle['policies'][profile]=cfg
    model=out/'model.joblib';joblib.dump(bundle,model,compress=3)
    sha=hashlib.sha256(model.read_bytes()).hexdigest()
    print('MODEL LOCKED BEFORE TEST',sha,json.dumps(bundle['policies']),flush=True)
    del parts,select,cal
    results=[];all_values={p:([],[]) for p in runtime.PROFILES};source_values={}
    with gzip.open(data/'test.jsonl.gz','rt') as stream:
        for number in range(1,11):
            rows=[]
            for _ in range(10000):
                line=stream.readline()
                if not line:raise ValueError('Incomplete test block')
                rows.append(json.loads(line))
            if fingerprint(rows)!=manifest['blocks'][number-1]['fingerprint']:raise ValueError('Test block drift')
            scores=runtime.candidates(bundle,[r['text'] for r in rows]);labels=[r['y'] for r in rows]
            metrics={}
            for profile,cfg in bundle['policies'].items():
                margins=scores[cfg['candidate']]-cfg['cutoff']
                metrics[profile]=core.measure(margins,labels,0)
                all_values[profile][0].extend(margins);all_values[profile][1].extend(labels)
                for i,row in enumerate(rows):
                    pair=source_values.setdefault(row['source'],{}).setdefault(profile,([],[]))
                    pair[0].append(float(margins[i]));pair[1].append(row['y'])
            results.append({'block':number,'metrics':metrics})
            print('BLOCK',number,json.dumps(metrics),flush=True)
        if stream.readline():raise ValueError('More than 100000 test emails')
    if sha!=hashlib.sha256(model.read_bytes()).hexdigest():raise ValueError('Model changed during TEST')
    total={p:core.measure(*values,0) for p,values in all_values.items()}
    report={'version':runtime.VERSION,'model_sha256':sha,'freeze':manifest,'policies':bundle['policies'],
        'blocks':results,'total':total,
        'by_source':{source:{p:core.measure(*values,0) for p,values in profiles.items()} for source,profiles in source_values.items()},
        'minimum_block_recall':{p:min(b['metrics'][p]['recall'] for b in results) for p in runtime.PROFILES},
        'every_fixed_block_recall_at_least80':{p:all(b['metrics'][p]['recall']>=.8 for b in results) for p in runtime.PROFILES},
        'arbitrary_10000_emails_guaranteed':False,'test_selected_using_predictions':False,
        'warning':'Ten predeclared mixed-source historical email blocks, not a guarantee for every arbitrary source or sequence.'}
    (out/'result.json').write_text(json.dumps(report,indent=2))
    print('FINAL',json.dumps({'total':total,'minimum_block_recall':report['minimum_block_recall'],
                           'all_pass':report['every_fixed_block_recall_at_least80']}),flush=True)


if __name__=='__main__':main()
