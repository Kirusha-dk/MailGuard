"""Compare locked v64 against v62 on frozen, explicitly reused engineering inputs."""
import argparse
import csv
import gzip
import json
import math
import os
from collections import Counter,defaultdict
from itertools import islice
from pathlib import Path
from evaluate_v62_500k import file_sha,rows,measure

PROFILES = ('conservative','high-recall')
EXPECTED = {'500k':500000,'ham':75271,'may':4146,'summer':11831,'october':122}
CAVEAT = ('All TEST sources have already been inspected: this is a repeated engineering comparison, '
          'not independent customer validation. The full 500k replay includes development messages; '
          'use its no-development-overlap subset for quality. The 75271 HAM set excludes the full '
          'TRAIN/SELECT/CAL partitions used by both versions. Original labels are preserved. '
          'No TEST labels fit, choose or calibrate v64; thresholds remain fixed.')


def model_root(root):
    candidates = [p.parent for p in Path(root).rglob('model.joblib') if (p.parent/'v62/model.joblib').exists()]
    if len(candidates)!=1:raise ValueError('Expected one v64 bundle')
    return candidates[0]


def add_result(tally,r):
    y = int(r['label'])
    if y not in (0,1):raise ValueError('Invalid label')
    for p,column in zip(PROFILES,('conservativeMargin','highRecallMargin')):
        margin = float(r[column])
        if not math.isfinite(margin):raise ValueError('Nonfinite score')
        positive = margin>=0
        key = ('tp' if positive else 'fn') if y else ('fp' if positive else 'tn')
        for group in ('all',r['scope'],r['scope']+'/source/'+r['source'],r['scope']+'/script/'+r['script']):
            tally[r['dataset'],group,p][key]+=1


def score(args):
    import numpy as np
    from v64_runtime import load_bundle,score_rows
    from v61_semantic import script
    root = model_root(args.model)
    training = json.loads((root/'training-report.json').read_text())
    locked = training['modelSha256']
    if file_sha(root/'model.joblib')!=locked or not training['runtimeParityPassed']:
        raise ValueError('Wrong or unvalidated model')
    bundle = load_bundle(root)
    artifact,_ = bundle
    if artifact['runtimeSha256']!=file_sha('tools/v64_runtime.py'):
        raise ValueError('Runtime changed since training')
    audit500 = json.loads(Path('reports/v62-500k-result.json').read_text())['audit']
    ham_root = Path(args.ham)
    ham_audit = json.loads((ham_root/'audit.json').read_text())
    if artifact['splitManifestSha256']!=audit500['developmentManifestSha256'] or artifact['splitManifestSha256']!=ham_audit['developmentManifestSha256']:
        raise ValueError('Old exclusions do not cover the new training partitions')
    if file_sha(args.input)!=audit500['shardSha256'][str(args.shard)] or file_sha(ham_root/'ham.jsonl.gz')!=ham_audit['hamSha256']:
        raise ValueError('Frozen evaluation input changed')
    out = Path(args.output);out.mkdir(parents=True,exist_ok=False)
    counts = Counter();tally = defaultdict(Counter)
    header = ['dataset','identity','family','source','script','label','scope','conservativeMargin','highRecallMargin']
    output = out/f'predictions-{args.shard}.csv.gz'
    with gzip.open(output,'wt',newline='') as f:
        writer = csv.writer(f);writer.writerow(header)
        def evaluate(iterator,dataset):
            while batch := list(islice(iterator,128)):
                values = score_rows(bundle,batch)
                for i,r in enumerate(batch):
                    scope = ('development-overlap' if r.get('overlap') else 'no-development-overlap') if dataset=='500k' else \
                            ('previously-scored' if r['previouslyScored'] else 'additional') if dataset=='ham' else 'engineering-regression'
                    record = [dataset,r['identity'],r['family'],r['source'],script(r),r['y'],scope,
                              float(values['conservative'][i]),float(values['high-recall'][i])]
                    if not all(math.isfinite(v) for v in record[-2:]):raise ValueError('Nonfinite score')
                    writer.writerow(record);add_result(tally,dict(zip(header,record)))
                counts[dataset]+=len(batch)
                if counts[dataset]%8192==0:print('V64 SCORE',args.shard,dataset,counts[dataset],flush=True)
            print('V64 DATASET COMPLETE',args.shard,dataset,counts[dataset],flush=True)
        first = list(islice(rows(args.input),1000))
        a,b = score_rows(bundle,first,batch_size=128),score_rows(bundle,first,batch_size=251)
        for p in PROFILES:
            if not np.allclose(a[p],b[p],rtol=1e-6,atol=1e-7) or np.any((a[p]>=0)!=(b[p]>=0)):
                raise ValueError('First 1000 runtime parity failed')
        print('V64 FIRST 1000 PARITY PASSED',args.shard,flush=True)
        evaluate(rows(args.input),'500k')
        evaluate((r for i,r in enumerate(rows(ham_root/'ham.jsonl.gz')) if i%5==args.shard),'ham')
        if args.shard==0:
            for directory,dataset in (('.cache/v60-data','may'),('.cache/v59-data','summer'),('.cache/v61-new','october')):
                path = Path(directory)/'test.jsonl.gz'
                provenance = json.loads((Path(directory)/'provenance.json').read_text())
                digest = provenance['testSha256'] if dataset=='october' else provenance['splitHashes']['test']
                if file_sha(path)!=digest:raise ValueError('Regression input changed')
                evaluate(rows(path),dataset)
    if counts['500k']!=100000 or counts['ham']!=(15055 if args.shard==0 else 15054) or file_sha(root/'model.joblib')!=locked:
        raise ValueError('Incomplete shard or modified model')
    metadata = dict(shard=args.shard,counts=dict(counts),modelSha256=locked,predictionsSha256=file_sha(output),
                    runtimeSha256=artifact['runtimeSha256'],first1000ParityPassed=True,profiles=artifact['profiles'],
                    modelCodeCommit=artifact['codeCommit'],splitManifestSha256=artifact['splitManifestSha256'],
                    evaluationCodeCommit=os.environ.get('GITHUB_SHA'),evaluationRunId=os.environ.get('GITHUB_RUN_ID'))
    (out/f'shard-{args.shard}.json').write_text(json.dumps(metadata,indent=2))


def aggregate(args):
    root = model_root(args.model)
    training = json.loads((root/'training-report.json').read_text())
    locked = training['modelSha256']
    if file_sha(root/'model.joblib')!=locked:raise ValueError('Model changed')
    inputs,out = Path(args.input),Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    tally,counts = defaultdict(Counter),Counter()
    seen = defaultdict(set);shards = []
    for shard in range(5):
        meta_files = list(inputs.rglob(f'shard-{shard}.json'))
        files = list(inputs.rglob(f'predictions-{shard}.csv.gz'))
        if len(meta_files)!=1 or len(files)!=1:raise ValueError('Missing or repeated shard')
        meta = json.loads(meta_files[0].read_text())
        if meta['modelSha256']!=locked or file_sha(files[0])!=meta['predictionsSha256'] or meta['profiles']!=training['profiles']:
            raise ValueError('Shard integrity or profile mismatch')
        local = Counter()
        with gzip.open(files[0],'rt',newline='') as f:
            for r in csv.DictReader(f):
                name = r['dataset']
                if name not in EXPECTED or r['identity'] in seen[name]:raise ValueError('Unknown dataset or duplicate identity')
                seen[name].add(r['identity']);counts[name]+=1;local[name]+=1;add_result(tally,r)
        if dict(local)!=meta['counts']:raise ValueError('Shard count mismatch')
        shards.append(meta)
    if dict(counts)!=EXPECTED:raise ValueError('Incomplete evaluation: '+str(counts))
    results = {}
    for (dataset,scope,p),c in sorted(tally.items()):
        results.setdefault(dataset,{}).setdefault(scope,{})[p]=measure(c)
    if results['500k']['no-development-overlap']['conservative']['count']!=351485 or results['ham']['all']['conservative']['ham']!=75271:
        raise ValueError('Holdout exclusion protocol changed')
    acceptance = {}
    for p in PROFILES:
        recall = results['500k']['no-development-overlap'][p]['recall']
        ham = results['ham']['all'][p]
        regressions = [results[d]['all'][p]['recall'] for d in ('may','summer','october')]
        primary = recall>=.95 and ham['hamFpr']<=.0001
        acceptance[p] = dict(primary500kAndHamGoalMet=primary,engineeringGoalMet=primary and min(regressions)>=.95,
                             minimumRegressionRecall=min(regressions),customerGoalValidated=False)
    baseline = dict(spam=json.loads(Path('reports/v62-500k-result.json').read_text())['results'],
                    ham=json.loads(Path('reports/v62-large-ham-result.json').read_text())['results'])
    report = dict(version=64,modelSha256=locked,modelCodeCommit=training['codeCommit'],counts=dict(counts),
                  evaluationCodeCommit=os.environ.get('GITHUB_SHA'),evaluationRunId=os.environ.get('GITHUB_RUN_ID'),
                  training=training,results=results,baselineV62=baseline,acceptance=acceptance,shards=shards,
                  thresholdsChanged=False,estimatorAndThresholdLockedBeforeTest=True,testLabelsUsedForTraining=False,
                  allTestsReused=True,productionGoalValidated=False,caveat=CAVEAT)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    md = '# MailGuard v64: frozen 500k and large HAM evaluation\n\n'+CAVEAT+'\n\n| Dataset | Scope | Profile | Count | Recall | FP / HAM |\n|---|---|---|---:|---:|---:|\n'
    for dataset,scope in (('500k','all'),('500k','no-development-overlap'),('ham','all'),('ham','additional'),('may','all'),('summer','all'),('october','all')):
        for p,m in results[dataset][scope].items():
            recall = 'n/a' if m['recall'] is None else f'{m["recall"]:.4%}'
            md += f'| {dataset} | {scope} | {p} | {m["count"]} | {recall} | {m["fp"]} / {m["ham"]} |\n'
    md += '\n## v62 / v64 on identical main inputs\n\n| Version | Profile | 500k holdout recall | FP / 75271 HAM |\n|---|---|---:|---:|\n'
    for p in PROFILES:
        md += f'| v62 | {p} | {baseline["spam"]["no-development-overlap"][p]["recall"]:.4%} | {baseline["ham"]["all"][p]["fp"]} |\n'
        md += f'| v64 | {p} | {results["500k"]["no-development-overlap"][p]["recall"]:.4%} | {results["ham"]["all"][p]["fp"]} |\n'
    (out/'report.md').write_text(md);print(md,flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase',choices=['score','aggregate'],required=True)
    p.add_argument('--model',default='.cache/v64')
    p.add_argument('--input',required=True)
    p.add_argument('--ham',default='prepared/ham')
    p.add_argument('--shard',type=int,choices=range(5))
    p.add_argument('--output',required=True)
    args = p.parse_args()
    (score if args.phase=='score' else aggregate)(args)


if __name__=='__main__':main()
