#!/usr/bin/env python3
"""Keep v54 votes; allow zero additional FP on untouched calibration HAM."""
import argparse
import csv
import gc
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
import torch
from benchmark_v48_calibrated import threshold_for_budget
from rspamd_benchmark_audit import load_frozen_v40
from train_v47_phishing_aware import from_eml
from train_v55_multilingual import corpus_rows, file_sha, CORPUS_SHA
from train_v56_consensus import development, candidate_score, gate_features, predict_margin_union, is_new, WARNING
from v57_runtime import load_bundle
from v58_runtime import residual_threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=1000)
    p.add_argument('--output', default='reports/v58-smoke')
    p.add_argument('--source', default='.cache/v57-artifact/v57-full')
    p.add_argument('--v55-root', default='.cache/v55-artifact/v55-full')
    p.add_argument('--corpus', default='.cache/v54-corpus/selected.jsonl.gz')
    p.add_argument('--baseline-predictions', default='.cache/v54-evidence/predictions.csv')
    args = p.parse_args()
    if args.limit < 0 or args.limit == 1:
        p.error('Limit must be zero or at least two')
    torch.set_num_threads(4)
    source, root, out = Path(args.source), Path(args.v55_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    prior = json.loads((source/'report.json').read_text())
    if prior['limit'] != 0 or file_sha(source/'model.joblib') != prior['modelSha256']:
        raise ValueError('Expected the completed frozen v57')
    artifact, old, neural, new = load_bundle(source)
    if file_sha(args.corpus) != CORPUS_SHA:
        raise ValueError('Frozen corpus hash mismatch')
    if file_sha(root/'model.joblib') != artifact['components']['v55/model.joblib']:
        raise ValueError('v55 source mismatch')
    with (root/'split-manifest.csv').open() as f: manifest = list(csv.DictReader(f))
    with Path(args.baseline_predictions).open() as f:
        baseline = {r['identity']:(int(r['label']),float(r['score'])) for r in csv.DictReader(f)}
    frozen, audit = load_frozen_v40('.cache/v40-fresh50k')
    # Reconstruct original development groups. Selection scores are not used to choose anything in v58.
    dev = development(args, manifest, baseline, frozen, old, neural, new)
    cal = dev['cal']; base = cal['old'] >= old['threshold']; residual = ~base
    cal_families={r['family'] for r in cal['rows']}
    test_families={r['family'] for r in manifest if r['split']=='test'}
    if cal_families & test_families:
        raise AssertionError('Calibration families overlap test')
    inherited = int(((cal['y']==0)&base).sum())
    changed, calibration = {}, {}
    for name, setting in artifact['profiles'].items():
        scores = candidate_score(setting['selected'], artifact['gates'], cal['x'], cal['old'], cal['new'], old['threshold'])
        frozen_threshold = setting['threshold']
        setting['threshold'] = residual_threshold(cal['y'], base, scores, frozen_threshold)
        setting['unionCalibrationEvaluated'] = True
        added_fp = int(((cal['y']==0)&residual&(scores>=setting['threshold'])).sum())
        if added_fp: raise AssertionError('Additional calibration FP allowance exceeded')
        changed[name] = setting['threshold'] != frozen_threshold
        calibration[name] = dict(ham=int((cal['y']==0).sum()), inheritedV54Fp=inherited, additionalFp=added_fp,
                                 unionFp=inherited+added_fp, sourceThreshold=frozen_threshold, threshold=setting['threshold'],
                                 thresholdRule='max(frozen source threshold, first boundary excluding every residual calibration ham)')
        print('v58 calibration', name, json.dumps(calibration[name]), flush=True)
    artifact['version']='v58-calibrated-protected-consensus'
    artifact['residualCalibration']=calibration
    artifact['sourceV57Run']=38172763308
    for name,digest in artifact['components'].items():
        dest=out/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source/name,dest)
        if file_sha(dest)!=digest: raise AssertionError('Component changed')
    joblib.dump(artifact,out/'model.joblib',compress=3)
    frozen_model_hash=file_sha(out/'model.joblib')
    del dev,cal
    gc.collect()
    if not any(changed.values()) and not args.limit:
        # Exact same components, gates, thresholds and protection rule imply identical decisions.
        results=prior['results']; n=prior['testCount']
        shutil.copyfile(source/'predictions.csv',out/'predictions.csv')
        print('All thresholds unchanged: reuse exact v57 test decisions.',flush=True)
    else:
        test_manifest = [r for r in manifest if r['split']=='test']
        if args.limit:
            test_manifest=[r for group in (False,True) for r in sorted((x for x in test_manifest if is_new(x['source'])==group),key=lambda x:x['identity'])[:args.limit//2]]
        wanted={r['identity'] for r in test_manifest}
        with (root/'predictions.csv').open() as f:
            cached_new={r['identity']:(int(r['label']),float(r['score'])) for r in csv.DictReader(f) if r['identity'] in wanted}
        texts={}
        for raw in frozen['test']:
            parsed=from_eml(raw['path'].read_bytes())
            if parsed['identity'] in wanted: texts.setdefault(parsed['identity'],parsed['ngram_text'])
        del frozen
        gc.collect()
        counts={name:{g:dict(tp=0,fn=0,fp=0,tn=0) for g in ('new','legacy','all')} for name in artifact['profiles']}
        seen=set()
        with (out/'predictions.csv').open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(['identity','family','source','label','v54Spam']+[f'{name}Spam' for name in counts])
            def consume(rows):
                op=np.zeros(len(rows)); npred=np.zeros(len(rows)); ix=[]
                for i,r in enumerate(rows):
                    label,npred[i]=cached_new[r['identity']]
                    if label!=r['y']: raise AssertionError('v55 label mismatch')
                    if is_new(r['source']):
                        label,op[i]=baseline[r['identity']]
                        if label!=r['y']: raise AssertionError('v54 label mismatch')
                    else: ix.append(i)
                for start in range(0,len(ix),128):
                    take=ix[start:start+128]
                    op[take]=predict_margin_union(old,neural,[texts[rows[i]['identity']] for i in take])
                features=gate_features(rows,op,npred,old['threshold']); old_votes=op>=old['threshold']
                decisions={name:old_votes|(candidate_score(s['selected'],artifact['gates'],features,op,npred,old['threshold'])>=s['threshold'])
                           for name,s in artifact['profiles'].items()}
                for i,r in enumerate(rows):
                    if r['identity'] in seen: raise AssertionError('Duplicate test identity')
                    seen.add(r['identity'])
                    for name,pred in decisions.items():
                        vote=bool(pred[i]); label=r['y']; key=('tp' if vote else 'fn') if label else ('fp' if vote else 'tn')
                        counts[name]['new' if is_new(r['source']) else 'legacy'][key]+=1; counts[name]['all'][key]+=1
                    writer.writerow([r['identity'],r['family'],r['source'],r['y'],int(old_votes[i])]+[int(decisions[name][i]) for name in counts])
            batch=[]
            for r in corpus_rows(root/'test.jsonl.gz'):
                if r['identity'] not in wanted: continue
                batch.append(r)
                if len(batch)==1000:consume(batch);batch=[]
            if batch:consume(batch)
        if seen!=wanted: raise AssertionError('Test inventory mismatch')
        n=len(seen)
        results={name:{g:dict(c,recall=c['tp']/max(1,c['tp']+c['fn']),fpr=c['fp']/max(1,c['fp']+c['tn'])) for g,c in groups.items()} for name,groups in counts.items()}
    if file_sha(out/'model.joblib') != frozen_model_hash:
        raise AssertionError('Model changed after calibration')
    report=dict(version=58,limit=args.limit,testCount=n,results=results,residualCalibration=calibration,thresholdsChanged=changed,
                protectedV54Decisions=True,modelSha256=file_sha(out/'model.joblib'),estimatorRefit=False,
                testLabelsUsedForEstimatorOrThresholdFit=False,testPreviouslyInspected=True,productionGoalValidated=False,
                sourceV57Run=38172763308,reusedExactV57TestDecisions=(not any(changed.values()) and not args.limit),
                calibrationManifestSha256=file_sha(root/'split-manifest.csv'),audit=audit,
                warning=WARNING+' Zero additional calibration FP is an empirical allowance, not an out-of-sample guarantee. v54 FP are inherited.')
    report['engineeringGoals']={name:dict(recall95AndFp50=(m['all']['recall']>=.95 and m['all']['fp']<=50 and not args.limit),
                                         bothSourceGroupsRecall95=(m['new']['recall']>=.95 and m['legacy']['recall']>=.95 and not args.limit)) for name,m in results.items()}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    md='# MailGuard v58 residual calibration\n\n'+report['warning']+'\n\n| Profile | Group | N | Ham | Spam recall | FP | FN |\n|---|---|---:|---:|---:|---:|---:|\n'
    for name,groups in results.items():
        for g,m in groups.items():
            total=sum(m[k] for k in ('tp','fn','fp','tn'));md+=f"| {name} | {g} | {total} | {m['fp']+m['tn']} | {m['recall']:.2%} | {m['fp']} | {m['fn']} |\n"
    (out/'report.md').write_text(md);print(md,flush=True)


if __name__=='__main__':main()
