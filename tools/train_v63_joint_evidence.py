"""Joint gate fitted on disjoint SELECT-fit, selected on SELECT-choose, then CAL."""
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from train_v55_multilingual import corpus_rows,file_sha
from train_v60_new_mail import restore_validation,legacy_rows
from train_v61_semantic import cuts,rank,measure,PROFILES
from v61_runtime import load_bundle as load61,margins
from v61_semantic import group
from v63_neighbours import neighbours
from v63_portable_encoder import prepare_encoder
from v62_runtime import load_bundle as load62
from v63_runtime import character_evidence,joint_features,load_bundle,predict_texts,train_only_evidence


def main():
    torch.set_num_threads(4)
    out=Path('reports/v63-full');out.mkdir(parents=True,exist_ok=False)
    root61=[p.parent for p in Path('.cache/v61-frozen').rglob('model.joblib') if (p.parent/'encoder').exists()]
    root62=list(Path('.cache/v62-frozen').rglob('model.joblib'))
    if len(root61)!=1 or len(root62)!=1:raise ValueError('Expected one bundle of each type')
    expected62='380991f74ae7ddc4d26cb36fd952ff7a1b6a24a1c8985becf28cb14d2f493aaf'
    if file_sha(root62[0])!=expected62:raise ValueError('Wrong frozen character checkpoint')
    # The source run failed only in release creation, after fitting, every TEST,
    # and both Docker checks. Verify those saved outputs rather than assuming
    # that a run-level failure means the estimator/packaging checks failed.
    for p in PROFILES:
        summaries=list(Path('.cache/v62-frozen/checks/output').glob(p+'/summary.json'))
        if len(summaries)!=1:raise ValueError('Missing source runtime packaging check')
        check=json.loads(summaries[0].read_text())
        if check['modelSha256']!=expected62 or check['failed']!=0 or check['scored']!=2 or check['thresholdChanged']:
            raise ValueError('Source runtime packaging check failed')
    prior_report=root62[0].parent/'report.json'
    if prior_report.exists():
        prior=json.loads(prior_report.read_text())
        if prior['modelSha256']!=expected62 or not prior['estimatorAndThresholdLockedBeforeTest'] or prior['testLabelsUsedForTraining']:
            raise ValueError('Source evaluation provenance failed')
        accepted=[p for p,v in prior['engineeringAcceptance'].items() if v['engineeringGoalMet']]
        if accepted:
            note=dict(skipped=True,reason='v62 already meets the engineering acceptance criterion',
                      useVersion=62,acceptedProfiles=accepted,customerGoalValidated=False)
            (out/'skipped.json').write_text(json.dumps(note,indent=2))
            (out/'report.md').write_text('v62 meets the engineering criterion; joint training skipped. Customer validation remains separate.\n')
            print(json.dumps(note),flush=True);return
    if file_sha(root61[0]/'model.joblib')!='07239ee4a2de4824bbf41df17197151ae8d425e7ea0d73106d5814b37f148116':raise ValueError('Wrong semantic checkpoint')
    bundle61=load61(root61[0]);semantic,_,_=bundle61
    encoder=prepare_encoder(root61[0]/'encoder',out/'encoder')
    char=load62(root62[0].parent)
    data=Path('.cache/v60-data');provenance=json.loads((data/'provenance.json').read_text())
    for s in ('select','cal'):
        if file_sha(data/f'{s}.jsonl.gz')!=provenance['splitHashes'][s]:raise ValueError('Development split changed')
    if char['splitManifestSha256']!=provenance['manifestSha256'] or semantic['splitManifestSha256']!=provenance['manifestSha256']:
        raise ValueError('Components trained on different frozen splits')
    selection=list(corpus_rows(data/'select.jsonl.gz'));cal=list(corpus_rows(data/'cal.jsonl.gz'))
    restore_validation(selection+cal)
    if any(r.get('weakLabel') for r in selection+cal):raise ValueError('Weak labels outside TRAIN')
    from hashlib import sha256
    cache=Path('.cache/v61-dev');manifest=json.loads((cache/'manifest.json').read_text())
    if manifest['priorManifestSha256']!=provenance['manifestSha256']:raise ValueError('Wrong cached development features')
    for s,rows in [('select',selection),('cal',cal)]:
        if sha256('\n'.join(r['identity'] for r in rows).encode()).hexdigest()!=manifest['identityHashes'][s]:
            raise ValueError('Development row order changed')
        filename=s+'-base-components.npy'
        if file_sha(cache/filename)!=manifest['files'][filename]:raise ValueError('Cached components changed')
    query=json.loads(Path('.cache/v61-cal-query/manifest.json').read_text())
    if query['identitySha256']!=manifest['identityHashes']['cal'] or file_sha('.cache/v61-cal-query/embeddings.npy')!=query['embeddingsSha256']:
        raise ValueError('CAL query cache changed')
    es=encoder.encode(selection,batch_size=1)
    # Both SELECT and CAL use the portable graph and exact one-message runtime.
    # Cross-run embeddings from the original 8-bit graph are not compatible.
    ec=encoder.encode(cal,batch_size=1)
    previous=np.load(cache/'select-base-components.npy');previous_cal=np.load(cache/'cal-base-components.npy')
    old=previous[:,0];old_cal=previous_cal[:,0]
    ns=neighbours(es,semantic['anchors'],semantic['anchorLabels']);nc=neighbours(ec,semantic['anchors'],semantic['anchorLabels'])
    cs=np.column_stack([character_evidence(char,selection),previous[:,1]])
    cc=np.column_stack([character_evidence(char,cal),previous_cal[:,1]])
    fit=np.array([i for i,r in enumerate(selection) if int(r['family'][-12:],16)%10<6])
    choose=np.array([i for i,r in enumerate(selection) if int(r['family'][-12:],16)%10>=6])
    choose_rows=[selection[i] for i in choose];y=np.array([r['y'] for r in selection])
    counts=Counter((group(selection[i]['source']),selection[i]['y']) for i in fit)
    gates={};profiles={};candidates=[]
    for p,rate in PROFILES.items():
        core=semantic['semanticCores'][p]
        xs=joint_features(selection,old,core.predict_proba(es)[:,1],ns,cs)
        xc=joint_features(cal,old_cal,core.predict_proba(ec)[:,1],nc,cc)
        gates[p]={}
        for cost in (1.,4.):
            w=np.array([min(1.,1000/counts[group(selection[i]['source']),selection[i]['y']])*(cost if y[i]==0 else 1.) for i in fit])
            for leaves in (15,31,63):
                name=f'gate{leaves}-ham{cost:g}'
                model=HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=leaves,min_samples_leaf=30,
                                                     l2_regularization=15.,early_stopping=False,random_state=63)
                model.fit(xs[fit],y[fit],sample_weight=w);gates[p][name]=model
            linear=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1200,random_state=63))
            linear.fit(xs[fit],y[fit],logisticregression__sample_weight=w);gates[p][f'linear-ham{cost:g}']=linear
        options=[]
        for name,model in gates[p].items():
            scores=model.predict_proba(xs[choose])[:,1];cut=cuts(choose_rows,scores,rate)
            ranking=rank(choose_rows,scores,cut)
            options.append((ranking,name,cut))
            candidates.append(dict(profile=p,name=name,key=ranking,selection=measure(choose_rows,margins(choose_rows,scores,cut))))
        _,name,cut=max(options);cal_scores=gates[p][name].predict_proba(xc)[:,1]
        cal_cut=cuts(cal,cal_scores,rate);final={k:max(cut[k],cal_cut[k]) for k in cut}
        profiles[p]=dict(selected=name,threshold=0.,scriptCutoffs=final,targetHamFpr=rate,
                         calibration=measure(cal,margins(cal,cal_scores,final)))
    shutil.copytree(root61[0],out/'v61');(out/'v62').mkdir();shutil.copyfile(root62[0],out/'v62/model.joblib')
    artifact=dict(version='v63-joint-character-semantic',gates=gates,profiles=profiles,
                  codeCommit=os.environ.get('GITHUB_SHA'),githubRunId=os.environ.get('GITHUB_RUN_ID'),
                  components={name:file_sha(out/name) for name in ('v61/model.joblib','v62/model.joblib')},
                  featureRoundingDecimals=5,splitManifestSha256=provenance['manifestSha256'])
    artifact['components'].update({str(p.relative_to(out)):file_sha(p) for p in (out/'encoder').iterdir() if p.is_file()})
    artifact['encoderQuantization']='signed-7bit-v1'
    joblib.dump(artifact,out/'model.joblib',compress=3);locked=file_sha(out/'model.joblib')
    print('V63 JOINT MODEL LOCKED',locked,json.dumps(profiles),flush=True)
    bundle=load_bundle(out);ix=np.unique(np.linspace(0,len(cal)-1,64,dtype=int));probe=[cal[i] for i in ix]
    for p in PROFILES:
        core=semantic['semanticCores'][p]
        x=joint_features(probe,old_cal[ix],core.predict_proba(ec[ix])[:,1],neighbours(ec[ix],semantic['anchors'],semantic['anchorLabels']),cc[ix])
        expected=margins(probe,gates[p][profiles[p]['selected']].predict_proba(x)[:,1],profiles[p]['scriptCutoffs'])
        actual=predict_texts(bundle,[r['text'] for r in probe],p,[r.get('old_text',r['text']) for r in probe])
        if not np.allclose(expected,actual,rtol=1e-5,atol=1e-6):
            (out/'runtime-parity-failure.json').write_text(json.dumps(dict(profile=p,expected=expected.tolist(),actual=actual.tolist())))
            raise ValueError('Saved v63 runtime differs')
    tests={}
    def evaluate(rows,title):
        print('V63 EVALUATING',title,len(rows),flush=True)
        e=encoder.encode(rows,batch_size=1);near=neighbours(e,semantic['anchors'],semantic['anchorLabels'])
        chars=character_evidence(char,rows);base=np.empty(len(rows))
        neural=np.empty(len(rows))
        for start in range(0,len(rows),128):
            batch=rows[start:start+128]
            base[start:start+len(batch)],neural[start:start+len(batch)]=train_only_evidence(bundle61[2],batch)
        chars=np.column_stack([chars,neural])
        scores={};report=dict(count=len(rows),ham=sum(r['y']==0 for r in rows),reused=True,profiles={})
        for p in PROFILES:
            x=joint_features(rows,base,semantic['semanticCores'][p].predict_proba(e)[:,1],near,chars)
            raw=gates[p][profiles[p]['selected']].predict_proba(x)[:,1]
            scores[p]=margins(rows,raw,profiles[p]['scriptCutoffs']);report['profiles'][p]=measure(rows,scores[p])
        with (out/(title+'-predictions.csv')).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['identity','family','source','label','conservativeMargin','highRecallMargin'])
            for i,r in enumerate(rows):w.writerow([r['identity'],r['family'],r['source'],r['y'],scores['conservative'][i],scores['high-recall'][i]])
        if file_sha(out/'model.joblib')!=locked:raise ValueError('Model changed after TEST')
        tests[title]=report;print('V63 TEST RESULT',title,json.dumps(report),flush=True)
    fresh=json.loads(Path('.cache/v61-new/provenance.json').read_text())
    if file_sha('.cache/v61-new/test.jsonl.gz')!=fresh['testSha256']:raise ValueError('October snapshot changed')
    evaluate(list(corpus_rows('.cache/v61-new/test.jsonl.gz')),'october-regression')
    evaluate(list(corpus_rows(data/'test.jsonl.gz')),'may-regression')
    evaluate(list(corpus_rows('.cache/v59-data/test.jsonl.gz')),'v59-regression')
    evaluate(legacy_rows(),'legacy-regression')
    acceptance={}
    for p in PROFILES:
        ham=sum(v['ham'] for v in tests.values());fp=sum(v['profiles'][p]['fp'] for v in tests.values());recall=min(v['profiles'][p]['recall'] for v in tests.values())
        acceptance[p]=dict(minimumDatasetRecall=recall,fp=fp,ham=ham,hamFpr=fp/ham,
                           engineeringGoalMet=recall>=.95 and fp<=int(ham*.0001),customerGoalValidated=False)
    report=dict(version=63,modelSha256=locked,codeCommit=artifact['codeCommit'],githubRunId=artifact['githubRunId'],
                profiles=profiles,candidates=candidates,tests=tests,engineeringAcceptance=acceptance,
                estimatorAndThresholdLockedBeforeTest=True,testLabelsUsedForTraining=False,allTestsReused=True,
                productionGoalValidated=False,featureCount=22+cs.shape[1],newTestProvenance=fresh,
                frozenGatesUsedAsInput=False,
                caveat='Gate inputs are TRAIN-fitted semantic, character and Unicode models; older SELECT-fitted gates are excluded. Disjoint SELECT-fit / SELECT-choose then CAL fit/select/calibrate this joint gate. All TEST datasets are explicitly reused engineering checks, not a customer production benchmark.')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    md='# MailGuard v63 — joint character / semantic evidence\n\n'+report['caveat']+'\n\n| Dataset | Profile | Recall | FP / HAM |\n|---|---|---:|---:|\n'
    for name,value in tests.items():
        for p,m in value['profiles'].items():md+=f'| {name} | {p} | {m["recall"]:.2%} | {m["fp"]} / {value["ham"]} |\n'
    (out/'report.md').write_text(md);print(md,flush=True)


if __name__=='__main__':main()
