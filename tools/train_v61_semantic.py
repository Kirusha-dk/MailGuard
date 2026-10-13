"""Fit semantic/neighbour ensemble; freeze both profiles before new October TEST."""
import csv
import gc
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.kernel_approximation import Nystroem
from train_v55_multilingual import corpus_rows,file_sha,views
from train_v59_new_mail import metrics,threshold
from train_v60_new_mail import restore_validation,legacy_rows
from v60_runtime import load_bundle as load60,predict_texts as predict60
from v61_semantic import Encoder,neighbours,script,group,VIEW_VERSION,REVISION
from v61_runtime import stack,raw_score,margins,score_rows

PROFILES={'conservative':.0001,'high-recall':.002}


def cuts(rows,scores,rate):
    overall=threshold(rows,scores,rate); result={}
    for name in ('latin','cjk','other'):
        ix=np.array([i for i,r in enumerate(rows) if script(r)==name],dtype=int)
        subset=[rows[i] for i in ix]
        # No HAM-free language bucket may receive a permissive cutoff. Japanese
        # and Chinese deliberately share CJK; source name never routes inference.
        result[name]=threshold(subset,scores[ix],rate) if sum(r['y']==0 for r in subset)>=300 else overall
    return result


def rank(rows,scores,cut):
    values=[]
    for name in sorted({group(r['source']) for r in rows}):
        ix=np.array([i for i,r in enumerate(rows) if group(r['source'])==name])
        batch=[rows[i] for i in ix]
        if any(r['y'] for r in batch): values.append(metrics(batch,margins(batch,scores[ix],cut),0.)['recall'])
    return min(values),float(np.mean(values))


def measure(rows,scores):
    result=metrics(rows,scores,0.)
    if result['tn']+result['fp']==0: result['fpr']=None
    return result


def main():
    torch.set_num_threads(4)
    root=Path('.cache/v61-dev'); out=Path('reports/v61-full'); out.mkdir(parents=True,exist_ok=False)
    data=Path('.cache/v60-data'); provenance=json.loads((data/'provenance.json').read_text())
    cache=json.loads((root/'manifest.json').read_text())
    if cache['priorManifestSha256']!=provenance['manifestSha256'] or cache['viewVersion']!=VIEW_VERSION or cache['encoderRevision']!=REVISION:
        raise ValueError('Wrong semantic feature cache')
    if cache['newTestRead']: raise ValueError('Feature cache used new TEST')
    if file_sha('.cache/v60-model/model.joblib')!=cache['baseModelSha256']:
        raise ValueError('Base model differs from cached development scores')
    for filename,digest in cache['files'].items():
        if file_sha(root/filename)!=digest: raise ValueError('Feature cache file changed')
    train=list(corpus_rows(root/'train.jsonl.gz'))
    selection=list(corpus_rows(data/'select.jsonl.gz')); cal=list(corpus_rows(data/'cal.jsonl.gz'))
    if any(r.get('weakLabel') for r in selection+cal): raise ValueError('Weak HAM outside TRAIN')
    for split,rows in [('train',train),('select',selection),('cal',cal)]:
        fingerprint=hashlib.sha256('\n'.join(r['identity'] for r in rows).encode()).hexdigest()
        if fingerprint!=cache['identityHashes'][split]: raise ValueError('Feature row order differs')
    restore_validation(selection+cal)
    embeddings=np.load(root/'train-embeddings.npy'); es=np.load(root/'select-embeddings.npy')
    query_root=Path('.cache/v61-cal-query')
    query_cache=json.loads((query_root/'manifest.json').read_text())
    if (query_cache['identitySha256']!=cache['identityHashes']['cal']
        or query_cache['calSplitSha256']!=provenance['splitHashes']['cal']
        or query_cache['queryEncodingBatchSize']!=1
        or query_cache['encoderRevision']!=REVISION
        or query_cache['viewVersion']!=VIEW_VERSION
        or query_cache['newTestRead'] or query_cache['estimatorFit']):
        raise ValueError('Wrong deterministic CAL cache')
    if file_sha(query_root/'embeddings.npy')!=query_cache['embeddingsSha256']:
        raise ValueError('Deterministic CAL embeddings changed')
    ec=np.load(query_root/'embeddings.npy')
    if ec.shape!=(len(cal),384):raise ValueError('Wrong CAL embedding shape')
    print('CALIBRATION restored from exact one-message runtime',flush=True)
    base_select=np.load(root/'select-base.npy'); base_cal=np.load(root/'cal-base.npy')
    resumed=os.environ.get('V61_FROZEN_CHECKPOINT')
    if resumed:
        from v61_runtime import load_bundle
        source=Path(resumed)
        if file_sha(source/'model.joblib')!='07239ee4a2de4824bbf41df17197151ae8d425e7ea0d73106d5814b37f148116':
            raise ValueError('Wrong frozen checkpoint')
        shutil.copytree(source,out,dirs_exist_ok=True)
        artifact,_,_=load_bundle(out)
        cores=artifact['semanticCores']; core_names=artifact['semanticCoreNames']
        anchors=artifact['anchors']; anchor_labels=artifact['anchorLabels']
        gates=artifact['gates']; profiles=artifact['profiles']
        core_results=[]; gate_results=[]
        locked=file_sha(out/'model.joblib')
        print('RESUMING UNCHANGED FROZEN MODEL',locked,flush=True)
    else:
        fit=np.array([i for i,r in enumerate(selection) if int(r['family'][-12:],16)%10<6])
        choose=np.array([i for i,r in enumerate(selection) if int(r['family'][-12:],16)%10>=6])
        choose_rows=[selection[i] for i in choose]
        y=np.array([r['y'] for r in train]); counts=Counter((r['source'],r['y']) for r in train)
        weights=np.array([min(1.,3000/counts[r['source'],r['y']])*(.25 if r.get('weakLabel') else 1.) for r in train])
        candidates=[]; core_results=[]
        core_candidates=[(f'linear-C{c}',LogisticRegression(C=c,max_iter=700,random_state=61)) for c in (4.,20.)]
        for gamma in (4.,12.):
            kernel=make_pipeline(Nystroem(kernel='rbf',gamma=gamma,n_components=min(512,len(train)),random_state=61),
                                 LogisticRegression(C=4.,max_iter=700,random_state=61))
            core_candidates.append((f'rbf-gamma{gamma}',kernel))
        for name,core in core_candidates:
            if name.startswith('rbf-'):
                core.fit(embeddings,y,logisticregression__sample_weight=weights)
            else:
                core.fit(embeddings,y,sample_weight=weights)
            scores=core.predict_proba(es[choose])[:,1]
            keys={}
            for p,rate in PROFILES.items():
                cut=cuts(choose_rows,scores,rate); keys[p]=rank(choose_rows,scores,cut)
                core_results.append(dict(profile=p,name=name,macroKey=keys[p],selection=measure(choose_rows,margins(choose_rows,scores,cut))))
                print('Semantic core SELECT',json.dumps(core_results[-1]),flush=True)
            candidates.append((keys,name,core))
        cores={}; core_names={}
        for p in PROFILES:
            _,core_names[p],cores[p]=max(candidates,key=lambda c:c[0][p])
        # Weak Japanese HAM helps fitting, but is not a gold nearest-neighbour anchor.
        anchor_ix=np.array([i for i,r in enumerate(train) if not r.get('weakLabel')])
        anchors=embeddings[anchor_ix]; anchor_labels=y[anchor_ix].astype(np.uint8)
        ns=neighbours(es,anchors,anchor_labels); nc=neighbours(ec,anchors,anchor_labels)
        ys=np.array([r['y'] for r in selection]); counts=Counter((group(selection[i]['source']),selection[i]['y']) for i in fit)
        gates,profiles,gate_results={},{},[]
        for profile,rate in PROFILES.items():
            core=cores[profile]
            sem_select=core.predict_proba(es)[:,1]; sem_cal=core.predict_proba(ec)[:,1]
            xs=stack(selection,base_select,sem_select,ns); xc=stack(cal,base_cal,sem_cal,nc)
            gates[profile]={}
            for ham_cost in (1.,4.,12.):
                w=np.array([min(1.,1000/counts[group(selection[i]['source']),selection[i]['y']])*(ham_cost if ys[i]==0 else 1.) for i in fit])
                linear=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1200,random_state=61))
                linear.fit(xs[fit],ys[fit],logisticregression__sample_weight=w); gates[profile][f'linear-ham{int(ham_cost)}']=linear
                for leaves in (7,15):
                    gate=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=leaves,min_samples_leaf=60,l2_regularization=15.,random_state=61)
                    gate.fit(xs[fit],ys[fit],sample_weight=w); gates[profile][f'gate{leaves}-ham{int(ham_cost)}']=gate
            options=[]
            for name in ('v60-only','semantic-only',*gates[profile]):
                scores=raw_score(name,gates[profile],xs[choose],base_select[choose],sem_select[choose])
                cut=cuts(choose_rows,scores,rate); key=rank(choose_rows,scores,cut)
                options.append((key,name,cut)); gate_results.append(dict(profile=profile,name=name,macroKey=key,selection=measure(choose_rows,margins(choose_rows,scores,cut))))
            _,name,select_cut=max(options)
            scores=raw_score(name,gates[profile],xc,base_cal,sem_cal); cal_cut=cuts(cal,scores,rate)
            final={k:max(select_cut[k],cal_cut[k]) for k in select_cut}
            profiles[profile]=dict(selected=name,threshold=0.,scriptCutoffs=final,targetHamFpr=rate,calibration=measure(cal,margins(cal,scores,final)))
        source=Path('.cache/v60-model'); (out/'v60').mkdir()
        for leaf in ('model.joblib','unicode.pt'): shutil.copyfile(source/leaf,out/'v60'/leaf)
        shutil.copytree(source/'v59',out/'v60/v59'); shutil.copytree(root/'encoder',out/'encoder')
        artifact=dict(version='v61-semantic-neighbour-ensemble',semanticCores=cores,semanticCoreNames=core_names,anchors=anchors,anchorLabels=anchor_labels,
                      anchorIdentities=[train[i]['identity'] for i in anchor_ix],
                      gates=gates,profiles=profiles,codeCommit=os.environ.get('GITHUB_SHA'),githubRunId=os.environ.get('GITHUB_RUN_ID'),
                      encoderRevision=REVISION,viewVersion=VIEW_VERSION,queryEncodingBatchSize=1,splitManifestSha256=provenance['manifestSha256'],
                      components={str(p.relative_to(out)):file_sha(p) for p in out.rglob('*') if p.is_file()})
        joblib.dump(artifact,out/'model.joblib',compress=3); locked=file_sha(out/'model.joblib')
    print('MODEL AND SCRIPT THRESHOLDS LOCKED',locked,json.dumps(profiles),flush=True)
    from v61_runtime import load_bundle,predict_texts
    bundle=load_bundle(out); encoder=bundle[1]
    runtime_rows=[dict(zip(('focus','context'),views(r['text']))) for r in cal]
    if any((r['focus'],r['context'])!=(t['focus'],t['context']) for r,t in zip(cal,runtime_rows)):
        raise ValueError('CAL views differ from runtime')
    actual_ec=encoder.encode(runtime_rows,batch_size=1)
    parity=dict(embeddingMaximumDifference=float(np.max(np.abs(ec-actual_ec))),count=len(cal),profiles={},newTestRead=False)
    for p in PROFILES:
        cached=score_rows(artifact,cal,ec,base_cal,p)
        actual=score_rows(artifact,cal,actual_ec,base_cal,p)
        parity['profiles'][p]=dict(maximumScoreDifference=float(np.max(np.abs(cached-actual))),
                                   decisionDifferences=int(np.count_nonzero((cached>=0)!=(actual>=0))),
                                   numericParity=bool(np.allclose(cached,actual,rtol=1e-5,atol=1e-6)))
    (out/'cal-runtime-parity.json').write_text(json.dumps(parity,indent=2))
    print('FULL CAL RUNTIME PARITY',json.dumps(parity),flush=True)
    if any(not v['numericParity'] or v['decisionDifferences'] for v in parity['profiles'].values()):
        raise ValueError('Saved CAL encoder differs on this runner')
    for profile in PROFILES:
        ix=np.unique(np.linspace(0,len(cal)-1,12,dtype=int)); probe=[cal[i] for i in ix]
        expected=score_rows(artifact,probe,ec[ix],base_cal[ix],profile)
        observed=predict_texts(bundle,[r['text'] for r in probe],profile,[r.get('old_text',r['text']) for r in probe])
        if not np.allclose(expected,observed,rtol=1e-5,atol=1e-6):
            (out/'probe-parity-failure.json').write_text(json.dumps(dict(profile=profile,expected=expected.tolist(),observed=observed.tolist())))
            raise ValueError('Saved v61 runtime differs')
    baseline_bundle=bundle[2]; previous_cache={}
    for filename in ('fresh-predictions.csv','v59-regression-predictions.csv','legacy-regression-predictions.csv'):
        with (Path('.cache/v60-model')/filename).open() as f:
            previous_cache.update({r['identity']:r for r in csv.DictReader(f)})
    def evaluate(rows,title):
        print('EVALUATING',title,len(rows),flush=True)
        embedding=encoder.encode(rows,batch_size=1); base=np.empty(len(rows)); baseline={p:np.empty(len(rows)) for p in PROFILES}
        for start in range(0,len(rows),128):
            batch=rows[start:start+128]; missing=[i for i,r in enumerate(batch) if r['identity'] not in previous_cache]
            for p,column in [('conservative','v60Conservative'),('high-recall','v60HighRecall')]:
                baseline[p][start:start+len(batch)]=[float(previous_cache[r['identity']][column]) if r['identity'] in previous_cache else np.nan for r in batch]
            if missing:
                texts=[batch[i]['text'] for i in missing]; originals=[batch[i].get('old_text',batch[i]['text']) for i in missing]
                for p in PROFILES:
                    baseline[p][start+np.array(missing)]=predict60(baseline_bundle,texts,p,originals)
        base=baseline['high-recall']; near=neighbours(embedding,anchors,anchor_labels)
        comparisons={}; scores={}
        for p,setting in profiles.items():
            semantic=cores[p].predict_proba(embedding)[:,1]; x=stack(rows,base,semantic,near)
            scores[p]=margins(rows,raw_score(setting['selected'],gates[p],x,base,semantic),setting['scriptCutoffs'])
            old_cut=baseline_bundle[0]['profiles'][p]['threshold']
            old=metrics(rows,baseline[p],old_cut)
            if old['tn']+old['fp']==0: old['fpr']=None
            comparisons[p]=dict(v61=measure(rows,scores[p]),v60=old)
        with (out/(title+'-predictions.csv')).open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(['identity','family','source','label','v61ConservativeMargin','v61HighRecallMargin','v60ConservativeScore','v60HighRecallScore'])
            for i,r in enumerate(rows): writer.writerow([r['identity'],r['family'],r['source'],r['y'],scores['conservative'][i],scores['high-recall'][i],baseline['conservative'][i],baseline['high-recall'][i]])
        if locked!=file_sha(out/'model.joblib'): raise ValueError('Model changed after TEST')
        return dict(count=len(rows),ham=sum(r['y']==0 for r in rows),profiles=comparisons)
    # Only now read the independently frozen new October TEST.
    fresh_provenance=json.loads(Path('.cache/v61-new/provenance.json').read_text())
    if file_sha('.cache/v61-new/test.jsonl.gz')!=fresh_provenance['testSha256']: raise ValueError('New TEST changed')
    fresh=evaluate(list(corpus_rows('.cache/v61-new/test.jsonl.gz')),'new-october')
    # October is a changing archive. Preserve the exact frozen research snapshot
    # in the draft model package so a later rerun can recover the same TEST.
    shutil.copyfile('.cache/v61-new/test.jsonl.gz',out/'new-october-test.jsonl.gz')
    may=evaluate(list(corpus_rows('.cache/v60-data/test.jsonl.gz')),'may-regression')
    chinese=evaluate(list(corpus_rows('.cache/v59-data/test.jsonl.gz')),'v59-regression')
    legacy=evaluate(legacy_rows(),'legacy-regression')
    acceptance={}
    for p in PROFILES:
        datasets=[fresh,may,chinese,legacy]
        recalls=[d['profiles'][p]['v61']['recall'] for d in datasets]
        ham=sum(d['ham'] for d in datasets); fp=sum(d['profiles'][p]['v61']['fp'] for d in datasets)
        acceptance[p]=dict(requiredRecall=.95,requiredHamFpr=.0001,
                           minimumDatasetRecall=min(recalls),fp=fp,ham=ham,
                           empiricalHamFpr=fp/ham,
                           engineeringGoalMet=min(recalls)>=.95 and fp<=int(ham*.0001),
                           customerGoalValidated=False,reusedHamOnly=True)
    result=dict(version=61,codeCommit=os.environ.get('GITHUB_SHA'),githubRunId=os.environ.get('GITHUB_RUN_ID'),modelSha256=locked,
                trainingCommit=artifact['codeCommit'],trainingRunId=artifact['githubRunId'],runtimeParity=parity,
                profiles=profiles,selectedSemanticCores=core_names,coreCandidates=core_results,gateSelection=gate_results,trainSemanticPool=cache['trainPoolCount'],
                freshTest=fresh,mayRegression=dict(may,reused=True),v59Regression=dict(chinese,reused=True),legacyRegression=dict(legacy,reused=True),
                newTestProvenance=fresh_provenance,newTestSnapshotFile='new-october-test.jsonl.gz',featureManifest=cache,calibrationFeatureManifest=query_cache,estimatorAndThresholdLockedBeforeTest=True,
                testLabelsUsedForTraining=False,productionGoalValidated=False,engineeringAcceptance=acceptance,
                caveat='New October TEST is SPAM-only. All HAM evaluations are explicitly reused engineering tests; this cannot establish the customer false-positive rate. Pretrained encoder is frozen; our semantic and stack classifiers fit development only.')
    (out/'report.json').write_text(json.dumps(result,indent=2)); (out/'new-test-provenance.json').write_text(json.dumps(fresh_provenance,indent=2))
    md='# MailGuard v61 — semantic neighbour ensemble\n\n'+result['caveat']+'\n\n'
    md+='| Dataset | Model/profile | Mail | HAM | Spam recall | FP | FN |\n|---|---|---:|---:|---:|---:|---:|\n'
    for title,dataset in [('New October',fresh),('Reused May',may),('Reused v59',chinese),('Reused English/CSDMC',legacy)]:
        for p,comparison in dataset['profiles'].items():
            for version,m in comparison.items(): md+=f'| {title} | {version} {p} | {dataset["count"]} | {dataset["ham"]} | {m["recall"]:.2%} | {m["fp"]} | {m["fn"]} |\n'
    (out/'report.md').write_text(md); print(md,flush=True)


if __name__=='__main__': main()
