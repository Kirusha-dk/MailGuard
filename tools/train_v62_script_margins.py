"""Fit language-specific NB-SVM on TRAIN, choose on SELECT, calibrate before TEST."""
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import save_npz,load_npz
from sklearn.linear_model import SGDClassifier
from train_v55_multilingual import corpus_rows,file_sha,transformed
from train_v59_new_mail import metrics,threshold
from train_v60_new_mail import restore_validation,legacy_rows
from v61_semantic import training_pool,script,group
from v62_runtime import FEATURES,features,score_rows,load_bundle,predict_texts

PROFILES={'conservative':.0001,'high-recall':.002}
SCOPES=('global','latin','cjk')


def key(rows,scores,cut):
    recalls=[]
    for name in sorted({group(r['source']) for r in rows}):
        ix=np.array([i for i,r in enumerate(rows) if group(r['source'])==name])
        batch=[rows[i] for i in ix]
        if any(r['y'] for r in batch):recalls.append(metrics(batch,scores[ix],cut)['recall'])
    return min(recalls),float(np.mean(recalls))


def candidate_scores(rows,cores,idf,ratio):
    result={scope:{name:np.empty(len(rows)) for name in values} for scope,values in cores.items()}
    for start in range(0,len(rows),128):
        raw=features(rows[start:start+128])
        for scope,values in cores.items():
            x=transformed(raw,idf[scope],ratio[scope])
            for name,model in values.items():result[scope][name][start:start+raw.shape[0]]=model.decision_function(x)
    return result


def measure(rows,scores):
    m=metrics(rows,scores,0.)
    if m['tn']+m['fp']==0:m['fpr']=None
    return m


def main():
    out=Path('reports/v62-full');out.mkdir(parents=True,exist_ok=False)
    data=Path('.cache/v60-data');provenance=json.loads((data/'provenance.json').read_text())
    for split,digest in provenance['splitHashes'].items():
        if file_sha(data/f'{split}.jsonl.gz')!=digest:raise ValueError('Frozen split changed')
    train=training_pool(list(corpus_rows(data/'train.jsonl.gz')))
    selection=list(corpus_rows(data/'select.jsonl.gz'));cal=list(corpus_rows(data/'cal.jsonl.gz'))
    restore_validation(selection+cal)
    if any(r.get('weakLabel') for r in selection+cal):raise ValueError('Weak labels outside TRAIN')
    rng=np.random.default_rng(62);train=[train[i] for i in rng.permutation(len(train))]
    labels=np.array([r['y'] for r in train]);scripts=np.array([script(r) for r in train])
    counts=Counter((r['source'],r['y']) for r in train)
    weights=np.array([min(1.,3000/counts[r['source'],r['y']])*(.25 if r.get('weakLabel') else 1.) for r in train])
    cache=Path('.cache/v62-fit');cache.mkdir(parents=True,exist_ok=False)
    df={g:np.zeros(FEATURES) for g in SCOPES}
    totals={g:[np.zeros(FEATURES),np.zeros(FEATURES)] for g in SCOPES}
    n={g:0 for g in SCOPES}
    for start in range(0,len(train),256):
        raw=features(train[start:start+256]);save_npz(cache/f'{start:08d}.npz',raw)
        ys=labels[start:start+raw.shape[0]];ss=scripts[start:start+raw.shape[0]];w=weights[start:start+raw.shape[0]]
        for scope in SCOPES:
            ix=np.flatnonzero(np.ones(len(ys),dtype=bool) if scope=='global' else ss==scope)
            if not len(ix):continue
            x=raw[ix];n[scope]+=len(ix);df[scope]+=np.bincount(x.indices,minlength=FEATURES)
            for y in (0,1):
                mask=ys[ix]==y
                totals[scope][y]+=np.asarray(x[mask].multiply(w[ix][mask,None]).sum(axis=0)).ravel()
        if start%8192==0:print('V62 TRAIN cache',start,'/',len(train),flush=True)
    idf={g:(np.log((1+n[g])/(1+df[g]))+1).astype(np.float32) for g in SCOPES}
    ratio={}
    for g in SCOPES:
        pos,neg=totals[g][1]+1.,totals[g][0]+1.
        ratio[g]=np.log((pos/pos.sum())/(neg/neg.sum())).astype(np.float32)
    del df,totals,pos,neg
    cores={g:{} for g in SCOPES}
    settings=[('hinge',a,c) for a in (1e-6,3e-7) for c in (1.,4.)]+[('log_loss',1e-6,4.)]
    for g in SCOPES:
        if len(set(labels[scripts==g]))<2 and g!='global':raise ValueError('Both gold classes required in specialist TRAIN')
        for loss,alpha,cost in settings:
            name=f'{loss}-alpha{alpha}-ham{cost:g}'
            cores[g][name]=SGDClassifier(loss=loss,alpha=alpha,average=True,random_state=62)
    for epoch in range(8):
        for start in rng.permutation(np.arange(0,len(train),256)):
            raw=load_npz(cache/f'{start:08d}.npz');within=rng.permutation(raw.shape[0]);raw=raw[within]
            ix_all=start+within
            for g,values in cores.items():
                local=np.flatnonzero(np.ones(len(within),dtype=bool) if g=='global' else scripts[ix_all]==g)
                if not len(local):continue
                ix=ix_all[local];x=transformed(raw[local],idf[g],ratio[g])
                for (loss,alpha,cost),(name,model) in zip(settings,values.items()):
                    model.partial_fit(x,labels[ix],classes=np.array([0,1]),sample_weight=weights[ix]*np.where(labels[ix]==0,cost,1.))
        print('V62 NB-margin epoch',epoch+1,'/8',flush=True)
    shutil.rmtree(cache)
    scores_select=candidate_scores(selection,cores,idf,ratio);scores_cal=candidate_scores(cal,cores,idf,ratio)
    profiles={};candidates=[]
    for profile,rate in PROFILES.items():
        scripts_selected={}
        for lang in ('latin','cjk','other'):
            ix=np.array([i for i,r in enumerate(selection) if script(r)==lang],dtype=int)
            safe=sum(selection[i]['y']==0 for i in ix)>=300
            choose=ix if safe else np.arange(len(selection));batch=[selection[i] for i in choose]
            scopes=('global',lang) if safe and lang in cores else ('global',)
            options=[]
            for scope in scopes:
                for name,values in scores_select[scope].items():
                    cut=threshold(batch,values[choose],rate);rank=key(batch,values[choose],cut)
                    candidates.append(dict(profile=profile,script=lang,scope=scope,name=name,key=rank,selection=metrics(batch,values[choose],cut)))
                    options.append((rank,scope,name,cut))
            rank,scope,name,cut=max(options)
            ci=np.array([i for i,r in enumerate(cal) if script(r)==lang],dtype=int)
            if not safe or sum(cal[i]['y']==0 for i in ci)<300:ci=np.arange(len(cal))
            cal_cut=threshold([cal[i] for i in ci],scores_cal[scope][name][ci],rate)
            scripts_selected[lang]=dict(scope=scope,model=name,cutoff=max(cut,cal_cut),selectionKey=rank,scriptHasGoldHam=safe)
        profiles[profile]=dict(threshold=0.,targetHamFpr=rate,scripts=scripts_selected)
    artifact=dict(version='v62-script-nb-margin',cores=cores,idf=idf,ratio=ratio,profiles=profiles,
                  components={},codeCommit=os.environ.get('GITHUB_SHA'),githubRunId=os.environ.get('GITHUB_RUN_ID'),
                  trainCount=len(train),trainIdentitySha256=hashlib.sha256('\n'.join(r['identity'] for r in train).encode()).hexdigest(),
                  splitManifestSha256=provenance['manifestSha256'])
    joblib.dump(artifact,out/'model.joblib',compress=3);locked=file_sha(out/'model.joblib')
    print('V62 MODEL LOCKED BEFORE TEST',locked,json.dumps(profiles),flush=True)
    bundle=load_bundle(out);calibration={}
    ix=np.unique(np.linspace(0,len(cal)-1,64,dtype=int));probe=[cal[i] for i in ix]
    for profile in PROFILES:
        expected=score_rows(artifact,probe,profile,batch_size=7)
        observed=predict_texts(bundle,[r['text'] for r in probe],profile)
        if not np.allclose(expected,observed,rtol=1e-6,atol=1e-7):raise ValueError('V62 saved runtime differs')
        calibration[profile]=measure(cal,score_rows(bundle,cal,profile))
    comparisons={}
    def evaluate(rows,title,reused):
        result=dict(count=len(rows),ham=sum(r['y']==0 for r in rows),reused=reused,profiles={})
        scores={p:score_rows(bundle,rows,p) for p in PROFILES}
        for p in PROFILES:result['profiles'][p]=measure(rows,scores[p])
        with (out/(title+'-predictions.csv')).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['identity','family','source','label','conservativeMargin','highRecallMargin'])
            for i,r in enumerate(rows):w.writerow([r['identity'],r['family'],r['source'],r['y'],scores['conservative'][i],scores['high-recall'][i]])
        if file_sha(out/'model.joblib')!=locked:raise ValueError('V62 model changed after TEST')
        print('V62 TEST',title,json.dumps(result),flush=True);comparisons[title]=result
    new=json.loads(Path('.cache/v61-new/provenance.json').read_text())
    if file_sha('.cache/v61-new/test.jsonl.gz')!=new['testSha256']:raise ValueError('October snapshot changed')
    evaluate(list(corpus_rows('.cache/v61-new/test.jsonl.gz')),'october-regression',True)
    evaluate(list(corpus_rows(data/'test.jsonl.gz')),'may-regression',True)
    evaluate(list(corpus_rows('.cache/v59-data/test.jsonl.gz')),'v59-regression',True)
    evaluate(legacy_rows(),'legacy-regression',True)
    acceptance={}
    for p in PROFILES:
        values=list(comparisons.values());ham=sum(v['ham'] for v in values);fp=sum(v['profiles'][p]['fp'] for v in values)
        recall=min(v['profiles'][p]['recall'] for v in values)
        acceptance[p]=dict(minimumDatasetRecall=recall,fp=fp,ham=ham,hamFpr=fp/ham,
                           engineeringGoalMet=recall>=.95 and fp<=int(ham*.0001),customerGoalValidated=False)
    report=dict(version=62,codeCommit=artifact['codeCommit'],githubRunId=artifact['githubRunId'],modelSha256=locked,
                profiles=profiles,calibration=calibration,trainCount=len(train),trainScriptCounts=dict(Counter(scripts)),
                candidates=candidates,tests=comparisons,engineeringAcceptance=acceptance,
                estimatorAndThresholdLockedBeforeTest=True,testLabelsUsedForTraining=False,
                productionGoalValidated=False,allTestsReused=True,newTestProvenance=new,
                caveat='All TEST sets are explicitly reused engineering regressions, not new production evidence. Script routing uses text only; unknown/underrepresented scripts fall back to globally calibrated margins.')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    md='# MailGuard v62 — script-specific NB margins\n\n'+report['caveat']+'\n\n| Dataset | Profile | Recall | FP / HAM |\n|---|---|---:|---:|\n'
    for name,value in comparisons.items():
        for p,m in value['profiles'].items():md+=f'| {name} | {p} | {m["recall"]:.2%} | {m["fp"]} / {value["ham"]} |\n'
    (out/'report.md').write_text(md);print(md,flush=True)


if __name__=='__main__':main()
