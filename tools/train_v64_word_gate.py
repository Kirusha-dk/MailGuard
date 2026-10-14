"""Fit new experts on full TRAIN; fit/select gates on separate SELECT families; CAL cuts."""
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import save_npz,load_npz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from train_v55_multilingual import corpus_rows,file_sha
from train_v59_new_mail import threshold
from v61_semantic import script,group
from v62_runtime import load_bundle as load62
from v64_runtime import BRANCHES,DIMENSIONS,word_features,transform,evidence,load_bundle,score_rows

PARENT_SHA = '380991f74ae7ddc4d26cb36fd952ff7a1b6a24a1c8985becf28cb14d2f493aaf'
PROFILES = {'conservative':.0001,'high-recall':.002}


def cuts(rows,scores,rate):
    result = {}
    for lang in ('latin','cjk','other'):
        ix = np.asarray([i for i,r in enumerate(rows) if script(r)==lang],dtype=int)
        if sum(rows[i]['y']==0 for i in ix)<300:
            ix = np.arange(len(rows))
        result[lang] = threshold([rows[i] for i in ix],scores[ix],rate)
    return result


def margins(rows,scores,cfg):
    return scores-np.asarray([cfg[script(r)] for r in rows])


def measure(rows,scores):
    from evaluate_v62_500k import measure as counts_measure
    y = np.asarray([r['y'] for r in rows]);positive = scores>=0
    return counts_measure(dict(tp=int(np.sum((y==1)&positive)),fn=int(np.sum((y==1)&~positive)),
                               fp=int(np.sum((y==0)&positive)),tn=int(np.sum((y==0)&~positive))))


def rank(rows,scores):
    buckets = {}
    for i,r in enumerate(rows):
        buckets.setdefault((group(r['source']),script(r)),[]).append(i)
    recalls = []
    for ix in buckets.values():
        positive = [i for i in ix if rows[i]['y']==1]
        if len(positive)>=50:
            recalls.append(float(np.mean(scores[positive]>=0)))
    return min(recalls),float(np.mean(recalls)),measure(rows,scores)['recall']


def main():
    out = Path('reports/v64-full');out.mkdir(parents=True,exist_ok=False)
    parent_files = list(Path('.cache/v62').rglob('model.joblib'))
    if len(parent_files)!=1 or file_sha(parent_files[0])!=PARENT_SHA:
        raise ValueError('Wrong frozen v62 parent')
    parent = load62(parent_files[0].parent)
    data = Path('.cache/v60-data');provenance = json.loads((data/'provenance.json').read_text())
    if parent['splitManifestSha256']!=provenance['manifestSha256']:
        raise ValueError('Parent and new models use different split manifests')
    splits = {}
    for s in ('train','select','cal'):
        if file_sha(data/f'{s}.jsonl.gz')!=provenance['splitHashes'][s]:
            raise ValueError('Frozen development split changed')
        splits[s] = list(corpus_rows(data/f'{s}.jsonl.gz'))
    train,selection,cal = (splits[s] for s in ('train','select','cal'))
    if any(r.get('weakLabel') for r in selection+cal):
        raise ValueError('Weak labels outside TRAIN')
    rng = np.random.default_rng(64)
    train = [train[i] for i in rng.permutation(len(train))]
    labels = np.asarray([r['y'] for r in train])
    counts = Counter((script(r),group(r['source']),r['y']) for r in train)
    weights = np.asarray([min(1.,4000/counts[script(r),group(r['source']),r['y']])*(.25 if r.get('weakLabel') else 1.) for r in train])
    cache = Path('.cache/v64-word-fit');cache.mkdir(parents=True,exist_ok=False)
    idf,ratio,cores = {},{},{}
    settings = [('hinge',1.),('hinge',4.),('log_loss',4.)]
    for branch in BRANCHES:
        root = cache/branch;root.mkdir()
        df = np.zeros(DIMENSIONS[branch]);totals = [np.zeros(DIMENSIONS[branch]),np.zeros(DIMENSIONS[branch])]
        for start in range(0,len(train),256):
            raw = word_features(train[start:start+256],branch)
            save_npz(root/f'{start:08d}.npz',raw)
            df += np.bincount(raw.indices,minlength=DIMENSIONS[branch])
            ys = labels[start:start+raw.shape[0]];w = weights[start:start+raw.shape[0]]
            for y in (0,1):
                mask = ys==y
                totals[y] += np.asarray(raw[mask].multiply(w[mask,None]).sum(axis=0)).ravel()
            if start%8192==0:
                print('V64 TRAIN FEATURES',branch,start,'/',len(train),flush=True)
        idf[branch] = (np.log((1+len(train))/(1+df))+1).astype(np.float32)
        pos,neg = totals[1]+1.,totals[0]+1.
        ratio[branch] = np.log((pos/pos.sum())/(neg/neg.sum())).astype(np.float32)
        cores[branch] = {f'{loss}-ham{cost:g}':SGDClassifier(loss=loss,alpha=1e-6,average=True,random_state=64)
                         for loss,cost in settings}
        del df,totals,pos,neg
        for epoch in range(6):
            for start in rng.permutation(np.arange(0,len(train),256)):
                raw = load_npz(root/f'{start:08d}.npz');within = rng.permutation(raw.shape[0]);ix = start+within
                x = transform(raw[within],idf[branch],ratio[branch])
                for loss,cost in settings:
                    cores[branch][f'{loss}-ham{cost:g}'].partial_fit(x,labels[ix],classes=np.asarray([0,1]),
                            sample_weight=weights[ix]*np.where(labels[ix]==0,cost,1.))
            print('V64 WORD EPOCH',branch,epoch+1,'/6',flush=True)
        shutil.rmtree(root)
    shutil.rmtree(cache)
    artifact = dict(version='v64-word-character-gate',cores=cores,idf=idf,ratio=ratio,
                    trainCount=len(train),trainIdentitySha256=hashlib.sha256('\n'.join(r['identity'] for r in train).encode()).hexdigest(),
                    splitManifestSha256=provenance['manifestSha256'],codeCommit=os.environ.get('GITHUB_SHA'),
                    githubRunId=os.environ.get('GITHUB_RUN_ID'),components={},profiles={},gates={},
                    runtimeSha256=file_sha('tools/v64_runtime.py'))
    def build(rows):
        result = []
        for start in range(0,len(rows),128):
            result.append(evidence(artifact,parent,rows[start:start+128]))
            if start%4096==0:print('V64 DEVELOPMENT EVIDENCE',start,'/',len(rows),flush=True)
        return np.concatenate(result)
    xs,xc = build(selection),build(cal)
    fit = np.asarray([i for i,r in enumerate(selection) if int(r['family'][-12:],16)%10<6])
    choose = np.asarray([i for i,r in enumerate(selection) if int(r['family'][-12:],16)%10>=6])
    choose_rows = [selection[i] for i in choose];y = np.asarray([r['y'] for r in selection])
    counts = Counter((group(selection[i]['source']),script(selection[i]),selection[i]['y']) for i in fit)
    for cost in (1.,4.,12.):
        w = np.asarray([min(1.,1500/counts[group(selection[i]['source']),script(selection[i]),y[i]])*(cost if y[i]==0 else 1.) for i in fit])
        for leaves in (7,15,31):
            name = f'gate{leaves}-ham{cost:g}'
            gate = HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=leaves,min_samples_leaf=40,
                        l2_regularization=20.,early_stopping=False,random_state=64)
            gate.fit(xs[fit],y[fit],sample_weight=w);artifact['gates'][name]=gate
        for c in (1.,10.):
            name = f'linear{c:g}-ham{cost:g}'
            gate = make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1200,random_state=64))
            gate.fit(xs[fit],y[fit],logisticregression__sample_weight=w);artifact['gates'][name]=gate
    candidates = []
    for p,rate in PROFILES.items():
        options = []
        for name,gate in artifact['gates'].items():
            scores = gate.predict_proba(xs[choose])[:,1];cut = cuts(choose_rows,scores,rate)
            predicted = margins(choose_rows,scores,cut);ranking = rank(choose_rows,predicted)
            options.append((ranking,name,cut))
            candidates.append(dict(profile=p,name=name,rank=ranking,selection=measure(choose_rows,predicted)))
        _,name,cut = max(options)
        cal_scores = artifact['gates'][name].predict_proba(xc)[:,1]
        cal_cut = cuts(cal,cal_scores,rate)
        final = {s:max(cut[s],cal_cut[s]) for s in cut}
        artifact['profiles'][p] = dict(selected=name,threshold=0.,scriptCutoffs=final,targetHamFpr=rate,
                                        calibration=measure(cal,margins(cal,cal_scores,final)))
    (out/'v62').mkdir();shutil.copyfile(parent_files[0],out/'v62/model.joblib')
    artifact['components'] = {'v62/model.joblib':PARENT_SHA}
    joblib.dump(artifact,out/'model.joblib',compress=3);locked = file_sha(out/'model.joblib')
    print('V64 MODEL LOCKED BEFORE TEST',locked,json.dumps(artifact['profiles']),flush=True)
    bundle = load_bundle(out)
    ix = np.unique(np.linspace(0,len(cal)-1,64,dtype=int));probe = [cal[i] for i in ix]
    expected = {p:margins(probe,artifact['gates'][cfg['selected']].predict_proba(xc[ix])[:,1],cfg['scriptCutoffs'])
                for p,cfg in artifact['profiles'].items()}
    actual = score_rows(bundle,probe,batch_size=7)
    for p in PROFILES:
        if not np.allclose(expected[p],actual[p],rtol=1e-6,atol=1e-7) or np.any((expected[p]>=0)!=(actual[p]>=0)):
            raise ValueError('Saved runtime differs on CAL')
    report = dict(version=64,modelSha256=locked,parentModelSha256=PARENT_SHA,trainCount=len(train),
                  codeCommit=artifact['codeCommit'],githubRunId=artifact['githubRunId'],profiles=artifact['profiles'],
                  trainScriptClassCounts={str(k):v for k,v in Counter((script(r),r['y']) for r in train).items()},
                  candidates=candidates,featureCount=xs.shape[1],runtimeParityPassed=True,
                  estimatorAndThresholdLockedBeforeTest=True,testLabelsUsedForTraining=False,
                  frozenGatesUsedAsInput=False,productionGoalValidated=False,
                  caveat='New word experts use full frozen TRAIN; every character expert is TRAIN-fitted. SELECT-fit and SELECT-choose families are disjoint; CAL fixes thresholds. TEST labels do not fit, choose or calibrate this model.')
    (out/'training-report.json').write_text(json.dumps(report,indent=2))
    print('V64 TRAINING COMPLETE',json.dumps(artifact['profiles']),flush=True)


if __name__=='__main__':main()
