"""Trace CAL-only runtime disagreement component by component; no fitting/TEST."""
import json
from pathlib import Path
import numpy as np
import torch
from train_v55_multilingual import corpus_rows,views,file_sha
from train_v60_new_mail import restore_validation
from v63_runtime import load_bundle,character_evidence,joint_features,train_only_evidence,predict_texts
from v63_neighbours import neighbours
from v61_runtime import margins


def diff(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return dict(maximum=float(np.max(np.abs(a-b))),perRow=np.max(np.abs(a-b).reshape(len(a),-1),axis=1).tolist())


def main():
    torch.set_num_threads(4)
    roots=[p.parent for p in Path('.cache/v63-frozen').rglob('model.joblib') if (p.parent/'v61').exists() and (p.parent/'v62').exists()]
    if len(roots)!=1:raise ValueError('Expected one frozen joint model')
    root=roots[0];bundle=load_bundle(root);art,semantic_bundle,char=bundle
    old,encoder,base_bundle=semantic_bundle
    rows=list(corpus_rows('.cache/v60-data/cal.jsonl.gz'));restore_validation(rows)
    ix=np.unique(np.linspace(0,len(rows)-1,64,dtype=int));probe=[rows[i] for i in ix]
    runtime=[dict(zip(('focus','context'),views(r['text']))) for r in probe]
    ec=np.load('.cache/v61-cal-query/embeddings.npy')[ix]
    actual_ec=encoder.encode(runtime,batch_size=1)
    expected=np.load('.cache/v61-dev/cal-base-components.npy')[ix]
    actual_sparse,actual_neural=train_only_evidence(base_bundle,runtime)
    # Original trainer scored every CAL row in blocks of 128; reproduce that
    # exact partition, including each probe's neighbouring rows in its block.
    expected_char=np.empty((len(probe),15))
    for start in sorted({int(i//128*128) for i in ix}):
        block=character_evidence(char,rows[start:start+128])
        for j,index in enumerate(ix):
            if start<=index<start+128:expected_char[j]=block[index-start]
    actual_char=character_evidence(char,runtime)
    ne=neighbours(ec,old['anchors'],old['anchorLabels']);na=neighbours(actual_ec,old['anchors'],old['anchorLabels'])
    report=dict(modelSha256=file_sha(root/'model.joblib'),newTestRead=False,estimatorFit=False,
                viewsEqual=[(r['focus'],r['context'])==(t['focus'],t['context']) for r,t in zip(probe,runtime)],
                embeddings=diff(ec,actual_ec),sparse=diff(expected[:,0],actual_sparse),neural=diff(expected[:,1],actual_neural),
                character=diff(expected_char,actual_char),neighbours=diff(ne,na),profiles={})
    original=json.loads((root/'runtime-parity-failure.json').read_text())
    for p,setting in art['profiles'].items():
        core=old['semanticCores'][p];se=core.predict_proba(ec)[:,1];sa=core.predict_proba(actual_ec)[:,1]
        xe=joint_features(probe,expected[:,0],se,ne,np.column_stack([expected_char,expected[:,1]]))
        xa=joint_features(runtime,actual_sparse,sa,na,np.column_stack([actual_char,actual_neural]))
        gate=art['gates'][p][setting['selected']]
        pe=margins(probe,gate.predict_proba(xe)[:,1],setting['scriptCutoffs'])
        pa=margins(runtime,gate.predict_proba(xa)[:,1],setting['scriptCutoffs'])
        changes=np.abs(xe-xa);locations=np.argwhere(changes>1e-6)
        report['profiles'][p]=dict(semantic=diff(se,sa),features=diff(xe,xa),
                                  perColumn=changes.max(0).tolist(),margins=diff(pe,pa),
                                  decisionDifferences=int(np.count_nonzero((pe>=0)!=(pa>=0))),
                                  changedValues=[dict(row=int(i),column=int(j),expected=float(xe[i,j]),actual=float(xa[i,j])) for i,j in locations[:64]])
        if p==original['profile']:report['profiles'][p]['originalTrainingVsLoadedCached']=diff(original['expected'],pe)
    out=Path('reports/v63-runtime-diagnostic.json');out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
