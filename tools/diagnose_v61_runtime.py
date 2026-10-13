"""CAL-only saved-runtime diagnostic; never fits a model or opens TEST."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from train_v55_multilingual import corpus_rows, views, file_sha
from train_v60_new_mail import restore_validation
from v60_runtime import predict_texts as predict60
from v61_semantic import windows, neighbours, script
from v61_runtime import load_bundle, stack, raw_score, margins


def delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return dict(shape=list(a.shape), maximum=float(np.max(np.abs(a-b))),
                perRow=np.max(np.abs(a-b).reshape(len(a),-1),axis=1).tolist())


def main():
    torch.set_num_threads(4)
    roots=list(Path('.cache/v61-evidence').rglob('model.joblib'))
    roots=[p.parent for p in roots if (p.parent/'encoder').exists()]
    if len(roots)!=1: raise ValueError('Expected one semantic bundle')
    root=roots[0]; bundle=load_bundle(root); art, encoder, base_bundle=bundle
    rows=list(corpus_rows('.cache/v60-data/cal.jsonl.gz'))
    restore_validation(rows)
    ix=np.unique(np.linspace(0,len(rows)-1,12,dtype=int)); probe=[rows[i] for i in ix]
    runtime=[dict(zip(('focus','context'),views(r['text']))) for r in probe]
    expected_e=np.load('.cache/v61-cal-query/embeddings.npy')[ix]
    actual_e=encoder.encode(runtime,batch_size=1)
    from_rows=encoder.encode(probe,batch_size=1)
    expected_b=np.load('.cache/v61-dev/cal-base.npy')[ix]
    actual_b=predict60(base_bundle,[r['text'] for r in probe],'high-recall',[r.get('old_text',r['text']) for r in probe])
    expected_n=neighbours(expected_e,art['anchors'],art['anchorLabels'])
    actual_n=neighbours(actual_e,art['anchors'],art['anchorLabels'])
    result=dict(modelSha256=file_sha(root/'model.joblib'),newTestRead=False,estimatorFit=False,
                identities=[r['identity'] for r in probe],sources=[r['source'] for r in probe],
                viewsEqual=[(r['focus'],r['context'])==(t['focus'],t['context']) for r,t in zip(probe,runtime)],
                windowsEqual=[windows(r)==windows(t) for r,t in zip(probe,runtime)],
                scriptsEqual=[script(r)==script(t) for r,t in zip(probe,runtime)],
                embeddings=delta(expected_e,actual_e),cachedVsSameViews=delta(expected_e,from_rows),
                base=delta(expected_b,actual_b),neighbours=delta(expected_n,actual_n),profiles={})
    for p in art['profiles']:
        core=art['semanticCores'][p]
        expected_s=core.predict_proba(expected_e)[:,1]; actual_s=core.predict_proba(actual_e)[:,1]
        expected_x=stack(probe,expected_b,expected_s,expected_n); actual_x=stack(runtime,actual_b,actual_s,actual_n)
        name=art['profiles'][p]['selected']; gates=art['gates'][p]
        expected_raw=raw_score(name,gates,expected_x,expected_b,expected_s)
        actual_raw=raw_score(name,gates,actual_x,actual_b,actual_s)
        expected=margins(probe,expected_raw,art['profiles'][p]['scriptCutoffs'])
        actual=margins(runtime,actual_raw,art['profiles'][p]['scriptCutoffs'])
        result['profiles'][p]=dict(semantic=delta(expected_s,actual_s),features=delta(expected_x,actual_x),
                                  featuresPerColumn=np.max(np.abs(expected_x-actual_x),axis=0).tolist(),
                                  raw=delta(expected_raw,actual_raw),margins=delta(expected,actual),
                                  expected=expected.tolist(),actual=actual.tolist(),
                                  parityPassed=bool(np.allclose(expected,actual,rtol=1e-5,atol=1e-6)))
    out=Path('reports/v61-runtime-diagnostic.json'); out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
