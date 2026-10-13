"""Inspect development mistakes only; public CAL, no TEST or model changes."""
import json
from pathlib import Path
import joblib
import numpy as np
from train_v55_multilingual import corpus_rows,file_sha
from v61_runtime import score_rows


def main():
    roots=[p.parent for p in Path('.cache/v61-evidence').rglob('model.joblib') if (p.parent/'encoder').exists()]
    if len(roots)!=1:raise ValueError('Expected one frozen semantic model')
    root=roots[0];artifact=joblib.load(root/'model.joblib')
    rows=list(corpus_rows('.cache/v60-data/cal.jsonl.gz'))
    embedding=np.load('.cache/v61-cal-query/embeddings.npy');base=np.load('.cache/v61-dev/cal-base.npy')
    cases={}
    def case(i,score):
        r=rows[i]
        return dict(identity=r['identity'],source=r['source'],label=r['y'],margin=float(score),
                    focus=r['focus'][:1000],context=r['context'][:1400])
    for p in artifact['profiles']:
        scores=score_rows(artifact,rows,embedding,base,p)
        false_positives=[case(i,scores[i]) for i,r in enumerate(rows) if r['y']==0 and scores[i]>=0]
        hardest_ham=sorted([i for i,r in enumerate(rows) if r['y']==0],key=lambda i:scores[i],reverse=True)[:12]
        missed=[]
        for source in sorted({r['source'] for r in rows}):
            ix=[i for i,r in enumerate(rows) if r['source']==source and r['y']==1 and scores[i]<0]
            for i in sorted(ix,key=lambda i:rows[i]['identity'])[:2]:missed.append(case(i,scores[i]))
        cases[p]=dict(falsePositives=false_positives,highestHamMargins=[case(i,scores[i]) for i in hardest_ham],missesBySource=missed)
    result=dict(modelSha256=file_sha(root/'model.joblib'),githubRunId=__import__('os').environ.get('GITHUB_RUN_ID'),
                split='CAL',publicCorpus=True,newTestRead=False,modelChanged=False,profiles=cases)
    out=Path('reports/v61-development-errors.json');out.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
