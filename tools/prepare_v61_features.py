"""Cache only TRAIN/SELECT/CAL embeddings and frozen base scores, never new TEST."""
import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from train_v55_multilingual import corpus_rows,file_sha
from train_v60_new_mail import restore_validation
from train_v59_new_mail import reference_scores,score_batches
from train_v56_consensus import gate_features
from v59_runtime import predict_texts as predict59
from v60_neural import predict as neural_predict
from v60_runtime import load_bundle,predict_texts,stack_features,candidate_score
from v61_semantic import training_pool,fetch_encoder,Encoder,VIEW_VERSION,REVISION


def main():
    torch.set_num_threads(4)
    data=Path('.cache/v60-data'); out=Path('.cache/v61-dev'); out.mkdir(parents=True,exist_ok=False)
    provenance=json.loads((data/'provenance.json').read_text())
    for split,sha in provenance['splitHashes'].items():
        if file_sha(data/f'{split}.jsonl.gz')!=sha: raise ValueError('Development split hash mismatch')
    train=training_pool(list(corpus_rows(data/'train.jsonl.gz')))
    with gzip.open(out/'train.jsonl.gz','wt',encoding='utf-8') as f:
        for r in train: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    rows_by_split=dict(train=train,select=list(corpus_rows(data/'select.jsonl.gz')),cal=list(corpus_rows(data/'cal.jsonl.gz')))
    restore_validation(rows_by_split['select']+rows_by_split['cal'])
    fetch_encoder(out/'encoder'); encoder=Encoder(out/'encoder')
    for split,rows in rows_by_split.items():
        print('ENCODING',split,len(rows),flush=True)
        np.save(out/f'{split}-embeddings.npy',encoder.encode(rows))
    del encoder,train
    bundle=load_bundle('.cache/v60-model'); art60,cnn60,bundle59=bundle
    art59,old54,nn54,stream55=bundle59
    with Path('.cache/v54-evidence/predictions.csv').open() as f:
        cached={r['identity']:float(r['score']) for r in csv.DictReader(f)}
    for split in ('select','cal'):
        rows=rows_by_split[split]; print('FROZEN BASE SCORES',split,len(rows),flush=True)
        old,stream=reference_scores(rows,old54,nn54,stream55,cached)
        new59=score_batches(art59['core'],rows)
        x59=np.column_stack([gate_features(rows,old,new59,old54['threshold']),stream])
        setting59=art59['profiles']['high-recall']
        if setting59['selected'] not in art59['gates']: raise ValueError('Expected published v59 gate')
        previous=art59['gates'][setting59['selected']].predict_proba(x59)[:,1]
        sparse=score_batches(art60['core'],rows); neural=neural_predict(cnn60,rows)
        setting60=art60['profiles']['high-recall']
        x60=stack_features(rows,sparse,neural,previous)
        scores=candidate_score(setting60['selected'],art60['gates']['high-recall'],x60,sparse,neural,previous)
        ix=np.unique(np.linspace(0,len(rows)-1,12,dtype=int)); probe=[rows[i] for i in ix]
        observed=predict_texts(bundle,[r['text'] for r in probe],'high-recall',[r.get('old_text',r['text']) for r in probe])
        if not np.allclose(observed,scores[ix],rtol=1e-6,atol=1e-7): raise ValueError('Cached base scores differ from frozen runtime')
        np.save(out/f'{split}-base.npy',scores.astype(np.float64))
        np.save(out/f'{split}-base-components.npy',np.column_stack([sparse,neural,previous]))
    identifiers={s:hashlib.sha256('\n'.join(r['identity'] for r in rows).encode()).hexdigest() for s,rows in rows_by_split.items()}
    result=dict(viewVersion=VIEW_VERSION,encoderRevision=REVISION,priorManifestSha256=provenance['manifestSha256'],
                trainPoolCount=len(rows_by_split['train']),identityHashes=identifiers,
                baseModelSha256=file_sha('.cache/v60-model/model.joblib'),
                newTestRead=False,files={p.name:file_sha(p) for p in out.glob('*.npy')})
    (out/'manifest.json').write_text(json.dumps(result,indent=2)); print('DEVELOPMENT FEATURES FROZEN',json.dumps(result),flush=True)


if __name__=='__main__':main()
