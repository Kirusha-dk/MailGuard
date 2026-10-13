"""Semantic v61 inference; script-specific cutoffs are learned only on development HAM."""
from pathlib import Path
import joblib
import numpy as np
from train_v55_multilingual import views,file_sha
from train_v56_consensus import gate_features
from v60_runtime import load_bundle as load60,predict_texts as predict60
from v61_semantic import Encoder,neighbours,script


def stack(rows,base,semantic,near):
    return np.column_stack([gate_features(rows,base,semantic,.9396340406399176),near])


def raw_score(name,gates,x,base,semantic):
    if name=='v60-only': return base
    if name=='semantic-only': return semantic
    return gates[name].predict_proba(x)[:,1].astype(np.float64)


def margins(rows,scores,cuts):
    return np.asarray(scores,dtype=np.float64)-np.array([cuts[script(r)] for r in rows],dtype=np.float64)


def load_bundle(root):
    root=Path(root); artifact=joblib.load(root/'model.joblib')
    if artifact.get('version')!='v61-semantic-neighbour-ensemble': raise ValueError('Expected v61 model')
    for filename,digest in artifact['components'].items():
        if file_sha(root/filename)!=digest: raise ValueError('Component changed: '+filename)
    return artifact,Encoder(root/'encoder'),load60(root/'v60')


def score_rows(artifact,rows,embeddings,base,profile):
    semantic=artifact['semanticCore'].predict_proba(embeddings)[:,1]
    near=neighbours(embeddings,artifact['anchors'],artifact['anchorLabels'])
    x=stack(rows,base,semantic,near); setting=artifact['profiles'][profile]
    scores=raw_score(setting['selected'],artifact['gates'][profile],x,base,semantic)
    return margins(rows,scores,setting['scriptCutoffs'])


def predict_texts(bundle,texts,profile='conservative',original_texts=None):
    artifact,encoder,base_bundle=bundle
    rows=[dict(zip(('focus','context'),views(t))) for t in texts]
    # Dynamic int8 activation scales can depend on other emails in a batch.
    # A fixed one-message query prevents the caller's batch composition from
    # moving a rare-FP decision across its calibrated boundary.
    embeddings=encoder.encode(rows,batch_size=1)
    base=predict60(base_bundle,texts,'high-recall',original_texts=original_texts)
    return score_rows(artifact,rows,embeddings,base,profile)
