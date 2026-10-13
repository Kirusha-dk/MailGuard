"""Joint semantic/character evidence, with a frozen development-only gate."""
from pathlib import Path
import joblib
import numpy as np
from train_v55_multilingual import views,file_sha,transformed,predict as sparse_predict
from v60_neural import predict as neural_predict
from v61_runtime import load_bundle as load61,stack as semantic_stack
from v61_semantic import neighbours,script
from v62_runtime import features,load_bundle as load62


def character_evidence(artifact,rows,batch_size=128):
    columns=[(g,k) for g in sorted(artifact['cores']) for k in sorted(artifact['cores'][g])]
    result=np.empty((len(rows),len(columns)),dtype=np.float64)
    for start in range(0,len(rows),batch_size):
        raw=features(rows[start:start+batch_size])
        for g in sorted(artifact['cores']):
            x=transformed(raw,artifact['idf'][g],artifact['ratio'][g])
            for j,(scope,k) in enumerate(columns):
                if scope==g:result[start:start+raw.shape[0],j]=artifact['cores'][g][k].decision_function(x)
    return result


def joint_features(rows,base,semantic,near,characters):
    # Remove insignificant numerical noise before fitting and inference.
    return np.round(np.column_stack([semantic_stack(rows,base,semantic,near),characters]),5)


def train_only_evidence(base_bundle,rows):
    artifact,cnn,_=base_bundle;core=artifact['core']
    sparse=sparse_predict(core['model'],rows,core['idf'],core['ratio'])
    neural=neural_predict(cnn,rows)
    return sparse,neural


def load_bundle(root):
    root=Path(root);artifact=joblib.load(root/'model.joblib')
    if artifact.get('version')!='v63-joint-character-semantic':raise ValueError('Expected v63')
    for name,digest in artifact['components'].items():
        if file_sha(root/name)!=digest:raise ValueError('Changed component: '+name)
    return artifact,load61(root/'v61'),load62(root/'v62')


def predict_texts(bundle,texts,profile='conservative',original_texts=None):
    artifact,semantic_bundle,character_bundle=bundle
    old,encoder,base_bundle=semantic_bundle
    rows=[dict(zip(('focus','context'),views(t))) for t in texts]
    e=encoder.encode(rows,batch_size=1)
    semantic=old['semanticCores'][profile].predict_proba(e)[:,1]
    near=neighbours(e,old['anchors'],old['anchorLabels'])
    base,neural=train_only_evidence(base_bundle,rows)
    chars=np.column_stack([character_evidence(character_bundle,rows),neural])
    x=joint_features(rows,base,semantic,near,chars)
    setting=artifact['profiles'][profile]
    scores=artifact['gates'][profile][setting['selected']].predict_proba(x)[:,1]
    return scores-np.array([setting['scriptCutoffs'][script(r)] for r in rows])
