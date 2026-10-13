"""Frozen script-specific NB margins; identical canonical views in fit and inference."""
from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from train_v55_multilingual import views, transformed
from v61_semantic import script

FEATURES=2**20+2**20+2**18


def features(rows):
    focus=[r['focus'] for r in rows]; context=[r['context'] for r in rows]
    def char(texts,bounds):
        return HashingVectorizer(analyzer='char',ngram_range=bounds,n_features=2**20,
                                 alternate_sign=False,norm=None,dtype=np.float32).transform(texts)
    return hstack([char(focus,(2,5)),char(context,(3,5)),
                   HashingVectorizer(ngram_range=(1,2),n_features=2**18,alternate_sign=False,
                                     norm=None,dtype=np.float32).transform(context)],format='csr')


def load_bundle(root):
    artifact=joblib.load(Path(root)/'model.joblib')
    if artifact.get('version')!='v62-script-nb-margin': raise ValueError('Expected v62 bundle')
    return artifact


def score_rows(artifact,rows,profile='conservative',batch_size=128):
    setting=artifact['profiles'][profile]; scores=np.empty(len(rows),dtype=np.float64)
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size]; raw=features(batch)
        for name in ('latin','cjk','other'):
            ix=np.array([i for i,r in enumerate(batch) if script(r)==name],dtype=int)
            if not len(ix):continue
            selected=setting['scripts'][name]; scope=selected['scope']
            core=artifact['cores'][scope][selected['model']]
            x=transformed(raw[ix],artifact['idf'][scope],artifact['ratio'][scope])
            scores[start+ix]=core.decision_function(x)-selected['cutoff']
    return scores


def predict_texts(bundle,texts,profile='conservative',original_texts=None):
    rows=[dict(zip(('focus','context'),views(t))) for t in texts]
    return score_rows(bundle,rows,profile)
