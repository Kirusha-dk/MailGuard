"""TRAIN-only word experts and frozen character experts with development-only gates."""
import math
import os
from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

BRANCHES = ('subject','head','body')
DIMENSIONS = {'subject':2**18+2**16,'head':2**18,'body':2**20}


def parts(row):
    # Both canonical views begin with the identical subject. Their following
    # markers differ, so words such as "body" inside the subject are harmless.
    common = os.path.commonprefix((row['focus'],row['context']))
    if not common.startswith('subject ') or not row['context'][len(common):].startswith('body '):
        raise ValueError('Expected canonical v55 views')
    subject = common[len('subject '):].rstrip()
    remaining = row['focus'][len(common):]
    hosts,sep,head = remaining.removeprefix('hosts ').partition(' start ')
    if not sep:
        raise ValueError('Canonical head marker missing')
    return subject,head,row['context'][len(common)+len('body '):],hosts


def word_features(rows,branch):
    ix = BRANCHES.index(branch)
    texts = [parts(r)[ix] for r in rows]
    words = HashingVectorizer(ngram_range=(1,2),n_features=2**16 if branch=='subject' else DIMENSIONS[branch],
                              alternate_sign=False,norm=None,dtype=np.float32).transform(texts)
    if branch != 'subject':
        return words
    chars = HashingVectorizer(analyzer='char',ngram_range=(3,5),n_features=2**18,
                              alternate_sign=False,norm=None,dtype=np.float32).transform(texts)
    return hstack([chars,words],format='csr')


def transform(raw,idf,ratio):
    x = raw.copy()
    x.data = np.log(x.data)+1.
    return normalize(x.multiply(idf).multiply(ratio).tocsr(),norm='l2',copy=False)


def structure(rows):
    from v61_semantic import script
    result = []
    for r in rows:
        subject,head,body,hosts = parts(r)
        language = script(r)
        result.append([math.log1p(len(subject)),math.log1p(len(head)),math.log1p(len(body)),
                       len(hosts.split())/24.,body.count('urltoken')/max(1,len(body.split())),
                       body.count('emailtoken')/max(1,len(body.split())),
                       body.count('numbertoken')/max(1,len(body.split())),
                       *[float(language==s) for s in ('latin','cjk','other')]])
    return np.asarray(result,dtype=np.float64)


def character_evidence(parent,rows):
    from v62_runtime import features
    from train_v55_multilingual import transformed
    raw = features(rows)
    values = []
    for scope in sorted(parent['cores']):
        x = transformed(raw,parent['idf'][scope],parent['ratio'][scope])
        for name in sorted(parent['cores'][scope]):
            values.append(parent['cores'][scope][name].decision_function(x))
    return np.column_stack(values)


def evidence(artifact,parent,rows):
    values = [character_evidence(parent,rows)]
    for branch in BRANCHES:
        x = transform(word_features(rows,branch),artifact['idf'][branch],artifact['ratio'][branch])
        values.append(np.column_stack([artifact['cores'][branch][k].decision_function(x)
                                      for k in sorted(artifact['cores'][branch])]))
    margins = np.column_stack(values)
    # Bounded monotone input scaling keeps linear gates well conditioned.
    margins = np.sign(margins)*np.log1p(np.abs(margins))
    result = np.round(np.column_stack([margins,structure(rows)]),6)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite evidence')
    return result


def load_bundle(root):
    from train_v55_multilingual import file_sha
    from v62_runtime import load_bundle as load62
    root = Path(root)
    artifact = joblib.load(root/'model.joblib')
    if artifact.get('version') != 'v64-word-character-gate':
        raise ValueError('Expected v64')
    for name,digest in artifact['components'].items():
        if file_sha(root/name) != digest:
            raise ValueError('Changed component: '+name)
    return artifact,load62(root/'v62')


def score_rows(bundle,rows,batch_size=128):
    from v61_semantic import script
    artifact,parent = bundle
    scores = {p:np.empty(len(rows),dtype=np.float64) for p in artifact['profiles']}
    for start in range(0,len(rows),batch_size):
        batch = rows[start:start+batch_size]
        x = evidence(artifact,parent,batch)
        for p,cfg in artifact['profiles'].items():
            raw = artifact['gates'][cfg['selected']].predict_proba(x)[:,1]
            scores[p][start:start+len(batch)] = raw-np.asarray([cfg['scriptCutoffs'][script(r)] for r in batch])
    return scores


def predict_texts(bundle,texts,profile='conservative',original_texts=None):
    from train_v55_multilingual import views
    rows = [dict(zip(('focus','context'),views(t))) for t in texts]
    return score_rows(bundle,rows)[profile]
