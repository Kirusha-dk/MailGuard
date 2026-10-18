"""Validated feature caches. Parallel lexical scoring never uses corpus labels."""
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import v80_mail as baseline
from v79_data import fingerprint
from v80_encoder import file_sha
from train_v80_en import cache_embeddings

_worker_bundle = None
_worker_threads = None


def _initialize(model):
    global _worker_bundle, _worker_threads
    _worker_threads = threadpool_limits(limits=1)
    _worker_bundle = joblib.load(model)
    _worker_bundle['_base'] = joblib.load(Path(model).parent / 'base/model.joblib')


def _score(texts):
    return baseline.fast_features(_worker_bundle, texts)


def lexical_parallel(model, texts, workers=4):
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_initialize, initargs=(str(model),)) as pool:
        chunks = [texts[start:start + 512] for start in range(0, len(texts), 512)]
        return np.vstack(list(pool.map(_score, chunks))) if chunks else np.empty((0, 5))


def cached_fast(model, rows, path):
    h = hashlib.sha256()
    for row in rows:
        raw = row['text'].encode(); h.update(len(raw).to_bytes(8, 'big')); h.update(raw)
    code = Path(__file__).parent
    expected = dict(model_sha256=file_sha(model), fingerprint=fingerprint(rows),
                    text_sha256=h.hexdigest(), n=len(rows),
                    code_sha256={s: file_sha(code / s) for s in
                                 ('v81_features.py', 'v80_mail.py', 'v77_mail.py')})
    meta = path.with_suffix('.json')
    if path.exists():
        if json.loads(meta.read_text()) != expected:
            raise ValueError('Lexical cache does not match inputs/code/model')
        fast = np.load(path)
    else:
        print('Lexical scoring', path.name, len(rows), flush=True)
        fast = lexical_parallel(model, [r['text'] for r in rows])
        np.save(path, fast); meta.write_text(json.dumps(expected, indent=2))
    if fast.shape != (len(rows), 5) or not np.isfinite(fast).all():
        raise ValueError('Invalid lexical feature cache')
    return fast


def cached_neural(bundle, rows, fast, mask, path):
    embedding = cache_embeddings(bundle['_encoder'], rows, path, mask)
    features = baseline.feature_matrix(bundle, [r['text'] for r in rows], fast, embedding)
    return features, embedding
