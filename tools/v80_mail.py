"""English semantic/phishing ensemble; profiles are calibration targets only."""
from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import normalize as normalize_sparse

import v77_mail as core
import v79_mail as base_runtime
from v79_data import normalize
from v80_encoder import Encoder, file_sha
from v80_metrics import summary_extra

VERSION = 'v80-english-phishing-semantic'
BASE_SHA = 'f331dc14cb289a5e27cc96904aad519cc07bc2a400e008ea71acffd200621d72'
PROFILES = {'ten-ppm': .00001, 'strict': .0001, 'target-80': .002,
            'balanced': .005, 'recall-98': None}
FAST_GATE = -.5
LEXICAL_CANDIDATES = frozenset(('v79-word', 'v79-char', 'v79-tfidf', 'v79-blend',
                                'phish-word', 'phish-mean', 'fast-or'))


def attach(bundle, args, *, load_encoder=True):
    if bundle.get('version') != VERSION:
        raise ValueError('Unsupported v80 model')
    root = Path(args.model).resolve().parent
    if file_sha(root / 'base/model.joblib') != BASE_SHA:
        raise ValueError('Unexpected v79 base model')
    bundle['_base'] = joblib.load(root / 'base/model.joblib')
    base_runtime.attach_language_model(bundle['_base'])
    bundle['_encoder_root'] = root / 'encoder'
    if load_encoder:
        bundle['_encoder'] = Encoder(root / 'encoder')


def attach_cli(bundle, args):
    # Research tools and v81 retain the historical eager attach contract.
    lexical = bundle.get('policies', {}).get(args.profile, {}).get('candidate') in LEXICAL_CANDIDATES
    attach(bundle, args, load_encoder=not lexical)


def supported(text, bundle):
    return base_runtime.supported(text, bundle['_base'])


def fast_features(bundle, texts):
    result = np.empty((len(texts), 5), dtype=np.float64)
    old = bundle['_base']['experts']
    # Both NB experts use identical hashed counts. Construct each word/char
    # matrix once and apply the two independently learned ratios to it.
    for kind, columns in (('word', (0, 3)), ('char', (1, 4))):
        vectorizer = core.vectorizer(kind)
        experts = (old[kind], bundle['experts']['phish-' + kind])
        for start in range(0, len(texts), 128):
            batch = texts[start:start + 128]
            raw = vectorizer.transform(core.content_view(t, kind) for t in batch)
            for column, expert in zip(columns, experts):
                weighted = normalize_sparse(raw.multiply(expert['ratio']).tocsr(), copy=False)
                result[start:start + len(batch), column] = expert['model'].decision_function(weighted)
    expert = old['tfidf']
    for start in range(0, len(texts), 128):
        batch = texts[start:start + 128]
        raw = expert['vectorizer'].transform(core.content_view(t, 'word') for t in batch)
        result[start:start + len(batch), 2] = expert['model'].decision_function(raw)
    return result


def needs_neural(fast):
    return np.max(fast, axis=1) >= FAST_GATE


def embedding_features(bundle, texts, fast):
    result = np.zeros((len(texts), 384), dtype=np.float32)
    mask = needs_neural(fast)
    if mask.any():
        if '_encoder' not in bundle:
            bundle['_encoder'] = Encoder(bundle['_encoder_root'])
        result[mask] = bundle['_encoder'].encode([text for text, yes in zip(texts, mask) if yes])
    return result


def feature_matrix(bundle, texts, fast=None, embeddings=None):
    fast = fast_features(bundle, texts) if fast is None else fast
    embeddings = embedding_features(bundle, texts, fast) if embeddings is None else embeddings
    mask = needs_neural(fast)
    semantic = np.zeros((len(texts), 2), dtype=np.float64)
    for index, name in enumerate(('semantic', 'semantic-phish')):
        if mask.any():
            semantic[mask, index] = bundle['experts'][name].decision_function(embeddings[mask])
    # No corpus/source/year identifiers reach any gate.
    return np.column_stack([fast, semantic, mask.astype(float)])


def lexical_scores(fast):
    word, char, tfidf, pw, pc = fast.T
    return {'v79-word': word, 'v79-char': char, 'v79-tfidf': tfidf,
            'v79-blend': .25 * (word + char) + .5 * tfidf,
            'phish-word': pw, 'phish-mean': .5 * (pw + pc),
            'fast-or': np.maximum(tfidf, .5 * (pw + pc))}


def candidate_scores(bundle, features):
    semantic, semantic_phish, mask = features[:, 5:].T
    result = lexical_scores(features[:, :5])
    result.update(semantic=np.where(mask, semantic, -1e6),
                  **{'semantic-phish': np.where(mask, semantic_phish, -1e6)})
    for name, model in bundle.get('gates', {}).items():
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(features)
        else:
            scores = model.predict_proba(features)[:, 1]
        result[name] = scores
    return result


def predict(bundle, texts, profile='balanced'):
    if bundle.get('version') != VERSION or profile not in PROFILES:
        raise ValueError('Unsupported v80 bundle/profile')
    texts = [normalize(text) for text in texts]
    selected = [i for i, text in enumerate(texts) if supported(text, bundle)]
    results = [None] * len(texts)
    if selected:
        chosen = [texts[i] for i in selected]
        fast = fast_features(bundle, chosen)
        policy = bundle['policies'][profile]
        lexical = lexical_scores(fast)
        if policy['candidate'] in lexical:
            scores = lexical[policy['candidate']]
        else:
            features = feature_matrix(bundle, chosen, fast=fast)
            scores = candidate_scores(bundle, features)[policy['candidate']]
        for index, score in zip(selected, scores):
            results[index] = {'language': 'en', 'score': float(score),
                              'cutoff': policy['cutoff'], 'spam': bool(score >= policy['cutoff'])}
    return results


if __name__ == '__main__':
    core.main(predictor=predict, version=VERSION, allowed_languages=('en',),
              prepare_bundle=attach_cli, text_supported=supported, profile_choices=PROFILES,
              default_profile='balanced', identity_fn=base_runtime.canonical_identity,
              description=__doc__, summary_extra=summary_extra)
