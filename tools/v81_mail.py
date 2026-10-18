"""Constrained false-positive rescue on top of the fixed v80 high-recall policy."""
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

import v77_mail as core
import v80_mail as baseline
from v79_data import normalize
from v80_encoder import file_sha
from v80_metrics import summary_extra

VERSION = 'v81-english-constrained-fp-rescue'
BASELINE_SHA = '6aa2e92b2e1ccfd764223eeea5d1c792e5ae15548ad60767be93023d690aa5c9'
PROFILES = ('v80-recall-98', 'fp-rescue')


def attach(bundle, args):
    if bundle.get('version') != VERSION:
        raise ValueError('Unsupported v81 model')
    model = Path(args.model).resolve().parent / 'baseline/model.joblib'
    if file_sha(model) != BASELINE_SHA:
        raise ValueError('Unexpected v80 baseline')
    bundle['_baseline'] = joblib.load(model)
    baseline.attach(bundle['_baseline'], SimpleNamespace(model=model))


def supported(text, bundle):
    return baseline.supported(text, bundle['_baseline'])


def base_scores(fast):
    return np.maximum(fast[:, 2], .5 * (fast[:, 3] + fast[:, 4]))


def regions(fast, partition):
    if partition == 'global':
        return np.zeros(len(fast), dtype=int)
    if partition == 'dominance':
        return (fast[:, 2] >= .5 * (fast[:, 3] + fast[:, 4])).astype(int)
    if partition == 'margin':
        return np.searchsorted([.5, 1.], base_scores(fast), side='right')
    raise ValueError('Unknown rescue partition')


def lexical_features(texts):
    values = []
    for text in texts:
        words = text.split(); n = max(len(words), 1); length = max(len(text), 1)
        values.append([np.log1p(length), np.log1p(n), len(set(words)) / n,
                       sum(w == 'urltoken' for w in words) / n,
                       sum(w == 'emailtoken' for w in words) / n,
                       sum(w == 'numbertoken' for w in words) / n,
                       sum(c in '!?$' for c in text) / length])
    return np.asarray(values, dtype=float).reshape(-1, 7)


def model_scores(model, features):
    if hasattr(model, 'decision_function'):
        result = model.decision_function(features)
    else:
        p = np.clip(model.predict_proba(features)[:, 1], 1e-12, 1 - 1e-12)
        result = np.log(p) - np.log1p(-p)
    result = np.asarray(result, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite rescue scores')
    return result


def candidate_scores(bundle, name, features, embeddings, lexical):
    old = bundle['_baseline']
    if name.startswith('v80-gate:'):
        return model_scores(old['gates'][name.split(':', 1)[1]], features)
    if name == 'v80-semantic':
        return features[:, 5]
    if name == 'v80-semantic-phish':
        return features[:, 6]
    if name == 'v79-word':
        return features[:, 0]
    if name == 'v79-char':
        return features[:, 1]
    if name == 'v79-blend':
        return .25 * (features[:, 0] + features[:, 1]) + .5 * features[:, 2]
    expert = bundle['experts'][name]
    if expert['input'] == 'neighbors':
        # Three distinct TRAIN HAM emails are needed; the closest spam anchor
        # counters a similar legitimate template. Only TRAIN supplies anchors.
        values = []
        for start in range(0, len(embeddings), 256):
            x = embeddings[start:start + 256]
            ham = np.partition(x @ expert['ham'].T, -3, axis=1)[:, -3]
            spam = np.max(x @ expert['spam'].T, axis=1)
            ham = np.clip(ham, -1., 1 - 1e-6); spam = np.clip(spam, -1., 1 - 1e-6)
            values.append(np.log1p(-ham) - np.log1p(-spam))
        return np.concatenate(values) if values else np.empty(0)
    x = embeddings if expert['input'] == 'embedding' else np.column_stack([features, lexical])
    return model_scores(expert['model'], x)


def apply_policy(fast, scores, policy):
    original = base_scores(fast) >= policy['baseline_cutoff']
    if not policy['enabled']:
        return original.copy()
    values = np.asarray(scores)
    if len(values) != len(fast) or not np.isfinite(values).all():
        raise ValueError('Invalid rescue scores')
    group = regions(fast, policy['partition'])
    veto = np.zeros(len(fast), dtype=bool)
    for region, threshold in policy['thresholds'].items():
        veto |= (group == int(region)) & (values < threshold)
    # Monotone rescue: it can never introduce a new spam decision/FP.
    return original & ~veto


def decision_margins(fast, scores, policy):
    """A nonnegative final margin means spam, including the rescue stage."""
    result = base_scores(fast) - policy['baseline_cutoff']
    if policy['enabled']:
        group = regions(fast, policy['partition'])
        for region, threshold in policy['thresholds'].items():
            selected = group == int(region)
            result[selected] = np.minimum(result[selected], scores[selected] - threshold)
    return result


def predict(bundle, texts, profile='v80-recall-98'):
    if bundle.get('version') != VERSION or profile not in PROFILES:
        raise ValueError('Unsupported v81 bundle/profile')
    texts = [normalize(t) for t in texts]
    indices = [i for i, t in enumerate(texts) if supported(t, bundle)]
    result = [None] * len(texts)
    if not indices:
        return result
    chosen = [texts[i] for i in indices]
    old = bundle['_baseline']; policy = bundle['policy']
    fast = baseline.fast_features(old, chosen)
    values = base_scores(fast)
    predicted = values >= policy['baseline_cutoff']
    margins = values - policy['baseline_cutoff']
    if profile == 'fp-rescue' and policy['enabled'] and predicted.any():
        pos = np.flatnonzero(predicted)
        positive_texts = [chosen[i] for i in pos]
        embedding = old['_encoder'].encode(positive_texts)
        features = baseline.feature_matrix(old, positive_texts, fast[pos], embedding)
        secondary = candidate_scores(bundle, policy['candidate'], features, embedding,
                                     lexical_features(positive_texts))
        predicted[pos] = apply_policy(fast[pos], secondary, policy)
        margins[pos] = decision_margins(fast[pos], secondary, policy)
    if not np.array_equal(margins >= 0, predicted):
        raise AssertionError('Final score and decision disagree')
    for i, score, original_score, spam in zip(indices, margins, values, predicted):
        result[i] = dict(language='en', score=float(score), cutoff=0., spam=bool(spam),
                         baseline_score=float(original_score), baseline_cutoff=policy['baseline_cutoff'],
                         rescued=bool(original_score >= policy['baseline_cutoff'] and not spam))
    return result


if __name__ == '__main__':
    core.main(predictor=predict, version=VERSION, allowed_languages=('en',),
              prepare_bundle=attach, text_supported=supported, profile_choices=PROFILES,
              default_profile='v80-recall-98', identity_fn=baseline.base_runtime.canonical_identity,
              description=__doc__, summary_extra=summary_extra)
