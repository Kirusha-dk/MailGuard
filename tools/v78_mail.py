"""English-only CPU MailGuard: word/character NBSVM and TF-IDF experts.

SVM decision scores are margins, not probabilities. Non-English mail abstains.
"""
import numpy as np
import v77_mail as core
from v78_data import normalize

VERSION = 'v78-en-trusted-source-balance'
PROFILES = core.PROFILES


def candidates(bundle, texts):
    word = core.expert_scores(bundle['experts']['word'], texts)
    char = core.expert_scores(bundle['experts']['char'], texts)
    tfidf = []
    expert = bundle['experts']['tfidf']
    for offset in range(0, len(texts), 128):
        x = expert['vectorizer'].transform(core.content_view(t, 'word') for t in texts[offset:offset + 128])
        tfidf.extend(expert['model'].decision_function(x))
    tfidf = np.asarray(tfidf)
    mean = .5 * (word + char)
    return {'word': word, 'char': char, 'nb-mean': mean,
            'tfidf': tfidf, 'blend': .5 * (mean + tfidf),
            'word-tfidf': .5 * (word + tfidf)}


def predict(bundle, texts, profile='low-fp'):
    if bundle.get('version') != VERSION or profile not in PROFILES:
        raise ValueError('Unsupported English model/profile')
    texts = [normalize(text) for text in texts]
    result = [None] * len(texts)
    ix = [i for i, text in enumerate(texts) if core.language(text) == 'en']
    if not ix:
        return result
    policy = bundle['policies'][profile]
    values = candidates(bundle, [texts[i] for i in ix])[policy['candidate']]
    for i, value in zip(ix, values):
        result[i] = {'language': 'en', 'score': float(value), 'cutoff': policy['cutoff'],
                     'spam': bool(value >= policy['cutoff'])}
    return result


if __name__ == '__main__':
    core.main(predictor=predict, version=VERSION, allowed_languages=('en',), description=__doc__)
