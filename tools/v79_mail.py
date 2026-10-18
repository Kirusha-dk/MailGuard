"""English-only CPU model with bundled fastText language identification."""
import hashlib
import tempfile
import fasttext
import v77_mail as core
import v78_mail as previous

VERSION = 'v79-en-100k-fixed-blocks'
PROFILES = {'low-fp': .001, 'target-80': .002, 'balanced': .005}
LID_SHA = '8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83'


def attach_language_model(bundle, args=None):
    if bundle.get('version') != VERSION:
        raise ValueError('Unsupported v79 model bundle')
    raw = bundle['language_model_bytes']
    if hashlib.sha256(raw).hexdigest() != LID_SHA:
        raise ValueError('Language identifier checksum mismatch')
    with tempfile.NamedTemporaryFile(suffix='.ftz') as temporary:
        temporary.write(raw); temporary.flush()
        bundle['_language_model'] = fasttext.load_model(temporary.name)


def supported(text, bundle):
    from v79_data import english_prediction, normalize
    if '_language_model' not in bundle:
        attach_language_model(bundle)
    return english_prediction(bundle['_language_model'], normalize(text))[0]


def candidates(bundle, texts):
    return previous.candidates(bundle, texts)


def canonical_identity(text):
    from v79_data import normalize
    from v78_data import identifiers
    return identifiers(normalize(text))[0]


def predict(bundle, texts, profile='target-80'):
    from v79_data import normalize
    if bundle.get('version') != VERSION or profile not in PROFILES:
        raise ValueError('Unsupported v79 bundle/profile')
    texts = [normalize(text) for text in texts]
    result = [None] * len(texts)
    ix = [i for i,text in enumerate(texts) if supported(text, bundle)]
    if not ix:
        return result
    cfg = bundle['policies'][profile]
    values = candidates(bundle, [texts[i] for i in ix])[cfg['candidate']]
    for i,value in zip(ix,values):
        result[i] = {'language':'en','score':float(value),'cutoff':cfg['cutoff'],'spam':bool(value>=cfg['cutoff'])}
    return result


if __name__ == '__main__':
    core.main(predictor=predict, version=VERSION, allowed_languages=('en',), description=__doc__,
              prepare_bundle=attach_language_model, text_supported=supported,
              profile_choices=PROFILES, default_profile='target-80', identity_fn=canonical_identity)
