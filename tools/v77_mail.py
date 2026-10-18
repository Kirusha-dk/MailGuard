"""CPU-only RU/EN text classifier, MIME reader and offline .eml evaluator.

The decision score is an SVM margin, NOT a probability. Public-corpus results
do not establish performance on arbitrary mail or ordinary Russian email.
"""
import argparse
import csv
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from email import policy
from email.parser import BytesParser
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

VERSION = 'v77-ru-en-cpu'
DIMENSIONS = 2**19
PROFILES = {'low-fp': .001, 'balanced': .005}
SPACE = re.compile(r'\s+')
URL = re.compile(r'https?://\S+|www\.\S+', re.I)
ADDRESS = re.compile(r'[\w.+-]+@[\w.-]+\.[\w.-]+', re.U)
TAGS = re.compile(r'<[^>]{0,2000}>')
SCRIPTS = re.compile(r'<(script|style)\b[^>]*>.*?</\1\s*>', re.I | re.S)
LETTERS = re.compile(r'[^\W\d_]', re.U)
CYRILLIC = re.compile(r'[а-яё]', re.I)
LATIN = re.compile(r'[a-z]', re.I)


def digest(value):
    return hashlib.sha256(value.encode('utf-8', 'replace')).hexdigest()


def clean(subject, body):
    """Use content only: never use publisher IDs, labels or spam verdict headers."""
    body = str(body or '')
    # Several public CSVs store RFC822 headers inside their body field.
    head, separator, tail = body.replace('\r\n', '\n').partition('\n\n')
    header_lines = re.findall(r'(?im)^(?:received|from|to|subject|return-path|date|message-id|mime-version|content-type|x-[\w-]+):', head)
    if separator and len(header_lines) >= 2:
        body = tail
    text = unicodedata.normalize('NFKC', str(subject or '') + '\n' + body).casefold()
    text = SCRIPTS.sub(' ', text)
    text = html.unescape(TAGS.sub(' ', text))
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]', '', text)
    text = URL.sub(' urltoken ', text)
    text = ADDRESS.sub(' emailtoken ', text)
    text = re.sub(r'\d+', ' numbertoken ', text)
    return SPACE.sub(' ', text).strip()


def language(text):
    letters = len(LETTERS.findall(text))
    if letters < 15:
        return None
    if len(CYRILLIC.findall(text)) / letters >= .70:
        return 'ru'
    if len(LATIN.findall(text)) / letters >= .90:
        return 'en'
    return None


def family(text):
    # Campaign-prefix grouping, independent of source or language identifiers.
    return digest(text[:200])


def content_view(text, kind):
    # Include the end of long messages; never silently use just 1,600 characters.
    head, tail = (8000, 2000) if kind == 'word' else (3000, 1000)
    return text if len(text) <= head + tail else text[:head] + ' tailtoken ' + text[-tail:]


def vectorizer(kind):
    if kind not in ('word', 'char'):
        raise ValueError('Unknown expert: ' + kind)
    return HashingVectorizer(analyzer='word' if kind == 'word' else 'char',
        ngram_range=(1, 2) if kind == 'word' else (3, 5), n_features=DIMENSIONS,
        alternate_sign=False, norm=None, binary=True, lowercase=False, dtype=np.float32)


def expert_scores(expert, texts):
    result = []
    hv = vectorizer(expert['kind'])
    for start in range(0, len(texts), 128):
        x = hv.transform(content_view(t, expert['kind']) for t in texts[start:start + 128])
        x = normalize(x.multiply(expert['ratio']).tocsr(), copy=False)
        result.extend(expert['model'].decision_function(x))
    return np.asarray(result, dtype=np.float64)


def candidates(experts, texts):
    word = expert_scores(experts['word'], texts)
    char = expert_scores(experts['char'], texts)
    return {'word': word, 'char': char, 'mean': (word + char) / 2}


def measure(values, truth, cutoff):
    values, truth = np.asarray(values), np.asarray(truth)
    if values.shape != truth.shape or not np.isfinite(values).all():
        raise ValueError('Invalid score/label arrays')
    if not set(np.unique(truth)) <= {0, 1}:
        raise ValueError('Binary labels required')
    yes = values >= cutoff
    tp = int(np.sum(yes & (truth == 1))); fn = int(np.sum(~yes & (truth == 1)))
    fp = int(np.sum(yes & (truth == 0))); tn = int(np.sum(~yes & (truth == 0)))
    return {'n': len(truth), 'spam': tp + fn, 'ham': fp + tn, 'tp': tp, 'fn': fn,
        'fp': fp, 'tn': tn, 'recall': tp / (tp + fn) if tp + fn else None,
        'fpr': fp / (fp + tn) if fp + tn else None,
        'accuracy': (tp + tn) / len(truth) if len(truth) else None}


def predict(bundle, texts, profile='low-fp'):
    if bundle.get('version') != VERSION or profile not in PROFILES:
        raise ValueError('Unsupported model/profile')
    result = [None] * len(texts)
    for lang in ('ru', 'en'):
        ix = [i for i, t in enumerate(texts) if language(t) == lang]
        if not ix:
            continue
        cfg = bundle['policies'][lang][profile]
        values = candidates(bundle['experts'][lang], [texts[i] for i in ix])[cfg['candidate']]
        for i, value in zip(ix, values):
            result[i] = {'language': lang, 'score': float(value), 'cutoff': cfg['cutoff'],
                         'spam': bool(value >= cfg['cutoff'])}
    return result  # Unsupported/short text explicitly abstains; never becomes HAM.


def decode_part(part):
    raw = part.get_payload(decode=True)
    if raw is None:
        value = part.get_payload()
        return value if isinstance(value, str) else ''
    declared = part.get_content_charset()
    if declared:
        try:
            return raw.decode(declared, 'strict')
        except (LookupError, UnicodeError):
            pass
    try:
        return raw.decode('utf-8', 'strict')
    except UnicodeError:
        pass
    # Recover undeclared Russian mail, rather than decoding Cyrillic as cp1252.
    recovered = []
    for encoding in ('cp1251', 'koi8-r'):
        candidate = raw.decode(encoding, 'replace')
        if language(candidate) == 'ru':
            lower = candidate.casefold()
            common = ('ст', 'но', 'то', 'на', 'ен', 'ов', 'ни', 'ра', 'во', 'ко', 'пр', 'ро', 'по', 'го', 'ос', 'ре', 'ер', 'ал', 'ли')
            recovered.append((sum(lower.count(pair) for pair in common), candidate))
    if recovered:
        return max(recovered, key=lambda item: item[0])[1]
    return raw.decode('cp1252', 'replace')


def parse_eml(raw):
    message = BytesParser(policy=policy.default).parsebytes(raw)
    parts = {'text/plain': [], 'text/html': []}
    for part in message.walk():
        typ = part.get_content_type()
        if typ in parts and part.get_content_disposition() != 'attachment':
            parts[typ].append(decode_part(part))
    body = '\n'.join(parts['text/plain'] or parts['text/html'])
    return clean(str(message.get('subject') or ''), body)


def main(predictor=None, version=None, allowed_languages=('ru', 'en'), description=None):
    predictor = predictor or predict
    version = version or VERSION
    p = argparse.ArgumentParser(description=description or __doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--spam', required=True)
    p.add_argument('--ham', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--profile', choices=PROFILES, default='low-fp')
    args = p.parse_args()
    roots = [Path(args.ham).resolve(), Path(args.spam).resolve()]
    if any(not root.is_dir() for root in roots) or roots[0] == roots[1] or any(a in b.parents for a in roots for b in roots if a != b):
        p.error('Use separate existing HAM and SPAM folders')
    out = Path(args.output).resolve()
    if any(root == out or root in out.parents for root in roots):
        p.error('Output must be outside input folders')
    files = [(path, y) for y, root in enumerate(roots) for path in sorted(root.rglob('*'))
             if path.is_file() and path.suffix.lower() == '.eml']
    if any(not any(y == label for _, y in files) for label in (0, 1)):
        p.error('Each input folder must contain .eml files')
    out.mkdir(parents=True, exist_ok=False)
    bundle = joblib.load(args.model)
    values, labels, by_lang = [], [], {lang: ([], []) for lang in allowed_languages}
    failures = Counter(); seen = {}; conflicts = set(); records = []
    for path, y in files:
        try:
            if path.stat().st_size > 20_000_000:
                raise ValueError('oversized message')
            text = parse_eml(path.read_bytes())
            if language(text) not in allowed_languages:
                failures['unsupported_or_short'] += 1
                continue
            identity = digest(text)
            if identity in seen:
                failures['duplicates'] += 1
                if seen[identity] != y:
                    conflicts.add(identity)
                continue
            seen[identity] = y
            records.append((path, y, identity, text))
        except Exception as exc:
            failures['read_errors'] += 1
            print('Cannot read:', path.name, type(exc).__name__)
    records = [r for r in records if r[2] not in conflicts]
    failures['conflicting_identities'] = len(conflicts)
    with (out / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['identity', 'label', 'language', 'score', 'cutoff', 'predicted'])
        for offset in range(0, len(records), 128):
            batch = records[offset:offset + 128]
            for row, result in zip(batch, predictor(bundle, [r[3] for r in batch], args.profile)):
                _, y, identity, _ = row
                value = result['score'] - result['cutoff']
                values.append(value); labels.append(y)
                by_lang[result['language']][0].append(value)
                by_lang[result['language']][1].append(y)
                writer.writerow([identity, y, result['language'], result['score'], result['cutoff'], int(result['spam'])])
    summary = {'version': version, 'profile': args.profile,
        'model_sha256': hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        'files_found': len(files), 'excluded': dict(failures), 'metrics': measure(values, labels, 0),
        'by_language': {k: measure(*v, 0) for k, v in by_lang.items()},
        'complete_10000_ru_en_email_test': len(values) == 10000 and not failures['read_errors']}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
