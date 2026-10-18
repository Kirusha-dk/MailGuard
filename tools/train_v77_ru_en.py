"""Reproducible CPU baseline. Frozen public TEST is never training/calibration.

Russian Telegram training is explicitly a proxy; Russian email HAM validation
requires a separate labelled mail corpus. No claim about arbitrary 10k mail.
"""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import vstack
from datasets import load_dataset
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
import v77_mail as runtime

EMAIL_SOURCE = 'JinqiangDing/seven-phishing-email-datasets'
EMAIL_REVISION = 'deb4cc27dc2754c14cb157aace412727f77b41dd'
RU_SOURCE = 'alt-gnome/telegram-spam'
RU_REVISION = 'af298cd351ce1d257ae789571f649d4d94d2cdd6'
SEED = 'v77-frozen-official-test-2026-10-18'


def rank(text):
    return runtime.digest(SEED + ':' + text)


def row(subject, body, y, source, split):
    if type(y) is not int or y not in (0, 1):
        raise ValueError('Unexpected publisher label')
    text = runtime.clean(subject, body)
    lang = runtime.language(text)
    if lang is None or len(text) < 30:
        return None
    return dict(text=text, y=y, source=source, official_split=split, lang=lang,
                identity=runtime.digest(text), family=runtime.family(text))


def acquire(cache):
    if cache.is_file():
        with gzip.open(cache, 'rt') as stream:
            return json.load(stream)
    data = load_dataset(EMAIL_SOURCE, revision=EMAIL_REVISION)
    all_rows = []
    for split in ('train', 'test'):
        for value in data[split]:
            item = row(value['subject'], value['text'], value['label'], value['dataset_name'], split)
            if item:
                all_rows.append(item)
        print('Indexed email', split, len(all_rows), flush=True)
    data = load_dataset(RU_SOURCE, revision=RU_REVISION, split='train')
    for value in data:
        item = row('', value['text'], int(value['label']), 'RU_TELEGRAM_PROXY', 'proxy')
        if item and item['lang'] == 'ru':
            all_rows.append(item)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, 'wt') as stream:
        json.dump(all_rows, stream, ensure_ascii=False)
    return all_rows


def partition(rows, test_count=10000):
    # Remove both sides of contradictory identities, even across providers.
    labels = {}; conflicts = set()
    for r in rows:
        previous = labels.setdefault(r['identity'], r['y'])
        if previous != r['y']:
            conflicts.add(r['identity'])
    rows = [r for r in rows if r['identity'] not in conflicts]
    test_pool = {}
    for r in rows:
        if r['official_split'] == 'test':
            test_pool.setdefault(r['identity'], r)
    # Protect ALL official-test families, including unsampled rows, from fitting.
    test_families = {r['family'] for r in test_pool.values()}
    available_ru = sorted((r for r in test_pool.values() if r['lang'] == 'ru'), key=lambda r: rank(r['identity']))
    ru = available_ru[:min(1000, test_count // 2, len(available_ru))]
    en = sorted((r for r in test_pool.values() if r['lang'] == 'en'), key=lambda r: rank(r['identity']))
    test = ru + en[:test_count - len(ru)]
    if len(test) != test_count:
        raise ValueError('Insufficient distinct real email TEST rows')
    train = {'en': [], 'ru': []}; cal = {'en': [], 'ru': []}; seen = set()
    for r in rows:
        if r['official_split'] == 'test' or r['family'] in test_families or r['identity'] in seen:
            continue
        seen.add(r['identity'])
        bucket = int(rank(r['family'])[:12], 16) % 100
        (cal if bucket >= 85 else train)[r['lang']].append(r)
    for lang, cap in (('en', 70000), ('ru', 22000)):
        # Balance only TRAIN source/label influence through weights, not TEST.
        train[lang] = sorted(train[lang], key=lambda r: rank(r['identity']))[:cap]
        cal[lang] = sorted(cal[lang], key=lambda r: rank(r['identity']))[:12000]
        if {r['y'] for r in train[lang]} != {0, 1} or {r['y'] for r in cal[lang]} != {0, 1}:
            raise ValueError('Both classes required in TRAIN and CAL')
        a = {r['family'] for r in train[lang]}; b = {r['family'] for r in cal[lang]}
        if a & b or (a | b) & test_families:
            raise ValueError('TRAIN/CAL/TEST family overlap')
    return train, cal, test, len(conflicts)


def fit_expert(rows, kind):
    # HashingVectorizer's all-at-once construction can temporarily allocate
    # many Python objects. Bound both construction batches and char TRAIN size.
    if kind == 'char':
        rows = rows[:30000]
    y = np.array([r['y'] for r in rows], dtype=np.int8)
    counts = Counter((r['source'], r['y']) for r in rows)
    weights = np.array([min(1., 6000 / counts[r['source'], r['y']]) for r in rows])
    hv = runtime.vectorizer(kind)
    chunks = []
    for offset in range(0, len(rows), 256):
        chunks.append(hv.transform(runtime.content_view(r['text'], kind) for r in rows[offset:offset + 256]))
    x = vstack(chunks, format='csr')
    del chunks
    p = np.asarray(x[y == 1].multiply(weights[y == 1, None]).sum(axis=0)).ravel() + 2
    q = np.asarray(x[y == 0].multiply(weights[y == 0, None]).sum(axis=0)).ravel() + 2
    ratio = (np.log(p / p.sum()) - np.log(q / q.sum())).astype(np.float32)
    x = normalize(x.multiply(ratio).tocsr(), copy=False)
    model = LinearSVC(C=2., dual=True, max_iter=2500, tol=.001, random_state=77)
    model.fit(x, y, sample_weight=weights)
    print('Trained', kind, len(rows), 'rows', x.nnz, 'nonzeros', flush=True)
    return {'kind': kind, 'ratio': ratio, 'model': model, 'training_rows': len(rows)}


def choose(scores, labels, fpr):
    labels = np.asarray(labels)
    best = None
    for name, values in scores.items():
        ham = np.sort(values[labels == 0])[::-1]
        if len(ham) < 500:
            raise ValueError('Not enough CAL HAM')
        cap = int(len(ham) * fpr)
        cutoff = float(np.nextafter(ham[cap], np.inf))
        m = runtime.measure(values, labels, cutoff)
        key = (m['tp'], -m['fp'])
        if best is None or key > best[0]:
            best = (key, {'candidate': name, 'cutoff': cutoff, 'calibration': m,
                          'calibration_fpr_budget': fpr})
    return best[1]


def fingerprint(rows):
    return runtime.digest('\n'.join(sorted(r['identity'] + ':' + r['family'] for r in rows)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='reports/v77')
    p.add_argument('--cache', default='.cache/v77/rows.json.gz')
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    rows = acquire(Path(args.cache))
    train, cal, test, conflicts = partition(rows)
    del rows
    manifest = {'email_source': EMAIL_SOURCE, 'email_revision': EMAIL_REVISION,
        'russian_proxy_source': RU_SOURCE, 'russian_proxy_revision': RU_REVISION,
        'seed': SEED, 'excluded_conflicting_identities': conflicts,
        'split_fingerprints': {lang: {k: fingerprint(v[lang]) for k, v in [('train', train), ('cal', cal)]} for lang in ('en', 'ru')},
        'test_fingerprint': fingerprint(test), 'test_count': len(test),
        'train_counts': {k: len(v) for k, v in train.items()},
        'cal_counts': {k: len(v) for k, v in cal.items()},
        'test_language_counts': dict(Counter(r['lang'] for r in test)),
        'test_is_real_email': True, 'test_is_new_source': False,
        'test_used_in_older_project_versions_possible': True,
        'train_cal_test_prefix_family_disjoint': True,
        'russian_ham_email_validation': False}
    (out / 'frozen.json').write_text(json.dumps(manifest, indent=2))
    bundle = {'version': runtime.VERSION, 'experts': {}, 'policies': {}, 'manifest': manifest}
    for lang in ('en', 'ru'):
        bundle['experts'][lang] = {kind: fit_expert(train[lang], kind) for kind in ('word', 'char')}
        scores = runtime.candidates(bundle['experts'][lang], [r['text'] for r in cal[lang]])
        y = [r['y'] for r in cal[lang]]
        bundle['policies'][lang] = {profile: choose(scores, y, budget) for profile, budget in runtime.PROFILES.items()}
        print('CAL', lang, json.dumps(bundle['policies'][lang]), flush=True)
    modelpath = out / 'model.joblib'
    joblib.dump(bundle, modelpath, compress=3)
    sha = hashlib.sha256(modelpath.read_bytes()).hexdigest()
    print('MODEL FROZEN BEFORE TEST', sha, flush=True)
    all_values = {profile: ([], []) for profile in runtime.PROFILES}; by_language = {}
    for lang in ('en', 'ru'):
        group = [r for r in test if r['lang'] == lang]
        values = runtime.candidates(bundle['experts'][lang], [r['text'] for r in group])
        y = [r['y'] for r in group]
        by_language[lang] = {}
        for profile, cfg in bundle['policies'][lang].items():
            margins = values[cfg['candidate']] - cfg['cutoff']
            by_language[lang][profile] = runtime.measure(margins, y, 0)
            all_values[profile][0].extend(margins); all_values[profile][1].extend(y)
    if hashlib.sha256(modelpath.read_bytes()).hexdigest() != sha:
        raise ValueError('Model changed during TEST')
    report = {'version': runtime.VERSION, 'model_sha256': sha, 'manifest': manifest,
        'policies': bundle['policies'], 'test_by_language': by_language,
        'test_all': {k: runtime.measure(*v, 0) for k, v in all_values.items()},
        'arbitrary_10000_mail_guarantee': False,
        'russian_email_ham_fpr_validated': False,
        'warning': 'Historical public within-corpus heldout, not new customer/source mail. Russian HAM email absent; RU model trained on Telegram proxy plus scarce real email.'}
    (out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
