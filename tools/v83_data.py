"""Freeze outcome-blind RU/ES email partitions without importing a training library.

Only subject/body become features. The caller must supply original public email
labels and an explicit email source kind, plus a pinned language detector. This
module never relabels messages or treats chat/translated proxies as real mail.
"""
from collections import Counter
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re

from v77_mail import clean

LANGUAGES = frozenset(('ru', 'es'))
SPLIT_SEED = 'v83-ru-es-20261030'
MIN_LANGUAGE_PROBABILITY = .8
WORDS = re.compile(r'[^\W_]+', re.UNICODE)
PLACEHOLDERS = frozenset(('numbertoken', 'urltoken', 'emailtoken'))


def _language(language):
    if language not in LANGUAGES:
        raise ValueError('Expected Russian or Spanish language code')


def normalize(subject, body):
    if not isinstance(subject, str) or not isinstance(body, str):
        raise ValueError('Subject and body must be strings')
    return clean(subject, body)


def make_row(subject, body, y, source, language):
    _language(language)
    if type(y) is not int or y not in (0, 1):
        raise ValueError('Original binary email label is required')
    if not isinstance(source, str) or not source.strip():
        raise ValueError('Explicit email source is required')
    text = normalize(subject, body)
    words = WORDS.findall(text)
    content_words = [w for w in words if w not in PLACEHOLDERS and any(c.isalpha() for c in w)]
    if len(text) < 30 or len(content_words) < 10:
        return None
    identity = hashlib.sha256(text.encode('utf-8')).hexdigest()
    family = hashlib.sha256(' '.join(words[:45]).encode('utf-8')).hexdigest()
    return dict(identity=identity, family=family, text=text, y=y, source=source, language=language)


def fingerprint(rows):
    """Same length-delimited content fingerprint as v82_neural.fingerprint."""
    digest = hashlib.sha256()
    for row in rows:
        data = json.dumps({k: row[k] for k in ('identity', 'family', 'source', 'y', 'text')},
                          ensure_ascii=False, sort_keys=True).encode()
        digest.update(len(data).to_bytes(8, 'big'))
        digest.update(data)
    return digest.hexdigest()


def _partition(family):
    bucket = int(hashlib.sha256((SPLIT_SEED + ':' + family).encode()).hexdigest(), 16) % 100
    return 'train' if bucket < 70 else 'select' if bucket < 85 else 'test'


def prepare_rows(rows, language, language_predictor):
    """Return (partitions, aggregate stats); no output files or fitting.

    language_predictor(text) must return (ISO language code, probability). Every
    raw row requires kind='email' or source_kind='email'. Any explicit non-email
    kind raises instead of silently enabling a chat/proxy experiment.
    """
    _language(language)
    if not callable(language_predictor):
        raise TypeError('A callable pinned language detector is required')
    stats = Counter()
    unique, conflicts = {}, set()
    for raw in rows:
        stats['input_rows'] += 1
        if not isinstance(raw, dict):
            raise ValueError('Email source rows must be dictionaries')
        kinds = [raw[key] for key in ('kind', 'source_kind') if key in raw]
        if not kinds or any(kind != 'email' for kind in kinds):
            raise ValueError('Only explicitly identified original email sources are allowed')
        if 'language' in raw and raw['language'] != language:
            raise ValueError('Source language annotation does not match requested language')
        row = make_row(raw.get('subject', ''), raw.get('body', ''), raw.get('y'),
                       raw.get('source'), language)
        if row is None:
            stats['unsupported_or_short'] += 1
            continue
        detected, probability = language_predictor(row['text'])
        if (isinstance(probability, bool) or not isinstance(probability, (int, float)) or
                not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError('Language detector returned an invalid probability')
        if detected != language or probability < MIN_LANGUAGE_PROBABILITY:
            stats['language_excluded'] += 1
            continue
        identity = row['identity']
        if identity in unique:
            stats['duplicates'] += 1
            if unique[identity]['y'] != row['y']:
                conflicts.add(identity)
            elif row['source'] < unique[identity]['source']:
                # Fingerprints must not depend on provider enumeration order.
                unique[identity] = row
        else:
            unique[identity] = row
    partitions = {name: [] for name in ('train', 'select', 'test')}
    for identity, row in sorted(unique.items()):
        if identity not in conflicts:
            partitions[_partition(row['family'])].append(row)
    family_sets = {name: {r['family'] for r in values} for name, values in partitions.items()}
    for left, right in (('train', 'select'), ('train', 'test'), ('select', 'test')):
        if family_sets[left] & family_sets[right]:
            raise ValueError('Internal family split overlap')
    stats['conflicting_identities_excluded'] = len(conflicts)
    stats['retained_rows'] = sum(len(values) for values in partitions.values())
    # Include zero counters so two sources with no exclusions have the same schema.
    for key in ('input_rows', 'unsupported_or_short', 'language_excluded', 'duplicates'):
        stats[key] += 0
    return partitions, dict(sorted(stats.items()))


def freeze(rows, language, lid, output, source_metadata):
    """Freeze a new data directory and return its manifest; lid is callable.

    Small corpora are retained as an audit with training_ready=false. The trainer
    must not bypass that indication or manufacture labels to meet minimums.
    """
    if not isinstance(source_metadata, dict):
        raise ValueError('Public source metadata must be a dictionary')
    partitions, stats = prepare_rows(rows, language, lid)
    counts = {name: dict(n=len(values), spam=sum(r['y'] == 1 for r in values),
                         ham=sum(r['y'] == 0 for r in values)) for name, values in partitions.items()}
    minimums = dict(train=100, select=30, test=30)
    ready = all(counts[name][label] >= minimums[name]
                for name in minimums for label in ('spam', 'ham'))
    manifest = dict(language=language, fingerprints={name: fingerprint(values) for name, values in partitions.items()},
                    counts=counts, source=source_metadata,
                    lang_lid=dict(method='caller-supplied pinned fastText detector',
                                  minimum_probability=MIN_LANGUAGE_PROBABILITY,
                                  target_language=language, label_blind=True),
                    normalization='v77_mail.clean; Unicode first45-word family; 30chars/10 non-placeholder word minimum',
                    split_policy=dict(seed=SPLIT_SEED, family_hash_buckets='train 0..69; select 70..84; test 85..99',
                                      families_disjoint=True, selection_uses_outcomes=False),
                    stats=stats, training_ready=ready,
                    required_class_minimums=minimums,
                    independent_source_test=False, optimizer_steps_executed=0,
                    customer_mail_used=False, automatically_relabelled=False)
    # Fail before creating the directory if metadata is not JSON serializable.
    rendered = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, values in partitions.items():
        with (output / (name + '.jsonl.gz')).open('wb') as binary:
            with gzip.GzipFile(filename='', mode='wb', fileobj=binary, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding='utf-8', newline='\n') as stream:
                    for row in values:
                        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
    (output / 'manifest.json').write_text(rendered, encoding='utf-8')
    return manifest
