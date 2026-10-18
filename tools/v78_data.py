"""English-only corpus preparation; quarantines the label-conflicted v76 provider.

GenSpam is entirely reserved for external evaluation, including its publisher
TRAIN partition. Only content is used; its XML metadata and label-bearing
filenames never reach a classifier. This is historical research mail.
"""
import gzip
import hashlib
import io
import json
import mailbox
import re
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
import urllib.request
import v77_mail as core

GENSPAM_URL = 'http://www.cl.cam.ac.uk/Research/NL/nl-download/GenSpam.tar.gz'
GENSPAM_SHA = '1e0b8f0eb73f16787df7b11144a2bc8937342d50fbedd124ba9fe379a076895f'
NAZARIO_URL = 'https://monkey.org/~jose/phishing/phishing-2025'
NAZARIO_SHA = 'f1fa7e0fe35c9a16f36d9aa20ade7e1d3908d1d0c1d917b8c53c8aa799ec1c8f'
SEED = 'v78-english-trusted-labels-external-genspam-2026-10-18'


def iter_array(stream, block_size=65536):
    decoder = json.JSONDecoder()
    buffer = stream.read(block_size).lstrip()
    if not buffer.startswith('['):
        raise ValueError('Expected JSON array')
    buffer = buffer[1:]
    while True:
        buffer = buffer.lstrip()
        if buffer.startswith(','):
            buffer = buffer[1:].lstrip()
        if buffer.startswith(']'):
            return
        try:
            value, end = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            more = stream.read(block_size)
            if not more:
                raise ValueError('Truncated JSON array')
            buffer += more
            continue
        yield value
        buffer = buffer[end:]


def normalize(text):
    text = core.clean('', text)
    text = re.sub(r'\[\s*(?:spam|ham|phishing)\s*\]|\*{2,}\s*spam\s*\*{2,}', ' ', text, flags=re.I)
    return core.SPACE.sub(' ', text).strip()


def identifiers(text):
    # Ignore punctuation differences introduced by old corpus converters.
    tokens = re.findall(r'[a-z]+', text.casefold())
    canonical = ' '.join(tokens)
    return core.digest(canonical), core.digest(' '.join(tokens[:45]))


def make_row(text, y, source):
    if type(y) is not int or y not in (0, 1):
        raise ValueError('Unknown source class')
    text = normalize(text)
    if core.language(text) != 'en' or len(text) < 30:
        return None
    identity, family = identifiers(text)
    return dict(text=text, y=y, source=source, identity=identity, family=family)


def unique(rows):
    seen = {}; conflicts = set(); counts = Counter()
    for row in rows:
        if row is None:
            counts['unsupported_or_short'] += 1
            continue
        key = row['identity']
        if key in seen:
            counts['duplicate'] += 1
            if seen[key]['y'] != row['y']:
                conflicts.add(key)
        else:
            seen[key] = row
    counts['conflicting_identities_excluded'] = len(conflicts)
    result = [r for key, r in seen.items() if key not in conflicts]
    return result, dict(counts)


def source_blob(name, archive_path=None):
    if name not in ('genspam.tar.gz', 'nazario-2025.mbox'):
        raise ValueError('Unknown public source name')
    expected, url = (GENSPAM_SHA, GENSPAM_URL) if name == 'genspam.tar.gz' else (NAZARIO_SHA, NAZARIO_URL)
    if archive_path:
        with zipfile.ZipFile(archive_path) as archive:
            blob = archive.read(name)
    else:
        with urllib.request.urlopen(url, timeout=90) as response:
            blob = response.read(60_000_001)
        if len(blob) > 60_000_000:
            raise ValueError('Unexpected source size')
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError('Pinned source changed: ' + name)
    return blob


def genspam_rows(blob):
    rows = []; raw_counts = Counter()
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as archive:
        for member in archive.getmembers():
            if not member.isfile() or member.size > 40_000_000:
                raise ValueError('Unexpected GenSpam archive member')
            name = Path(member.name).name
            if name not in {f'{split}_{kind}.ems' for split in ('train', 'adapt', 'test') for kind in ('GEN', 'SPAM')}:
                raise ValueError('Unknown GenSpam label filename')
            y = int('_SPAM.' in name)
            text = archive.extractfile(member).read().decode('utf-8', 'replace')
            messages = re.findall(r'<MESSAGE>(.*?)</MESSAGE>', text, flags=re.S)
            raw_counts[name] = len(messages)
            for message in messages:
                subject = re.search(r'<SUBJECT>(.*?)</SUBJECT>', message, re.S)
                body = re.search(r'<MESSAGE_BODY>(.*?)</MESSAGE_BODY>', message, re.S)
                if body is None:
                    raise ValueError('GenSpam message has no body element')
                content = (subject.group(1) if subject else '') + '\n' + body.group(1)
                # Resolve corpus placeholders before html.unescape and strip tags.
                mapping = {'NAME': 'nametoken', 'ORG': 'orgtoken', 'NUM': 'numbertoken',
                           'WEBSITE': 'urltoken', 'EMAIL': 'emailtoken', 'CHAR': ' ', 'SMILEY': ' '}
                for key, value in mapping.items():
                    content = re.sub(r'&' + key + r'\b;?', ' ' + value + ' ', content)
                content = re.sub(r'(?m)^\s*\^\s*', '', content)
                rows.append(make_row(content, y, 'GenSpam'))
    result, counts = unique(rows)
    return result, {'publisher_messages': dict(raw_counts), 'dedup': counts,
                    'eligible': len(result), 'classes': dict(Counter(r['y'] for r in result))}


def nazario_rows(blob):
    # mailbox.mbox needs a filename. Bytes are read as data, never executed.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.mbox') as temporary:
        temporary.write(blob); temporary.flush()
        box = mailbox.mbox(temporary.name)
        rows = [make_row(core.parse_eml(message.as_bytes()), 1, 'Nazario-2025') for message in box]
        box.close()
    return unique(rows)


def fingerprint(rows):
    return core.digest('\n'.join(sorted(r['identity'] + ':' + r['family'] for r in rows)))


def prepare(cache, archive_path=None, legacy_cache=Path('.cache/v77/rows.json.gz')):
    cache = Path(cache)
    if cache.is_file():
        with gzip.open(cache, 'rt') as stream:
            return json.load(stream)
    test_pool, test_provenance = genspam_rows(source_blob('genspam.tar.gz', archive_path))
    if len(test_pool) < 10000:
        raise ValueError('Fewer than 10000 unique GenSpam emails')
    test = sorted(test_pool, key=lambda r: core.digest(SEED + ':test:' + r['identity']))[:10000]
    protected_ids = {r['identity'] for r in test_pool}
    protected_families = {r['family'] for r in test_pool}
    legacy = []
    if not legacy_cache.is_file():
        from train_v77_ru_en import acquire
        acquire(legacy_cache)
    with gzip.open(legacy_cache, 'rt') as stream:
        for old in iter_array(stream):
            if old['lang'] == 'en' and old['official_split'] == 'train':
                legacy.append(make_row(old['text'], old['y'], old['source']))
    modern, modern_counts = nazario_rows(source_blob('nazario-2025.mbox', archive_path))
    development, dev_counts = unique(legacy + modern)
    parts = {'train': [], 'select': [], 'cal': []}; excluded = Counter()
    for row in development:
        if row['identity'] in protected_ids or row['family'] in protected_families:
            excluded['external_identity_or_prefix_overlap'] += 1
            continue
        bucket = int(core.digest(SEED + ':development:' + row['family'])[:12], 16) % 100
        part = 'train' if bucket < 70 else 'select' if bucket < 80 else 'cal'
        parts[part].append(row)
    # Label-blind per-source sampling retains rare positive phishing examples.
    train = []
    for source in sorted({r['source'] for r in parts['train']}):
        subset = [r for r in parts['train'] if r['source'] == source]
        subset.sort(key=lambda r: core.digest(SEED + ':train:' + r['identity']))
        train.extend(subset[:15000])
    parts['train'] = sorted(train, key=lambda r: core.digest(SEED + ':shuffle:' + r['identity']))
    for part in ('select', 'cal'):
        parts[part] = sorted(parts[part], key=lambda r: core.digest(SEED + ':' + part + ':' + r['identity']))[:18000]
    sets = [{r['family'] for r in parts[name]} for name in ('train', 'select', 'cal')]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)) or any(s & protected_families for s in sets):
        raise ValueError('Source holdout/group split overlap')
    provenance = {'seed': SEED, 'english_only': True, 'test_count': len(test),
        'test_source': 'GenSpam (all publisher splits reserved, never used for fitting)',
        'test_source_url': GENSPAM_URL, 'test_archive_sha256': GENSPAM_SHA,
        'test_source_provenance': test_provenance, 'test_fingerprint': fingerprint(test),
        'sampling': 'smallest SHA256(seed:content identity), no labels or predictions in ranking',
        'gen_spam_source_held_out_entirely': True, 'prefix_token_groups_disjoint': True,
        'source_level_test_previously_seen_by_other_project_versions_possible': True,
        'quarantined_provider': 'sniff07/phishing-dataset: 9673 canonical HAM-to-SPAM label conflicts; no rows used',
        'nazario_2025_sha256': NAZARIO_SHA, 'nazario_counts': modern_counts,
        'development_dedup': dev_counts, 'development_exclusions': dict(excluded),
        'counts': {name: len(values) for name, values in parts.items()},
        'split_fingerprints': {name: fingerprint(values) for name, values in parts.items()},
        'source_class_counts': {name: dict(Counter(f"{r['source']}:{r['y']}" for r in values)) for name, values in parts.items()},
        'historical_research_mail_not_live_customer_mail': True,
        'semantic_near_duplicates_not_fully_excluded': True}
    payload = {'development': parts, 'test': test, 'provenance': provenance}
    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, 'wt') as stream:
        json.dump(payload, stream)
    return payload
