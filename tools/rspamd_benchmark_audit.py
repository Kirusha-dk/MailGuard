"""Strict, local-only Rspamd benchmark transport and split integrity checks."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

SPAM_ACTIONS = {'reject', 'add header', 'rewrite subject', 'quarantine', 'discard'}


def request_scan(url, raw, timeout=30):
    # Force full evaluation even when a prefilter requests an early action.
    request = Request(url, data=raw, method='POST', headers={
        'Content-Type': 'message/rfc822', 'Flags': 'pass_all',
        'User-Agent': 'MailGuard-benchmark-v41'})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def parse_scan(reply):
    if not isinstance(reply, dict) or reply.get('error') or reply.get('is_skipped'):
        raise ValueError('Rspamd returned an error or skipped scan')
    for key in ('score', 'required_score'):
        value = reply.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Rspamd missing/invalid {key}; refusing to replace it with zero')
    action = reply.get('action')
    if not isinstance(action, str) or not action:
        raise ValueError('Rspamd missing action')
    action = action.lower()
    if action in {'soft reject', 'greylist'}:
        raise ValueError('Deferred scan is not a completed binary classification')
    symbols = reply.get('symbols')
    if not isinstance(symbols, dict):
        raise ValueError('Rspamd missing symbols')
    parsed = []
    for name, value in symbols.items():
        score = value.get('score') if isinstance(value, dict) else None
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f'Invalid symbol score: {name}')
        parsed.append((name, float(score)))
    return dict(action=action, rscore=float(reply['score']),
                required=float(reply['required_score']), symbols=parsed,
                rspam=action in SPAM_ACTIONS)


def check_learning_reply(reply):
    if not isinstance(reply, dict) or reply.get('success') is not True or reply.get('error'):
        raise ValueError('Bayes did not acknowledge learning')


def scan_health(rows):
    if not rows:
        raise ValueError('Empty scan batch')
    counts = Counter(name for r in rows for name, _ in r['symbols'])
    scores = [r['rscore'] for r in rows]
    info = {'count': len(rows), 'scoreMin': min(scores), 'scoreMax': max(scores),
            'actions': dict(Counter(r['action'] for r in rows)),
            'bayesHits': sum(counts[x] for x in ('BAYES_SPAM', 'BAYES_HAM')),
            'topSymbols': counts.most_common(25)}
    if min(scores) == max(scores) == 0 or not info['bayesHits']:
        raise ValueError('Unhealthy baseline: all scores zero or no Bayes symbols')
    return info


def load_frozen_v40(cache, expected_counts=None):
    cache = Path(cache)
    manifest = cache / 'manifest.json'
    obj = json.loads(manifest.read_text())
    if obj.get('version') != 40:
        raise ValueError('Expected the original v40 manifest')
    result, audit, seen = {}, {}, set()
    root = (cache / 'eml').resolve()
    expected_counts = expected_counts or {'train': 50000, 'val': 10000, 'test': 50000}
    for split, expected in expected_counts.items():
        entries = obj[split]
        if len(entries) != expected:
            raise ValueError(f'Wrong {split} size')
        rows, digest, current = [], hashlib.sha256(), set()
        for item in entries:
            path = (root / item['file']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Manifest path escapes corpus')
            label = item['y']
            if type(label) is not int or label not in (0, 1):
                raise ValueError('Invalid label')
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if content_hash in seen or content_hash in current:
                raise ValueError('Duplicate EML within/across frozen splits')
            current.add(content_hash)
            digest.update(json.dumps([item['file'], content_hash, label, item['source']],
                                     ensure_ascii=True).encode() + b'\n')
            rows.append(dict(path=path, y=label, source=item['source']))
        seen.update(current)
        result[split] = rows
        audit[split] = {'count': len(rows), 'orderedSha256': digest.hexdigest(),
                        'spam': sum(r['y'] for r in rows)}
    audit['manifestSha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return result, audit
