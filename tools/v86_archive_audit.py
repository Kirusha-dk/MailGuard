"""Audit pinned native email archives; this command never trains a model.

Raw mail and normalized candidate rows stay in a local/runner cache. A source
inventory is not a frozen TRAIN set or evidence that a monthly run happened.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import time

import libarchive
import requests


def initialize(lid_path):
    global lid, parse_eml, normalize, english_prediction, identifiers
    import fasttext
    import v77_mail
    from v77_mail import parse_eml
    from v79_data import normalize, english_prediction, LID_SHA
    from v78_data import identifiers
    from v84_replay_data import EmailSubstitution
    v77_mail.ADDRESS = EmailSubstitution(v77_mail.ADDRESS)
    with Path(lid_path).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != LID_SHA:
            raise ValueError('Language model drift')
    lid = fasttext.load_model(str(lid_path))


def process(batch):
    result = []
    for source, name, raw in batch:
        try:
            text = normalize(parse_eml(raw))
            if len(text) < 30 or len(text.split()) < 10:
                result.append(('short', None)); continue
            if not english_prediction(lid, text)[0]:
                result.append(('non_english', None)); continue
            identity, family = identifiers(text)
            result.append(('english', dict(identity=identity, family=family,
                text=text, y=1, source=source, origin=name,
                raw_sha256=hashlib.sha256(raw).hexdigest())))
        except Exception as error:
            result.append(('parse_error:' + type(error).__name__, None))
    return result


def batches(path, source, stats, maximum_bytes):
    batch = []; examined = 0
    with libarchive.file_reader(str(path)) as archive:
        for entry in archive:
            if not entry.isfile:
                continue
            limit = source['member_scan_limit']
            if limit is not None and examined >= limit:
                break
            examined += 1; stats['archive_files'] += 1
            if entry.size > maximum_bytes:
                stats['oversized'] += 1; continue
            batch.append(('SpamArchive-' + str(source['year']), entry.pathname,
                          b''.join(entry.get_blocks())))
            if len(batch) == 64:
                yield batch; batch = []
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=Path('research/v86/archive-sources.json'))
    parser.add_argument('--raw', type=Path, default=Path('.cache/v86/raw'))
    parser.add_argument('--output', type=Path, default=Path('.cache/v86/archive-audit'))
    parser.add_argument('--lid', type=Path, default=Path('.cache/v79/frozen/lid.176.ftz'))
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    from v79_data import LID_SHA
    if catalog['language_model_sha256'] != LID_SHA or catalog['english_probability_min'] != .8:
        raise ValueError('Language gate drift')
    args.raw.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Audit output exists; use a separate output directory')
    args.output.mkdir(parents=True)
    seen = set(); stats = Counter(); by_year = {}; started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(args.lid,)) as pool, gzip.open(
            args.output / 'candidates.jsonl.gz', 'wt') as output:
        for source in catalog['sources']:
            if source['label'] != 1:
                raise ValueError('Archive has no legitimate/HAM label')
            path = args.raw / (str(source['year']) + '.7z')
            if not path.exists():
                pending = path.with_suffix('.partial')
                with requests.get(source['url'], stream=True, timeout=90) as response:
                    response.raise_for_status()
                    with pending.open('wb') as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                pending.replace(path)
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != source['sha256'] or path.stat().st_size != source['bytes']:
                raise ValueError('Pinned archive drift')
            iterator = iter(batches(path, source, stats, catalog['raw_member_limit_bytes']))
            pending = deque(); finished = False; counts = Counter()
            while pending or not finished:
                while len(pending) < 8 and not finished:
                    try:
                        pending.append(pool.submit(process, next(iterator)))
                    except StopIteration:
                        finished = True
                if not pending:
                    break
                for reason, row in pending.popleft().result():
                    stats['parsed'] += 1; stats[reason] += 1; counts[reason] += 1
                    if row:
                        if row['identity'] in seen:
                            stats['duplicate_identity'] += 1
                            counts['duplicate_identity'] += 1
                        else:
                            seen.add(row['identity']); counts['unique_english'] += 1
                            output.write(json.dumps(row) + '\n')
                if stats['parsed'] // 10000 > (stats['parsed'] - 64) // 10000:
                    print(dict(stats, canonical_unique_english=len(seen),
                               seconds=round(time.monotonic() - started)), flush=True)
            by_year[str(source['year'])] = dict(counts)
    report = dict(sources=catalog['sources'], stats=dict(stats), by_year=by_year,
        canonical_unique_english=len(seen), old_overlap_checked=False,
        prior_model_scoring_performed=False, optimizer_steps=0, training_ready=False,
        customer_mail_used=False, labels_are_weak=True, supplies_ham=False,
        language_model_sha256=catalog['language_model_sha256'],
        fixed_prefix_limits_are_not_full_archive_counts=True)
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
