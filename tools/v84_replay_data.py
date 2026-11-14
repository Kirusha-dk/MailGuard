"""Freeze an expanded English replay corpus. Previously scored pools become development.

No model is fitted here. SELECT remains isolated; old TEST/CAL names are retired
for this experiment. Raw public source text stays in the private runner cache.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import zipfile

import fasttext

from v77_mail import parse_eml
from v78_data import identifiers
from v79_data import LID_SHA, english_prediction, normalize
from v82_neural import CONFIG, fingerprint as old_fingerprint

PARENT_SHA = '7be6f93c074e4460941ba444abba486b23cd601d3a21bebe130c1723a6358784'
PARENT_STEP = 8500
PARENT_CURSOR = 45312
SOURCE_SHA = '65bdb680fe95f3d2f52d24d99af7b7d6490c03fc5508313e8f5edefb4672de41'
ERROR_ZIP_SHA = '5dfc274f7d984940a0a8938a8171eaae4d6b1bcd55c2f36d90c3351e2528e9ab'


class EmailSubstitution:
    """An exact no-op fast path; the email pattern requires a literal @."""
    def __init__(self, pattern):
        self.pattern = pattern

    def sub(self, replacement, text, count=0):
        return self.pattern.sub(replacement,text,count=count) if '@' in text else text


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_rows(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def fingerprint(rows):
    h = hashlib.sha256()
    for r in rows:
        value = json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
        h.update(len(value).to_bytes(8, 'big')); h.update(value)
    return h.hexdigest()


def write_rows(path, rows):
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8') as f:
                for r in rows:
                    f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n')


def windows(text):
    words = re.sub(r'\b(?:numbertoken|urltoken|emailtoken)\b', '', text).split()[:120]
    return {hashlib.sha256(' '.join(words[i:i+20]).encode()).digest()
            for i in range(max(0, len(words)-19))}


def prepared_text(row):
    # Historical v64 artifacts preserve subject and body excerpts, not full EML.
    text = row['context']
    if not text.startswith('subject ') or ' body ' not in text:
        raise ValueError('Unexpected historical prepared view')
    subject, body = text[8:].split(' body ', 1)
    return normalize(subject + ' ' + body.replace(' tail ', ' '))


class Pool:
    def __init__(self, select):
        self.select_ids = {r['identity'] for r in select}
        self.select_families = {r['family'] for r in select}
        self.select_windows = set().union(*(windows(r['text']) for r in select))
        self.rows = {}; self.family_labels = {}; self.conflicts = set(); self.stats = Counter()

    def add(self, r, *, historical_excerpt=False):
        if type(r['y']) is not int or r['y'] not in (0, 1):
            raise ValueError('Unknown source label')
        ident, family = r['identity'], r['family']
        if (ident in self.select_ids or family in self.select_families or
                (historical_excerpt and windows(r['text']) & self.select_windows)):
            self.stats['select_overlap_excluded'] += 1; return
        labels = self.family_labels.setdefault(family, set()); labels.add(r['y'])
        if len(labels) != 1:
            self.conflicts.add(family); self.stats['conflicting_family'] += 1; return
        if ident in self.rows:
            old = self.rows[ident]
            if old['y'] != r['y']:
                self.conflicts.update((old['family'], family))
            old['hard'] = old['hard'] or r['hard']
            old['parent_exposures'] = max(old['parent_exposures'], r['parent_exposures'])
            self.stats['duplicate_identity'] += 1; return
        self.rows[ident] = r

    def export(self):
        return sorted((r for r in self.rows.values() if r['family'] not in self.conflicts),
                      key=lambda r: r['identity'])


def error_rows(path, archive, work):
    """Recover only the640 known false negatives from the consumed30k test."""
    import numpy as np
    if sha(path) != ERROR_ZIP_SHA or sha(archive) != SOURCE_SHA:
        raise ValueError('Consumed audit/source checksum mismatch')
    with zipfile.ZipFile(path) as z:
        report = json.loads(z.read('report.json'))
        origins = {r['identity']: r for r in json.loads(z.read('origins.json'))}
        with np.load(io.BytesIO(z.read('scores.npz')), allow_pickle=False) as a:
            hard = {i for i, score in zip(a['identity'].tolist(), a['scores'])
                    if score < report['input_lock']['threshold']}
    if len(hard) != 640 or report['metrics']['fn'] != 640:
        raise ValueError('Original audit errors do not reproduce')
    selected = [origins[i] for i in sorted(hard)]
    for r in selected:
        p = PurePosixPath(r['origin'])
        if p.is_absolute() or '..' in p.parts or not re.fullmatch(r'[\w./-]+', r['origin']):
            raise ValueError('Unsafe source origin')
    dest = work / 'error-mail'; dest.mkdir(parents=True, exist_ok=True)
    targets = [r['origin'] for r in selected]
    if shutil.which('7z'):
        listing = work / 'error-paths.txt'; listing.write_text('\n'.join(targets)+'\n')
        subprocess.run(['7z', 'x', '-y', '-bd', '-bso0', '-bsp0', '-i@'+str(listing),
                        '-o'+str(dest), str(archive)], check=True)
    else:
        import py7zr
        with py7zr.SevenZipFile(archive) as z:
            z.extract(path=dest, targets=targets)
    result = []
    for origin in selected:
        p = dest / origin['origin']
        if sha(p) != origin['raw_sha256']:
            raise ValueError('Original message bytes changed')
        text = normalize(parse_eml(p.read_bytes())); ident, family = identifiers(text)
        if ident != origin['identity'] or family != origin['family']:
            raise ValueError('Consumed audit normalization changed')
        result.append(dict(identity=ident, family=family, text=text, y=1,
                           source='SpamArchive-2012', hard=True, parent_exposures=0))
    shutil.rmtree(dest)
    return result


def freeze(args):
    import v77_mail
    # Already normalized v64 excerpts usually contain emailtoken, not @.
    # Avoid quadratic regex scans through long alphanumeric runs in that case.
    v77_mail.ADDRESS = EmailSubstitution(v77_mail.ADDRESS)
    old_train = []; select = []; retired = []
    for root, ref in ((Path('.cache/v79/frozen'), 'reports/v79-frozen.json'),
                      (Path('.cache/v80/extra'), 'reports/v80-extra-frozen.json')):
        if json.loads((root/'frozen.json').read_text()) != json.loads(Path(ref).read_text()):
            raise ValueError('Original corpus manifest drift')
        old_train.extend(read_rows(root/'train.jsonl.gz'))
        select.extend(read_rows(root/'select.jsonl.gz'))
        for name in ('cal', 'test', 'fresh-test'):
            if (root/(name+'.jsonl.gz')).exists():
                retired.extend(read_rows(root/(name+'.jsonl.gz')))
    import torch
    if sha(args.parent) != PARENT_SHA:
        raise ValueError('Parent checkpoint mismatch')
    cp = torch.load(args.parent, map_location='cpu', weights_only=True)
    if (cp['progress']['step'] != PARENT_STEP or cp['progress']['active_session'] is not None or
            cp['metadata']['train'] != old_fingerprint(old_train) or
            cp['metadata']['select'] != old_fingerprint(select) or cp['metadata']['config'] != CONFIG):
        raise ValueError('Parent corpus/config/completion mismatch')
    if cp['progress']['epoch'] != 1 or cp['progress']['cursor'] != PARENT_CURSOR:
        raise ValueError('Expected precise parent row exposure history')
    order = sorted(old_train, key=lambda r: hashlib.sha256(f"82:1:{r['identity']}".encode()).hexdigest())
    twice = {r['identity'] for r in order[:PARENT_CURSOR]}
    pool = Pool(select)
    def wrap(r, parent=False):
        return {k: r[k] for k in ('identity', 'family', 'text', 'y', 'source')} | dict(
            hard=False, parent_exposures=(2 if r['identity'] in twice else 1) if parent else 0)
    for r in old_train: pool.add(wrap(r, True))
    for r in retired: pool.add(wrap(r))
    lid_path = Path('.cache/v79/frozen/lid.176.ftz')
    if sha(lid_path) != LID_SHA: raise ValueError('Language model drift')
    lid = fasttext.load_model(str(lid_path))
    expected = json.loads(Path('reports/v64-transfer-result.json').read_text())['audit']['shardSha256']
    inputs = {}; read = 0; english = 0
    for shard, digest in sorted(expected.items()):
        p = args.old / ('input-'+shard+'.jsonl.gz')
        if sha(p) != digest: raise ValueError('Previously scored shard drift')
        inputs[p.name] = digest; count = 0
        for old in read_rows(p):
            count += 1; text = prepared_text(old)
            if not english_prediction(lid, text)[0]: continue
            english += 1; ident, family = identifiers(text)
            pool.add(dict(identity=ident, family=family, text=text, y=old['y'], source=old['source'],
                          hard=False, parent_exposures=0), historical_excerpt=True)
        if count != 100000: raise ValueError('Incomplete previously scored input')
        read += count
        print(json.dumps(dict(previously_scored_rows_read=read,english_candidates=english,
                              retained_before_conflict_cleanup=len(pool.rows))),flush=True)
    args.work.mkdir(parents=True, exist_ok=True)
    recovered = error_rows(args.errors, args.archive, args.work)
    for r in recovered: pool.add(r)
    train = pool.export()
    if len(train) <= len(old_train) or len({r['identity'] for r in select}) != len(select):
        raise ValueError('Corpus not expanded or SELECT not unique')
    if {r['family'] for r in train} & {r['family'] for r in select}:
        raise ValueError('TRAIN/SELECT family overlap')
    manifest = dict(version=84, dataset_version='v84-en-replay-20261105-v1',
        train_rows=len(train), select_rows=len(select),
        train_by_source_class=dict(Counter(f"{r['source']}:{r['y']}" for r in train)),
        train_spam=sum(r['y'] for r in train), train_ham=sum(1-r['y'] for r in train),
        train_fingerprint=fingerprint(train), select_fingerprint=fingerprint(select),
        original_v82_train_rows=len(old_train), original_v82_train_fingerprint=old_fingerprint(old_train),
        original_v82_select_fingerprint=old_fingerprint(select), parent_checkpoint_sha256=PARENT_SHA,
        parent_step=PARENT_STEP, parent_exposure_counts_carried_forward=True,
        max_lifetime_row_exposures=3, retired_cal_test_rows_read=len(retired),
        previously_scored_v64_rows_read=read, previously_scored_v64_english_candidates=english,
        previous_shard_sha256=inputs, previous_30k_audit_zip_sha256=ERROR_ZIP_SHA,
        previous_30k_cohort_retired_from_independent_evaluation=True,
        recovered_original_fn_rows=len(recovered), retained_known_fn_rows=sum(r['hard'] for r in train),
        deduplication=dict(pool.stats), training_ready=True,
        select_is_development=True, new_independent_test_available=False,
        customer_mail_used=False, semantic_near_duplicates_fully_excluded=False,
        source_label_limits='Inherited public email labels; trap positives are weak labels. No labels invented or corrected automatically.',
        old_view_limits='v64 rows are subject/body excerpts with preexisting normalization, not complete original EML.',
        historical_unknown_merged_provenance_preserved=True)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, rows in (('train', train), ('select', select)):
        write_rows(args.out/(name+'.jsonl.gz'), rows)
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--parent', type=Path, default=Path('.cache/v84/warmstart/checkpoint.pt'))
    p.add_argument('--old', type=Path, default=Path('.cache/v84/old-tests'))
    p.add_argument('--errors', type=Path, default=Path('.cache/v84/errors/result.zip'))
    p.add_argument('--archive', type=Path, default=Path('.cache/v84/raw/2012.7z'))
    p.add_argument('--work', type=Path, default=Path('.cache/v84/work'))
    p.add_argument('--out', type=Path, default=Path('.cache/v84/corpus'))
    freeze(p.parse_args())
