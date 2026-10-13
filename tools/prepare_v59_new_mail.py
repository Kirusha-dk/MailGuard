#!/usr/bin/env python3
"""Freeze genuinely new public email families before estimator fitting.

Old development mail is reconstructed from the v55 manifest. No old TEST
errors are promoted to TRAIN: ambiguous ham labels are not memorized.
"""
import argparse
import csv
import gzip
import hashlib
import html
import json
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

from rspamd_benchmark_audit import load_frozen_v40
from train_v47_phishing_aware import from_eml as old_parse
from train_v47_phishing_aware import canonical_fields
from train_v55_multilingual import campaign_key, compact, corpus_rows, file_sha, CORPUS_SHA
from evaluate_v54_unique500k import Pool, block_previous, content_keys, download, has_mail_headers
from v59_mail import from_eml

SPLITS = ('train', 'select', 'cal', 'test')
TREC_URL = 'https://cdn.aibydoing.com/aibydoing/files/trec06c.zip'
W3C = 'https://lists.w3.org/Archives/Public/public-html-ig-jp/'


def family_split(family, source):
    n = int(family[-12:], 16) % 100
    if source == 'TREC06-Chinese':
        return 'train' if n < 50 else 'select' if n < 65 else 'cal' if n < 80 else 'test'
    if source.startswith('SpamArchive-2026-'):
        month = int(source.rsplit('-', 1)[1])
        return 'test' if month >= 6 else 'train' if n < 50 else 'select' if n < 75 else 'cal'
    if source == 'W3C-Japanese-weak-ham':
        return 'train'
    raise ValueError('Unknown new source')


class Collector:
    def __init__(self, pool):
        self.pool = pool
        self.rows, self.conflicts, self.keys = {}, set(), set()
        self.stats = Counter()

    def add(self, raw, label, source, origin, canonical=None):
        parsed = canonical or from_eml(raw)
        family = campaign_key(parsed['ngram_text'])
        keys, body = content_keys(parsed)
        if len(body) < 20 or not any(c.isalpha() for c in body):
            self.stats['shortOrEmpty'] += 1; return
        old = old_parse(raw) if raw else parsed
        old_body = old['ngram_text'].split('\nBODY: ', 1)[-1]
        # Garbled ASCII-default decoding must not merge unrelated Chinese mail.
        aliases = list(keys) + ['family:' + family]
        if old_body.count('\ufffd') / max(1, len(old_body)) < .005:
            aliases += list(content_keys(old)[0]) + ['family:' + campaign_key(old['ngram_text'])]
        if any(self.pool.db.execute('SELECT 1 FROM keys WHERE key=?', (k,)).fetchone() for k in aliases):
            self.stats['relatedToPriorDevelopment'] += 1; return
        if family in self.rows and self.rows[family]['y'] != label:
            self.conflicts.add(family)
            self.stats['conflictingFamily'] += 1; return
        if family in self.rows or any(k in self.keys for k in keys):
            self.stats['duplicateNewContentOrFamily'] += 1; return
        self.keys.update(keys)
        row = compact(dict(parsed, y=label, source=source), family)
        row.update(text=parsed['ngram_text'], old_text=old['ngram_text'], origin=origin,
                   rawSha256=hashlib.sha256(raw).hexdigest() if raw else None,
                   weakLabel=source == 'W3C-Japanese-weak-ham', split=family_split(family, source))
        self.rows[family] = row


def collect_trec(collector, work):
    path = work / 'trec06c.zip'
    provenance = download(TREC_URL, path)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        indexes = [n for n in names if n.endswith('/full/index')]
        if len(indexes) != 1:
            raise ValueError('Expected exactly one original TREC Chinese label index')
        index = indexes[0]
        prefix = index[:-len('full/index')]
        label_bytes = z.read(index)
        labels = [line.split() for line in label_bytes.decode('ascii').splitlines() if line.strip()]
        counts = Counter(r[0] for r in labels)
        if counts != {'ham': 21766, 'spam': 42854} or len(labels) != 64620:
            raise ValueError(f'Unexpected original TREC Chinese labels: {counts}')
        for label, name in labels:
            safe = PurePosixPath(name)
            if safe.is_absolute() or safe.parts[:2] != ('..', 'data') or '..' in safe.parts[1:]:
                raise ValueError('Unexpected index path')
            member = prefix + str(PurePosixPath(*safe.parts[1:]))
            collector.add(z.read(member), int(label == 'spam'), 'TREC06-Chinese', name)
        provenance.update(labelIndexSha256=hashlib.sha256(label_bytes).hexdigest(), originalLabelCounts=dict(counts),
                          originalCorpus='https://cormack.uwaterloo.ca/treccorpus06/',
                          officialDescription='https://trec.nist.gov/pubs/trec15/papers/SPAM06.OVERVIEW.pdf',
                          redistribution='Raw mail is not uploaded; archive and label hashes identify the research input.',
                          archiveReadmeNames=[n for n in names if 'readme' in n.lower()][:20])
    path.unlink()
    print('Chinese collection:', dict(collector.stats), flush=True)
    return provenance


def collect_month(collector, month, work):
    name = f'2026-{month:02d}'
    path = work / (name + '.7z')
    provenance = download(f'https://untroubled.org/spam/{name}.7z', path)
    listing = subprocess.run(['7z', 'l', '-slt', str(path)], check=True, capture_output=True, text=True).stdout
    for entry in listing.split('----------\n', 1)[-1].split('\n\n'):
        if 'Symbolic Link =' in entry or 'Hard Link =' in entry:
            raise ValueError('Archive link rejected')
        for line in entry.splitlines():
            if line.startswith('Path = '):
                p = PurePosixPath(line[7:].replace('\\', '/'))
                if p.is_absolute() or '..' in p.parts:
                    raise ValueError('Unsafe archive path')
    dest = work / name
    subprocess.run(['7z', 'x', '-y', '-bd', '-bso0', '-bsp0', f'-o{dest}', str(path)], check=True)
    count = 0
    for file in sorted(dest.rglob('*')):
        if file.is_symlink():
            raise ValueError('Archive symlink rejected')
        if not file.is_file() or file.stat().st_size > 8 * 1024 * 1024:
            continue
        raw = file.read_bytes()
        if has_mail_headers(raw):
            collector.add(raw, 1, 'SpamArchive-' + name, str(file.relative_to(dest)))
            count += 1
    provenance['rawEmailsRead'] = count
    provenance['labels'] = 'Spam trap; not individually human adjudicated'
    shutil.rmtree(dest); path.unlink()
    print(name, 'raw', count, 'unique accumulated', len(collector.rows), flush=True)
    return provenance


def read_url(url):
    with urlopen(Request(url, headers={'User-Agent': 'MailGuard-public-research/1.0'}), timeout=45) as f:
        return f.read()


def collect_japanese(collector):
    # Weak training-only labels: public technical discussions are not a gold HAM test.
    count, hashes = 0, []
    for month in ('2010Jan', '2010Mar', '2010Jul', '2010Dec', '2011Jan', '2011Apr'):
        page = read_url(W3C + month + '/').decode('utf-8', 'replace')
        ids = sorted(set(re.findall(r'href=["\'](?:\./)?(\d{4}\.html)["\']', page)))
        for name in ids:
            raw = read_url(W3C + month + '/' + name)
            text = raw.decode('utf-8', 'replace')
            if 'spam marked' in text.lower() or 'spam-marked' in text.lower():
                continue
            subject = re.search(r'<h1[^>]*>(.*?)</h1>', text, re.S | re.I)
            body = re.search(r'<pre[^>]*>(.*?)</pre>', text, re.S | re.I)
            if not body:
                body = re.search(r'<!-- body="start" -->(.*?)<!-- body="end" -->', text, re.S)
            if not subject or not body:
                continue
            plain = lambda s: html.unescape(re.sub(r'<[^>]+>', ' ', s))
            body_text = plain(body.group(1))
            if sum('\u3040' <= c <= '\u30ff' for c in body_text) < 20:
                continue
            parsed = canonical_fields(plain(subject.group(1)), body_text)
            collector.add(b'', 0, 'W3C-Japanese-weak-ham', W3C + month + '/' + name, canonical=parsed)
            hashes.append(hashlib.sha256(raw).hexdigest()); count += 1
        print('Japanese weak TRAIN', month, count, flush=True)
    return dict(url=W3C, pagesUsed=count, pageHashesSha256=hashlib.sha256('\n'.join(hashes).encode()).hexdigest(),
                labels='Weak ham assumption for training only; excluded from all selection/calibration/tests')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='.cache/v59-data')
    p.add_argument('--manifest', default='.cache/v55/v55-full/split-manifest.csv')
    p.add_argument('--corpus', default='.cache/v54-corpus/selected.jsonl.gz')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = list(csv.DictReader(Path(args.manifest).open()))
    wanted = {r['identity']: r for r in manifest if r['split'] != 'test'}
    legacy = {s: [] for s in SPLITS}
    pool = Pool(out / 'blocked.sqlite')
    original_block = pool.block
    def block(row):
        original_block(row)
        pool.db.execute('INSERT OR IGNORE INTO keys VALUES (?,NULL,NULL)', ('family:' + campaign_key(row['ngram_text']),))
    pool.block = block
    previous = block_previous(pool, '.cache/v40-fresh50k')
    if file_sha(args.corpus) != CORPUS_SHA:
        raise ValueError('Old corpus hash mismatch')
    def retain(parsed, y, source):
        entry = wanted.get(parsed['identity'])
        if entry:
            if (int(entry['label']), entry['source']) != (y, source):
                raise ValueError('Old development manifest mismatch')
            row = compact(dict(parsed, y=y, source=source), entry['family'])
            if entry['split'] != 'train':
                row['text'] = parsed['ngram_text']
            legacy[entry['split']].append(row)
    frozen, _ = load_frozen_v40('.cache/v40-fresh50k')
    for split in ('train', 'val'):
        for r in frozen[split]:
            retain(old_parse(r['path'].read_bytes()), int(r['y']), r['source'])
    for r in corpus_rows(args.corpus):
        pool.block(r)
        retain(r, int(r['y']), r['source'])
    if sum(map(len, legacy.values())) != len(wanted):
        raise ValueError('Old development reconstruction incomplete')
    pool.db.commit()
    print('Legacy development reconstructed; all previous pools blocked', flush=True)
    collector = Collector(pool)
    work = out / 'downloads'; work.mkdir()
    sources = [collect_trec(collector, work)]
    for month in (1, 2, 3, 6, 7, 8, 9):
        sources.append(collect_month(collector, month, work))
    try:
        sources.append(collect_japanese(collector))
    except (OSError, ValueError) as error:
        sources.append(dict(url=W3C, optionalWeakTrainingSourceIncomplete=True, error=str(error)))
        print('Optional Japanese weak source incomplete:', error, flush=True)
    counts, new_counts = {}, {}
    split_keys = []
    with (out / 'manifest.csv').open('w', newline='') as mf:
        writer = csv.writer(mf); writer.writerow(['identity', 'family', 'split', 'label', 'source', 'rawSha256', 'weakLabel'])
        for split in SPLITS:
            new_rows = sorted((r for f, r in collector.rows.items() if f not in collector.conflicts and r['split'] == split), key=lambda r: r['identity'])
            rows = legacy[split] + new_rows
            if {r['y'] for r in rows} != {0, 1}:
                raise ValueError(f'{split} requires both gold classes')
            with gzip.open(out / f'{split}.jsonl.gz', 'wt', encoding='utf-8') as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
                    writer.writerow([r['identity'], r['family'], split, r['y'], r['source'], r.get('rawSha256'), r.get('weakLabel', False)])
            counts[split] = dict(Counter(f'{r["source"]}:{r["y"]}' for r in rows))
            new_counts[split] = dict(Counter(f'{r["source"]}:{r["y"]}' for r in new_rows))
            split_keys.append({r['family'] for r in rows})
    if any(split_keys[i] & split_keys[j] for i in range(4) for j in range(i + 1, 4)):
        raise AssertionError('Campaign family overlap between splits')
    provenance = dict(version=59, counts=counts, newCounts=new_counts, sourceArchives=sources, priorBlocking=previous,
                      exclusions=dict(collector.stats), oldCorpusSha256=CORPUS_SHA, manifestSha256=file_sha(out / 'manifest.csv'),
                      splitHashes={s: file_sha(out / f'{s}.jsonl.gz') for s in SPLITS}, familyOverlap=0,
                      testPreviouslyScored=False, testLabelsUsedForTraining=False, weakLabelsUsedOutsideTrain=False,
                      caveat='Public historical Chinese mail is not a customer sample. Spam traps are weak spam labels. Family fingerprints cannot exclude every semantic near-duplicate. Unselected old SpamArchive pool is unavailable; all selected 500k, whole HF source pools and frozen 110k are blocked.')
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    pool.close()
    print('Locked new splits:', json.dumps(new_counts), flush=True)


if __name__ == '__main__':
    main()
