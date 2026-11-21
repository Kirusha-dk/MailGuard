"""Freeze the exact public SpamAssassin archives used by the parser differential audit.

Reads local checksum-pinned archives only; never trains, calls an AI API or extracts
archive paths onto the filesystem. Output must be a new directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile

ARCHIVES = (
    ('easy_ham.tar.bz2', 'ham', '2b7b65904bcfcc31d2b5f51946f2d261370b257402cbbd62930b46ab83367438',
     'https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2'),
    ('spam.tar.bz2', 'spam', 'c08debc32413804949a866be45ef78195cec2cbafd1da744ed76cdb860589743',
     'https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2'))


def prepare(archive_dir, output):
    archives, items, conflicts, duplicates = [], {}, set(), 0
    for name, label, expected, url in ARCHIVES:
        path = archive_dir / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Archive checksum mismatch: ' + name)
        archives.append(dict(url=url, sha256=actual, bytes=path.stat().st_size, label=label))
        with tarfile.open(path, 'r:bz2') as archive:
            for member in archive:
                if not member.isfile() or not re.match(r'^\d{5}\.', Path(member.name).name):
                    continue
                with archive.extractfile(member) as stream:
                    raw = stream.read(20_000_001)
                if len(raw) > 20_000_000:
                    raise ValueError('Oversized public corpus member')
                identity = hashlib.sha256(raw).hexdigest()
                if identity in items:
                    duplicates += 1
                    if items[identity][0] != label:
                        conflicts.add(identity)
                else:
                    items[identity] = label, raw
    entries = sorted((identity, label) for identity, (label, _) in items.items() if identity not in conflicts)
    manifest = dict(source='SpamAssassin public corpus 2003', archives=archives,
                    selected_messages=len(entries), ham=sum(y == 'ham' for _, y in entries),
                    spam=sum(y == 'spam' for _, y in entries), raw_duplicates_removed=duplicates,
                    label_conflicting_identities_removed=len(conflicts),
                    selection='All regular files matching five-digit mail filenames; raw SHA256 deduplication before parsing',
                    content_and_label_fingerprint=hashlib.sha256(json.dumps(entries, separators=(',', ':')).encode()).hexdigest(),
                    public_corpus=True, independent_quality_test=False, customer_mail_used=False, model_used=False)
    output.mkdir(parents=True, exist_ok=False)
    for identity, label in entries:
        path = output / 'messages' / label / (identity + '.eml')
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(items[identity][1])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.archive_dir, args.output)
    print(json.dumps({k: manifest[k] for k in ('selected_messages', 'ham', 'spam', 'content_and_label_fingerprint')}, indent=2))


if __name__ == '__main__':
    main()
