"""Fetch pinned v79 inputs and rebuild required legacy caches without fitting.

The --archive ZIP must contain the three SHA256-pinned public email archives.
Downloading the merged public text dataset does not make it eligible for TEST.
"""
import argparse
import gzip
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from train_v77_ru_en import EMAIL_SOURCE, EMAIL_REVISION, row
from v78_data import prepare, GENSPAM_SHA, NAZARIO_SHA
from v79_data import LID_SHA, PARQUETS, MERGED_REV


def fetch(url, target, expected):
    if not target.is_file():
        pending = target.with_suffix(target.suffix + '.partial')
        request = urllib.request.Request(url, headers={'User-Agent': 'MailGuard-research/79'})
        with urllib.request.urlopen(request, timeout=90) as response, pending.open('wb') as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if hashlib.file_digest(pending.open('rb'), 'sha256').hexdigest() != expected:
            raise ValueError('Downloaded input checksum mismatch: ' + target.name)
        pending.rename(target)
    if hashlib.file_digest(target.open('rb'), 'sha256').hexdigest() != expected:
        raise ValueError('Cached input checksum mismatch: ' + target.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        for name, expected in [('genspam.tar.gz', GENSPAM_SHA), ('nazario-2025.mbox', NAZARIO_SHA),
                ('labelled-mail.zip', 'a51c47191cb32e0f62e56e4e265a20be2393d326a003001e0830601f6f444eb7')]:
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('Email archive checksum mismatch: ' + name)
    raw = Path('.cache/v79/raw'); raw.mkdir(parents=True, exist_ok=True)
    fetch('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz', raw / 'lid.176.ftz', LID_SHA)
    for name, expected in PARQUETS.items():
        fetch('https://huggingface.co/datasets/locuoco/the-biggest-spam-ham-phish-email-dataset-300000/'
              'resolve/' + MERGED_REV + '/default/train/' + name, raw / name, expected)
    cache = Path('.cache/v77/rows.json.gz')
    if not cache.is_file():
        from datasets import load_dataset
        original = load_dataset(EMAIL_SOURCE, revision=EMAIL_REVISION)
        cache.parent.mkdir(parents=True, exist_ok=True)
        # v79 ignores the former Russian proxy: acquire the original email rows only.
        with gzip.open(cache, 'wt') as stream:
            stream.write('['); separator = ''
            for split in ('train', 'test'):
                for record in original[split]:
                    item = row(record['subject'], record['text'], record['label'], record['dataset_name'], split)
                    if item:
                        stream.write(separator + json.dumps(item, ensure_ascii=False)); separator = ','
            stream.write(']')
    prepare('.cache/v78/data.json.gz', archive_path=args.archive)
    print('Pinned inputs ready. Run tools/v79_data.py; no classifiers fitted.')


if __name__ == '__main__':
    main()
