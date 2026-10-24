"""Acquire checksum-pinned public English corpus and pretrained float weights."""
import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from fetch_v79_sources import fetch
from prepare_v80_phishing import ARCHIVE_SHA
from v78_data import GENSPAM_SHA, GENSPAM_URL, NAZARIO_SHA, NAZARIO_URL
from v82_neural import MODEL, REVISION

ROOT = Path('.cache/v82')
FILES = ('config.json', 'tokenizer.json', 'tokenizer_config.json',
         'special_tokens_map.json', 'vocab.txt', 'model.safetensors')
WEIGHT_SHA = '53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db'


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def backbone():
    out = ROOT / 'backbone'; out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest['model'] != MODEL or manifest['revision'] != REVISION or set(manifest['files']) != set(FILES):
            raise ValueError('Backbone manifest mismatch')
        for name, sha in manifest['files'].items():
            if file_sha(out / name) != sha:
                raise ValueError('Cached pretrained file changed: ' + name)
        if manifest['files']['model.safetensors'] != WEIGHT_SHA:
            raise ValueError('Unexpected pretrained float weights')
        return
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true', timeout=90) as response:
        info = json.load(response)
    if info['sha'] != REVISION:
        raise ValueError('Pretrained revision drift')
    siblings = {item['rfilename']: item for item in info['siblings']}
    hashes = {}
    for name in FILES:
        target = out / name
        if not target.is_file():
            pending = target.with_suffix(target.suffix + '.partial')
            with urllib.request.urlopen(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}', timeout=120) as response, pending.open('wb') as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            pending.replace(target)
        item = siblings[name]
        content = target.read_bytes()
        if item.get('lfs'):
            valid = hashlib.sha256(content).hexdigest() == item['lfs']['sha256']
        else:
            valid = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() == item['blobId']
        if not valid:
            raise ValueError('Pretrained checksum mismatch: ' + name)
        hashes[name] = hashlib.sha256(content).hexdigest()
    if hashes['model.safetensors'] != WEIGHT_SHA:
        raise ValueError('Unexpected pretrained float weights')
    manifest_path.write_text(json.dumps(dict(model=MODEL, revision=REVISION, files=hashes), indent=2))
    print('Pinned pretrained safetensors ready', flush=True)


def corpus():
    data = Path('.cache/v79/frozen'); extra = Path('.cache/v80/extra')
    if data.exists() and extra.exists():
        for directory, reference in ((data, 'reports/v79-frozen.json'), (extra, 'reports/v80-extra-frozen.json')):
            if json.loads((directory / 'frozen.json').read_text()) != json.loads(Path(reference).read_text()):
                raise ValueError('Public corpus differs from the published freeze: ' + str(directory))
        print('Verified frozen English corpus', flush=True)
        return
    raw = ROOT / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    sources = [('genspam.tar.gz', GENSPAM_URL, GENSPAM_SHA),
               ('nazario-2025.mbox', NAZARIO_URL, NAZARIO_SHA),
               ('labelled-mail.zip', 'https://codeload.github.com/realprogrammersusevim/email-dataset/zip/84209612df4c55831f074a65b7c137a4b980059d',
                'a51c47191cb32e0f62e56e4e265a20be2393d326a003001e0830601f6f444eb7')]
    for name, url, sha in sources:
        fetch(url, raw / name, sha)
    archive = ROOT / 'public-sources.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as output:
        for name, _, _ in sources:
            output.write(raw / name, name)
    if not data.exists():
        subprocess.run([sys.executable, 'tools/fetch_v79_sources.py', '--archive', str(archive)], check=True)
        subprocess.run([sys.executable, 'tools/v79_data.py', '--archive', str(archive)], check=True)
    if not extra.exists():
        extra_raw = Path('.cache/v80/raw'); extra_raw.mkdir(parents=True, exist_ok=True)
        for year, sha in ARCHIVE_SHA.items():
            fetch('https://monkey.org/~jose/phishing/phishing-' + str(year), extra_raw / f'nazario-{year}.mbox', sha)
        subprocess.run([sys.executable, 'tools/prepare_v80_phishing.py'], check=True)
    corpus()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone-only', action='store_true')
    args = parser.parse_args()
    backbone()
    if not args.backbone_only:
        corpus()
