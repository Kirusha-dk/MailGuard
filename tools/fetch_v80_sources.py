"""Download only the SHA256-pinned extra phishing years and neural encoder."""
import argparse
import json
from pathlib import Path

from fetch_v79_sources import fetch
from prepare_v80_phishing import ARCHIVE_SHA
from prepare_v80_encoder import SOURCE_SHA
from v80_encoder import MODEL, REVISION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('.cache/v80'))
    args = p.parse_args(); root = args.root
    encoder = root / 'encoder'; encoder.mkdir(parents=True, exist_ok=True)
    raw = root / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    files = []
    for year, sha in ARCHIVE_SHA.items():
        url = 'https://monkey.org/~jose/phishing/phishing-' + str(year)
        target = raw / ('nazario-' + str(year) + '.mbox')
        fetch(url, target, sha)
        files.append(dict(url=url, sha256=sha, file=str(target)))
    base = 'https://huggingface.co/' + MODEL + '/resolve/' + REVISION + '/'
    downloads = [
        (base + 'onnx/model_qint8_avx512_vnni.onnx', 'model-source-int8.onnx', SOURCE_SHA),
        (base + 'tokenizer.json', 'tokenizer.json', 'be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037'),
        (base + 'README.md', 'README.md', 'dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0'),
        ('https://www.apache.org/licenses/LICENSE-2.0.txt', 'LICENSE', 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'),
    ]
    for url, name, sha in downloads:
        fetch(url, encoder / name, sha)
        files.append(dict(url=url, sha256=sha, file=str(encoder / name)))
    (root / 'sources-pinned.json').write_text(json.dumps(dict(encoder_revision=REVISION, files=files), indent=2))
    print('Extra sources verified; no fitting or test scoring performed.')


if __name__ == '__main__':
    main()
