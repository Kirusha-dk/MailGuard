"""Prepare separate RU/ES/TR pilot inputs; never substitute chats or invent labels."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
FILES = {
    'model.safetensors': 'eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b',
    'config.json': '6300193cb75e01cf80c96decef7187dfb33094d97cc1490b7ead6ff134476e4e',
    'tokenizer.json': '2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8',
    'tokenizer_config.json': '5036ea374ffedd706e3bef33e2e0d6953cb868ef8a490e76e32ba0faa37a6b9b',
    'special_tokens_map.json': '378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d'
}
LID_SHA = '8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(url, path, expected, limit=500_000_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pending = path.with_suffix(path.suffix + '.partial')
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'MailGuard-v83-research'})
            total = 0
            with urllib.request.urlopen(request, timeout=120) as response, pending.open('wb') as stream:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > limit:
                        raise ValueError('Public input exceeds its pinned size budget')
                    stream.write(chunk)
            if sha(pending) != expected:
                raise ValueError('Downloaded public input checksum mismatch: ' + path.name)
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
    if sha(path) != expected:
        raise ValueError('Cached public input checksum mismatch: ' + path.name)


def backbone(root):
    for name, expected in FILES.items():
        fetch(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}', root / name, expected)
    manifest = dict(model=MODEL, revision=REVISION, files=FILES)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Pinned multilingual pretrained backbone verified', flush=True)


def prepare(language, root):
    import fasttext
    from fetch_v83_sources import fetch_spanish, fetch_turkish
    from v83_data import freeze
    output = root / language
    if output.exists():
        raise ValueError('Language corpus already exists; use its verified manifest rather than overwrite')
    if language == 'ru':
        audit = dict(language='ru', training_ready=False, optimizer_steps_executed=0,
                     blocked_reason='A sufficiently labelled native Russian EMAIL SPAM/HAM corpus has not been established.',
                     rejected_substitutes=['Telegram/SMS/social posts', 'translated or synthetic messages presented as native email',
                                          'SPAM-versus-phishing data with no HAM'],
                     customer_mail_used=False, production_acceptance=False)
        output.mkdir(parents=True)
        (output / 'data-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        return audit
    rows, source = (fetch_turkish(root / 'raw') if language == 'tr' else fetch_spanish(root / 'raw'))
    lid_path = root / 'raw/lid.176.ftz'
    fetch('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz', lid_path, LID_SHA, limit=2_000_000)
    lid = fasttext.load_model(str(lid_path))
    def predict(text):
        import re
        view = re.sub(r'\b(?:numbertoken|urltoken|emailtoken)\b', ' ', text)
        view = ' '.join(view[:4000].split())
        result = lid.f.predict(view + '\n', 1, 0., 'strict')
        return (result[0][1].removeprefix('__label__'), result[0][0]) if result else ('unknown', 0.)
    return freeze(rows, language, predict, output, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=['ru', 'es', 'tr'], required=True)
    parser.add_argument('--root', type=Path, default=Path('.cache/v83'))
    parser.add_argument('--backbone-only', action='store_true')
    args = parser.parse_args()
    if args.backbone_only:
        backbone(args.root / 'backbone')
        return
    manifest = prepare(args.language, args.root)
    print(json.dumps({k: manifest.get(k) for k in ('language', 'training_ready', 'counts', 'blocked_reason')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
