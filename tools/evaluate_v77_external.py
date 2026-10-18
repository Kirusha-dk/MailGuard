"""First frozen-v77 score on the historical v76 external English mail provider.

This source was examined in earlier project versions; source originality is
unverified. It is never used to fit v77 or choose its decision threshold.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import joblib
import v77_mail as runtime
import v76_source_audit as source
from v76_external_nn_firstuse import EXPECTED, classify_label
from train_v77_ru_en import fingerprint


def iter_json_array(stream, block_size=65536):
    """Read the development cache without materialising its message bodies."""
    decoder = json.JSONDecoder()
    buffer = stream.read(block_size).lstrip()
    if not buffer.startswith('['):
        raise ValueError('Expected cache array')
    buffer = buffer[1:]
    while True:
        buffer = buffer.lstrip()
        if buffer.startswith(','):
            buffer = buffer[1:].lstrip()
        if buffer.startswith(']'):
            return
        try:
            item, end = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            more = stream.read(block_size)
            if not more:
                raise ValueError('Truncated development cache')
            buffer += more
            continue
        yield item
        buffer = buffer[end:]


def acquire(cache):
    if cache.exists():
        with gzip.open(cache, 'rt') as stream:
            return json.load(stream)
    rows = []; seen = {}; conflicts = set(); counts = Counter()
    csv.field_size_limit(1000000)
    for name in source.FILES:
        raw = source.fetch(name)
        if hashlib.sha256(raw).hexdigest() != EXPECTED[name]:
            raise ValueError('Source content changed')
        for rec in csv.DictReader(io.StringIO(raw.decode('utf-8-sig', 'replace'))):
            counts['raw'] += 1
            if not str(rec.get('category_text') or '').strip():
                counts['unlabelled'] += 1
                continue
            y, _ = classify_label(rec, name)
            text = runtime.clean(rec.get('subject'), rec.get('body'))
            if runtime.language(text) != 'en' or len(text) < 30:
                counts['unsupported_or_short'] += 1
                continue
            identity = runtime.digest(text)
            if identity in seen:
                counts['duplicate'] += 1
                if seen[identity] != y:
                    conflicts.add(identity)
                continue
            seen[identity] = y
            rows.append(dict(identity=identity, family=runtime.family(text), text=text, y=y, lang='en'))
        print('Collected pinned external file', name, counts['raw'], flush=True)
    rows = [r for r in rows if r['identity'] not in conflicts]
    result = {'rows': rows, 'input_counts': dict(counts), 'conflicting_identities': len(conflicts)}
    with gzip.open(cache, 'wt') as stream:
        json.dump(result, stream)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='reports/v77/model.joblib')
    p.add_argument('--development-cache', default='.cache/v77/rows.json.gz')
    p.add_argument('--source-cache', default='.cache/v77/external.json.gz')
    p.add_argument('--output', default='reports/v77-external')
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    acquired = acquire(Path(args.source_cache))
    ids = set(); families = set()
    with gzip.open(args.development_cache, 'rt') as stream:
        for r in iter_json_array(stream):
            ids.add(r['identity']); families.add(r['family'])
    rows = [r for r in acquired['rows'] if r['identity'] not in ids and r['family'] not in families]
    rows = sorted(rows, key=lambda r: runtime.digest('v77-external-prefit-fixed:' + r['identity']))[:10000]
    if len(rows) < 1000:
        raise ValueError('Insufficient overlap-screened external mail')
    manifest = {'source': source.SOURCE, 'revision': source.REV, 'file_sha256': EXPECTED,
        'input_counts': acquired['input_counts'], 'eligible_before_prior_screen': len(acquired['rows']),
        'sample_count': len(rows), 'sample_fingerprint': fingerprint(rows),
        'label_blind_hash_sampling': True, 'not_used_for_training_or_thresholds': True,
        'previously_examined_by_older_versions': True, 'english_only': True,
        'original_message_provenance_unverified': True}
    (out / 'frozen.json').write_text(json.dumps(manifest, indent=2))
    modelpath = Path(args.model); sha = hashlib.sha256(modelpath.read_bytes()).hexdigest()
    bundle = joblib.load(modelpath)
    values = runtime.candidates(bundle['experts']['en'], [r['text'] for r in rows])
    truth = [r['y'] for r in rows]
    metrics = {name: runtime.measure(values[cfg['candidate']], truth, cfg['cutoff'])
               for name, cfg in bundle['policies']['en'].items()}
    if sha != hashlib.sha256(modelpath.read_bytes()).hexdigest():
        raise ValueError('Model modified by evaluation')
    report = {'version': runtime.VERSION, 'model_sha256': sha, 'manifest': manifest,
              'metrics': metrics, 'policies_unchanged': True, 'russian_email_validated': False}
    (out / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
