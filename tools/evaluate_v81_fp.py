"""Paired evaluation of one locked rescue; no fitting and no threshold changes."""
import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

import v81_mail as runtime
from train_v80_en import read_rows
from train_v81_fp import paired
from v79_data import fingerprint
from v80_encoder import file_sha
from v80_metrics import enrich
from v81_features import cached_fast, cached_neural


def evaluate(data, extra, output):
    model = output / 'model.joblib'
    locked = json.loads((output / 'policy-locked.json').read_text())
    if file_sha(model) != locked['model_sha256']:
        raise ValueError('Model differs from pre-regression lock')
    bundle = joblib.load(model)
    if bundle['policy'] != locked['policy']:
        raise ValueError('Policy differs from pre-regression lock')
    runtime.attach(bundle, SimpleNamespace(model=model))
    for location, key in ((data, 'original_manifest'), (extra, 'extra_manifest')):
        if json.loads((location / 'frozen.json').read_text()) != bundle[key]:
            raise ValueError('Corpus manifest drift')
    code = Path(__file__).parent
    receipt = dict(model_sha256=locked['model_sha256'],
                   code_sha256={name: file_sha(code / name) for name in
                                ('v81_mail.py', 'v81_features.py', 'evaluate_v81_fp.py',
                                 'train_v81_fp.py', 'v80_mail.py', 'v80_encoder.py', 'v77_mail.py')})
    cache = Path('.cache/v81/features'); cache.mkdir(parents=True, exist_ok=True)
    checkpoints = output / 'evaluation'; checkpoints.mkdir(exist_ok=True)
    old_model = output / 'baseline/model.joblib'; old = bundle['_baseline']; policy = bundle['policy']

    def cohort(name, rows):
        identity = fingerprint(rows)
        expected = dict(**receipt, fingerprint=identity)
        path = checkpoints / (name + '.json')
        if path.exists():
            result = json.loads(path.read_text())
            if result['receipt'] != expected:
                raise ValueError('Cannot reuse a different evaluation')
            return result
        fast = cached_fast(old_model, rows, cache / (name + '-fast.npy'))
        original = runtime.base_scores(fast) >= policy['baseline_cutoff']
        if policy['enabled']:
            features, embedding = cached_neural(old, rows, fast, original, cache / (name + '-embedding.npy'))
            scores = runtime.candidate_scores(bundle, policy['candidate'], features, embedding,
                                              runtime.lexical_features([r['text'] for r in rows]))
            proposed = runtime.apply_policy(fast, scores, policy)
        else:
            proposed = original.copy()
        def measure(indices):
            part = [rows[i] for i in indices]
            result = paired(original[indices], proposed[indices], part)
            for key in ('baseline', 'proposed'):
                result[key] = enrich(result[key])
            return result
        result = dict(receipt=expected, paired=measure(np.arange(len(rows))), by_source={},
                      lost_spam_identities=[r['identity'] for i, r in enumerate(rows)
                                            if r['y'] == 1 and original[i] and not proposed[i]])
        for source in sorted({r['source'] for r in rows}):
            ix = np.array([i for i, r in enumerate(rows) if r['source'] == source])
            result['by_source'][source] = measure(ix)
        path.write_text(json.dumps(result, indent=2))
        print('PAIRED', name, json.dumps(result['paired']), flush=True)
        return result

    fresh = read_rows(extra / 'fresh-test.jsonl.gz')
    if fingerprint(fresh) != bundle['extra_manifest']['fingerprints']['fresh-test']:
        raise ValueError('2024 regression drift')
    phish = cohort('nazario-2024-regression', fresh)
    blocks = []
    import gzip
    with gzip.open(data / 'test.jsonl.gz', 'rt') as stream:
        for number, expected in enumerate(bundle['original_manifest']['blocks'], 1):
            rows = [json.loads(stream.readline()) for _ in range(10000)]
            if fingerprint(rows) != expected['fingerprint']:
                raise ValueError('100k regression block drift')
            block = cohort('regression-' + str(number).zfill(2), rows)
            blocks.append(dict(block=number, **block))
        if stream.readline():
            raise ValueError('Extra regression messages')
    def aggregate(pairs):
        result = {key: Counter() for key in ('baseline', 'proposed')}
        changes = Counter()
        for pair in pairs:
            for key in result:
                result[key].update({k: pair[key][k] for k in ('n', 'spam', 'ham', 'tp', 'fp', 'tn', 'fn')})
            changes.update({k: pair[k] for k in ('baseline_spam_detections_lost', 'spam_detections_gained',
                                               'false_positives_removed', 'false_positives_added')})
        return dict(**{k: enrich(dict(v)) for k, v in result.items()}, **changes)
    total = aggregate([b['paired'] for b in blocks])
    sources = {s: aggregate([b['by_source'][s] for b in blocks if s in b['by_source']])
               for s in sorted({s for b in blocks for s in b['by_source']})}
    # Exact baseline parity is necessary before making any improvement claim.
    previous = json.loads(Path('reports/v80-result.json').read_text())['regression_total']['recall-98']
    if any(previous[k] != total['baseline'][k] for k in ('n', 'spam', 'ham', 'tp', 'fp', 'tn', 'fn')):
        raise ValueError('Baseline does not reproduce the published v80 result')
    if phish['paired']['baseline']['tp'] != 230:
        raise ValueError('Baseline does not reproduce the published 2024 result')
    accepted = (total['false_positives_removed'] > 0 and total['baseline_spam_detections_lost'] == 0
                and total['false_positives_added'] == 0
                and phish['paired']['baseline_spam_detections_lost'] == 0)
    report = dict(version=runtime.VERSION, receipt=receipt, policy=policy,
                  development=bundle['development'], regression_total=total,
                  regression_blocks=blocks, regression_by_source=sources,
                  phishing_2024_regression=phish, accepted_on_observed_regression=bool(accepted),
                  regression_is_a_new_blind_test=False, million_message_test_performed=False,
                  arbitrary_future_mail_guarantee=False, customer_mail_validated=False,
                  rejected_candidate_must_not_replace_baseline=not accepted)
    if file_sha(model) != locked['model_sha256']:
        raise ValueError('Evaluation mutated the model')
    (output / 'result.json').write_text(json.dumps(report, indent=2))
    print('COMPLETE', json.dumps(dict(accepted=accepted, total=total,
                                      phish=phish['paired'])), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('.cache/v79/frozen'))
    p.add_argument('--extra', type=Path, default=Path('.cache/v80/extra'))
    p.add_argument('--model-dir', type=Path, default=Path('reports/v81'))
    a = p.parse_args(); evaluate(a.data, a.extra, a.model_dir)
