"""Evaluate/resume one already locked model; never fit or change thresholds."""
import argparse
import copy
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import v77_mail as core
import v80_mail as runtime
from v79_data import fingerprint
from v80_encoder import file_sha
from v80_metrics import enrich


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def measure(values, rows, cutoff):
    return enrich(core.measure(values, [r['y'] for r in rows], cutoff))


def add_counts(target, metric):
    target.update({k: metric[k] for k in ('n', 'spam', 'ham', 'tp', 'fp', 'tn', 'fn')})


def evaluate(data, extra, output):
    model = output / 'model.joblib'
    locked = json.loads((output / 'policies-locked.json').read_text())
    sha = file_sha(model)
    if sha != locked['model_sha256']:
        raise ValueError('Model differs from pre-test lock')
    bundle = joblib.load(model)
    if bundle['policies'] != locked['policies']:
        raise ValueError('Policies differ from pre-test lock')
    runtime.attach(bundle, SimpleNamespace(model=model))
    manifest, extra_manifest = bundle['original_manifest'], bundle['extra_manifest']
    if manifest != json.loads((data / 'frozen.json').read_text()) or extra_manifest != json.loads((extra / 'frozen.json').read_text()):
        raise ValueError('Data manifests changed')
    runtime_hash = file_sha(Path(__file__).parent / 'v80_mail.py')
    encoder_hash = file_sha(Path(__file__).parent / 'v80_encoder.py')
    def scores(rows):
        features = runtime.feature_matrix(bundle, [r['text'] for r in rows])
        mask = features[:, -1].astype(bool)
        gate = dict(n=len(rows), passed=int(mask.sum()), spam=sum(r['y'] for r in rows),
                    spam_passed=sum(r['y'] for r, passed in zip(rows, mask) if passed))
        return runtime.candidate_scores(bundle, features), gate
    fresh = read_rows(extra / 'fresh-test.jsonl.gz')
    if fingerprint(fresh) != extra_manifest['fingerprints']['fresh-test']:
        raise ValueError('Fresh-test drift')
    values, gate = scores(fresh)
    metrics = {p: measure(values[c['candidate']], fresh, c['cutoff']) for p, c in bundle['policies'].items()}
    old = runtime.base_runtime.candidates(bundle['_base'], [r['text'] for r in fresh])
    baseline = {p: measure(old[c['candidate']], fresh, c['cutoff']) for p, c in bundle['_base']['policies'].items()}
    fresh_report = dict(v80=metrics, v79_same_fresh_test=baseline, n=len(fresh), ham=0, fast_gate=gate)
    (output / 'fresh-result.json').write_text(json.dumps(fresh_report, indent=2))
    print('FRESH 2024', json.dumps(fresh_report), flush=True)
    checkpoint = output / 'evaluation-blocks'; checkpoint.mkdir(exist_ok=True)
    blocks, totals, source_totals = [], {p: Counter() for p in runtime.PROFILES}, {}
    with gzip.open(data / 'test.jsonl.gz', 'rt') as stream:
        for number in range(10):
            rows = [json.loads(stream.readline()) for _ in range(10000)]
            block_id = fingerprint(rows)
            if block_id != manifest['blocks'][number]['fingerprint']:
                raise ValueError('Regression block drift')
            receipt = dict(model_sha256=sha, runtime_code_sha256=runtime_hash,
                           encoder_code_sha256=encoder_hash, block_fingerprint=block_id)
            path = checkpoint / (f'{number + 1:02d}.json')
            if path.is_file():
                block = json.loads(path.read_text())
                if block['receipt'] != receipt:
                    raise ValueError('Cannot reuse a different evaluation checkpoint')
            else:
                values, gate = scores(rows)
                metrics = {p: measure(values[c['candidate']], rows, c['cutoff']) for p, c in bundle['policies'].items()}
                sources = {}
                for source in sorted({r['source'] for r in rows}):
                    ix = [i for i, row in enumerate(rows) if row['source'] == source]
                    group = [rows[i] for i in ix]
                    sources[source] = {p: measure(values[c['candidate']][ix], group, c['cutoff']) for p, c in bundle['policies'].items()}
                block = dict(block=number + 1, receipt=receipt, metrics=metrics,
                             by_source=sources, fast_gate=gate)
                path.write_text(json.dumps(block, indent=2))
            blocks.append(block)
            for p, metric in block['metrics'].items():
                add_counts(totals[p], metric)
            for source, profiles in block['by_source'].items():
                for p, metric in profiles.items():
                    add_counts(source_totals.setdefault(source, {}).setdefault(p, Counter()), metric)
            print('REGRESSION BLOCK', number + 1, json.dumps(block['metrics']), flush=True)
        if stream.readline():
            raise ValueError('Unexpected extra regression emails')
    policies = copy.deepcopy(bundle['policies'])
    for cfg in policies.values():
        cfg['calibration'] = enrich(cfg['calibration']); cfg['selection'] = enrich(cfg['selection'])
    report = dict(version=runtime.VERSION, model_sha256=sha, policies=policies,
                  extra_manifest=extra_manifest, fresh_2024=fresh_report, regression_blocks=blocks,
                  regression_total={p: enrich(v) for p, v in totals.items()},
                  regression_by_source={s: {p: enrich(v) for p, v in profiles.items()} for s, profiles in source_totals.items()},
                  regression_100k_was_seen_in_v79=True, million_message_test_performed=False,
                  arbitrary_mail_guarantee=False, customer_mail_validated=False,
                  zero_fp_trials_needed_for_95pct_upper_10ppm_iid=math.ceil(math.log(.05) / math.log1p(-.00001)),
                  runtime_code_sha256=runtime_hash, encoder_code_sha256=encoder_hash)
    if file_sha(model) != sha:
        raise ValueError('Evaluation mutated the model')
    (output / 'result.json').write_text(json.dumps(report, indent=2))
    print('COMPLETE', json.dumps(report['regression_total']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('.cache/v79/frozen'))
    p.add_argument('--extra', type=Path, default=Path('.cache/v80/extra'))
    p.add_argument('--model-dir', type=Path, default=Path('reports/v80'))
    a = p.parse_args(); evaluate(a.data, a.extra, a.model_dir)
