#!/usr/bin/env python3
"""Replay frozen v62 on 500k distinct messages; report development overlap separately."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sqlite3
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

MODEL_SHA = '380991f74ae7ddc4d26cb36fd952ff7a1b6a24a1c8985becf28cb14d2f493aaf'
CORPUS_SHA = '5c0993c0258b854ce9cf2d5a31172d56692aaa351a533ca57c1ecc6e97c81f82'
COUNT = 500000
SHARDS = 5
PROFILES = ('conservative', 'high-recall')
CAVEAT = ('Previously inspected historical public corpus, predominantly spam. The full replay '
          'includes development messages and is not an independent quality estimate. '
          'Use the no-development-overlap subset for the engineering comparison; it is also '
          'previously inspected and is not customer validation. Distinct identity/body/template '
          'keys do not guarantee semantic independence. Few legitimate messages limit FP evidence.')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def model_file(root):
    candidates = list(Path(root).rglob('model.joblib'))
    if len(candidates) != 1:
        raise ValueError('Expected one frozen v62 model in artifact')
    return candidates[0]


def overlap(identity, family, identities, families):
    return [s for s in ('train', 'select', 'cal')
            if identity in identities[s] or family in families[s]]


def wilson_upper(errors, total):
    if not total:
        return None
    z = 1.959963984540054
    p = errors / total
    return (p + z*z/(2*total) + z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1+z*z/total)


def measure(c):
    tp, fn, tn, fp = (c.get(k, 0) for k in ('tp', 'fn', 'tn', 'fp'))
    spam, ham = tp+fn, tn+fp
    return dict(count=spam+ham, spam=spam, ham=ham, tp=tp, fn=fn, tn=tn, fp=fp,
                recall=tp/spam if spam else None, hamFpr=fp/ham if ham else None,
                hamFprWilson95Upper=wilson_upper(fp, ham),
                precision=tp/(tp+fp) if tp+fp else None)


def prepare(args):
    # Imports are lazy so aggregation and its tests do not depend on ML packages.
    import joblib
    import numpy as np
    from train_v55_multilingual import campaign_key, views
    from v61_semantic import training_pool
    from evaluate_v54_unique500k import content_keys
    root, out = Path(args.corpus), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    model = model_file(args.model)
    if file_sha(model) != MODEL_SHA:
        raise ValueError('Frozen v62 model SHA mismatch')
    artifact = joblib.load(model)
    if artifact['version'] != 'v62-script-nb-margin':
        raise ValueError('Wrong model version')
    data = Path(args.development)
    provenance = json.loads((data/'provenance.json').read_text())
    if artifact['splitManifestSha256'] != provenance['manifestSha256']:
        raise ValueError('Development provenance does not match model')
    identities, families = {}, {}
    train = None
    for split in ('train', 'select', 'cal'):
        path = data/f'{split}.jsonl.gz'
        if file_sha(path) != provenance['splitHashes'][split]:
            raise ValueError('Development split changed: '+split)
        batch = list(rows(path))
        identities[split] = {r['identity'] for r in batch}
        families[split] = {r['family'] for r in batch}
        if split == 'train':
            train = training_pool(batch)
    ix = np.random.default_rng(62).permutation(len(train))
    train_sha = hashlib.sha256('\n'.join(train[i]['identity'] for i in ix).encode()).hexdigest()
    if len(train) != artifact['trainCount'] or train_sha != artifact['trainIdentitySha256']:
        raise ValueError('Actual v62 training pool does not match provenance')
    del train, batch, artifact
    corpus = root/'selected.jsonl.gz'
    manifest = json.loads((root/'manifest.json').read_text())
    if file_sha(corpus) != CORPUS_SHA or manifest['selectedSha256'] != CORPUS_SHA:
        raise ValueError('Frozen 500k corpus SHA mismatch')
    if manifest['selected'] != COUNT or not manifest['uniqueAuditPassed']:
        raise ValueError('Expected exactly 500000 distinct messages')
    counts, sources, shard_counts = Counter(), Counter(), Counter()
    db = sqlite3.connect(out/'unique-audit.sqlite')
    db.execute('CREATE TABLE seen(identity TEXT PRIMARY KEY, body TEXT UNIQUE, template TEXT UNIQUE)')
    started = time.monotonic()
    with ExitStack() as stack:
        outputs = [stack.enter_context(gzip.open(out/f'input-{i}.jsonl.gz', 'wt', encoding='utf-8')) for i in range(SHARDS)]
        mf = stack.enter_context((root/'manifest.csv').open(newline=''))
        index = csv.DictReader(mf)
        count = 0
        for count, r in enumerate(rows(corpus), 1):
            entry = next(index, None)
            if type(r['y']) is not int or r['y'] not in (0, 1):
                raise ValueError('Invalid gold label')
            keys, _ = content_keys(r)
            if entry is None or (entry['identity'], entry['bodykey'], entry['templatekey'], entry['label'], entry['source']) != (r['identity'], keys[1], keys[2], str(r['y']), r['source']):
                raise ValueError('Corpus and independent manifest disagree')
            db.execute('INSERT INTO seen VALUES (?,?,?)', (r['identity'], keys[1], keys[2]))
            family = campaign_key(r['ngram_text'])
            blocked = overlap(r['identity'], family, identities, families)
            scope = 'development-overlap' if blocked else 'no-development-overlap'
            counts['all'] += 1
            counts[scope] += 1
            counts['ham' if r['y'] == 0 else 'spam'] += 1
            counts[scope+(':ham' if r['y'] == 0 else ':spam')] += 1
            for split in blocked:
                counts['overlap:'+split] += 1
            sources[r['source']] += 1
            focus, context = views(r['ngram_text'])
            shard = (count-1) % SHARDS
            outputs[shard].write(json.dumps(dict(identity=r['identity'], family=family, source=r['source'], y=r['y'],
                                               focus=focus, context=context, overlap=blocked), ensure_ascii=False)+'\n')
            shard_counts[shard] += 1
            if count % 50000 == 0:
                print('AUDIT', count, '/', COUNT, dict(counts), flush=True)
        if next(index, None) is not None or count != COUNT:
            raise ValueError('Wrong corpus or manifest row count')
    db.close()
    (out/'unique-audit.sqlite').unlink()
    if file_sha(model) != MODEL_SHA:
        raise ValueError('Model changed during audit')
    audit = dict(count=COUNT, counts=dict(counts), bySource=dict(sources), shardCounts=dict(shard_counts),
                 modelSha256=MODEL_SHA, corpusSha256=CORPUS_SHA, corpusManifest=manifest,
                 developmentManifestSha256=provenance['manifestSha256'], actualTrainIdentitySha256=train_sha,
                 auditSeconds=time.monotonic()-started, codeCommit=os.environ.get('GITHUB_SHA'),
                 githubRunId=os.environ.get('GITHUB_RUN_ID'), uniqueIdentityBodyTemplateAuditPassed=True,
                 fixedModelAndThresholds=True, freshTest=False, caveat=CAVEAT,
                 shardSha256={str(i):file_sha(out/f'input-{i}.jsonl.gz') for i in range(SHARDS)})
    (out/'audit.json').write_text(json.dumps(audit, indent=2))
    print('AUDIT COMPLETE', json.dumps(dict(counts)), flush=True)


def score(args):
    import numpy as np
    from itertools import islice
    from v62_runtime import load_bundle, score_rows, predict_texts
    from v61_semantic import script
    from train_v55_multilingual import views
    model = model_file(args.model)
    if file_sha(model) != MODEL_SHA:
        raise ValueError('Frozen v62 SHA mismatch')
    audit = json.loads(Path(args.audit).read_text())
    if audit['modelSha256'] != MODEL_SHA or file_sha(args.input) != audit['shardSha256'][str(args.shard)]:
        raise ValueError('Input or model does not match completed audit')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    bundle = load_bundle(model.parent)
    settings = json.dumps(bundle['profiles'], sort_keys=True)
    iterator = rows(args.input)
    count, started = 0, time.monotonic()
    prediction = out/f'predictions-{args.shard}.csv.gz'
    with gzip.open(prediction, 'wt', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['identity', 'family', 'source', 'script', 'label', 'overlap', 'conservativeMargin', 'highRecallMargin'])
        while batch := list(islice(iterator, 128)):
            scores = {p:score_rows(bundle, batch, p) for p in PROFILES}
            if not count:
                probe = batch[:8]
                # Check chunking and exact saved deployment path before bulk inference.
                for p in PROFILES:
                    baseline = score_rows(bundle, probe, p, batch_size=3)
                    replay = score_rows(bundle, probe, p, batch_size=1)
                    texts = ['SUBJECT: parity\nBODY: Confirm the meeting tomorrow.', 'SUBJECT: 会议\nBODY: 请确认会议时间谢谢']
                    canonical = predict_texts(bundle, texts, p)
                    prepared = [dict(zip(('focus', 'context'), views(t))) for t in texts]
                    deployment = score_rows(bundle, prepared, p, batch_size=1)
                    if not np.allclose(baseline, replay, rtol=1e-6, atol=1e-7) or not np.allclose(canonical, deployment, rtol=1e-6, atol=1e-7):
                        raise ValueError('Saved runtime parity failed')
            for i,r in enumerate(batch):
                values = [float(scores[p][i]) for p in PROFILES]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError('Nonfinite score')
                writer.writerow([r['identity'], r['family'], r['source'], script(r), r['y'], '|'.join(r['overlap']), *values])
            count += len(batch)
            if count % 8192 == 0:
                print('SCORE', args.shard, count, flush=True)
    if count != audit['shardCounts'][str(args.shard)] or file_sha(model) != MODEL_SHA or json.dumps(bundle['profiles'], sort_keys=True) != settings:
        raise ValueError('Count, model or threshold changed')
    metadata = dict(shard=args.shard, count=count, modelSha256=MODEL_SHA, inputSha256=file_sha(args.input),
                    predictionsSha256=file_sha(prediction), profiles=bundle['profiles'],
                    seconds=time.monotonic()-started, runtimeParityPassed=True)
    (out/f'shard-{args.shard}.json').write_text(json.dumps(metadata, indent=2))
    print('SCORE COMPLETE', args.shard, count, flush=True)


def aggregate(args):
    audit = json.loads(Path(args.audit).read_text())
    root, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    tally = defaultdict(Counter)
    identities, total, profiles = set(), 0, None
    shards = []
    for shard in range(SHARDS):
        candidates = list(root.rglob(f'shard-{shard}.json'))
        predictions = list(root.rglob(f'predictions-{shard}.csv.gz'))
        if len(candidates) != 1 or len(predictions) != 1:
            raise ValueError('Missing or duplicate shard '+str(shard))
        meta = json.loads(candidates[0].read_text())
        if meta['modelSha256'] != MODEL_SHA or meta['inputSha256'] != audit['shardSha256'][str(shard)] or file_sha(predictions[0]) != meta['predictionsSha256']:
            raise ValueError('Shard integrity mismatch')
        if profiles is None:
            profiles = meta['profiles']
        elif profiles != meta['profiles']:
            raise ValueError('Profiles differ between shards')
        local = 0
        with gzip.open(predictions[0], 'rt', newline='') as f:
            for r in csv.DictReader(f):
                if r['identity'] in identities:
                    raise ValueError('Repeated identity across shards')
                identities.add(r['identity'])
                y = int(r['label'])
                if y not in (0, 1):
                    raise ValueError('Invalid prediction label')
                scope = 'development-overlap' if r['overlap'] else 'no-development-overlap'
                groups = ('all', scope, scope+'/source/'+r['source'], scope+'/script/'+r['script'])
                for p, column in zip(PROFILES, ('conservativeMargin', 'highRecallMargin')):
                    margin = float(r[column])
                    if not math.isfinite(margin):
                        raise ValueError('Nonfinite prediction')
                    positive = margin >= 0
                    key = ('tp' if positive else 'fn') if y else ('fp' if positive else 'tn')
                    for name in groups:
                        tally[name, p][key] += 1
                total += 1
                local += 1
        if local != meta['count'] or local != audit['shardCounts'][str(shard)]:
            raise ValueError('Incomplete shard')
        shards.append(meta)
    if total != COUNT or audit['count'] != COUNT:
        raise ValueError('Incomplete 500k replay')
    results = {}
    for (name,p),counter in sorted(tally.items()):
        results.setdefault(name, {})[p] = measure(counter)
    for name in ('all', 'development-overlap', 'no-development-overlap'):
        for p in PROFILES:
            m = results.setdefault(name, {}).setdefault(p, measure({}))
            expected = audit['counts'].get('all' if name == 'all' else name, 0)
            if m['count'] != expected:
                raise ValueError('Scope count differs from audit')
    acceptance = {}
    for p in PROFILES:
        m = results['no-development-overlap'][p]
        acceptance[p] = dict(recall95AndFpAtMost50=m['recall'] is not None and m['recall'] >= .95 and m['fp'] <= 50,
                             rareFpPointGoalMet=m['hamFpr'] is not None and m['hamFpr'] <= .0001,
                             customerGoalValidated=False)
    report = dict(version=62, evaluation='500k-replay', modelSha256=MODEL_SHA, corpusSha256=CORPUS_SHA,
                  codeCommit=audit['codeCommit'], githubRunId=audit['githubRunId'], count=total,
                  profiles=profiles, results=results, acceptance=acceptance, audit=audit, shards=shards,
                  testLabelsUsedForTraining=False, thresholdsChanged=False, freshTest=False,
                  customerGoalValidated=False, caveat=CAVEAT)
    (out/'report.json').write_text(json.dumps(report, indent=2))
    md = '# MailGuard v62: 500000 messages\n\n'+CAVEAT+'\n\n| Scope | Profile | Count | Spam recall | FP / HAM | HAM FPR | 95% FPR upper bound |\n|---|---|---:|---:|---:|---:|---:|\n'
    for name in ('all', 'no-development-overlap', 'development-overlap'):
        for p in PROFILES:
            m = results[name][p]
            percent = lambda v: 'n/a' if v is None else f'{v:.4%}'
            md += f'| {name} | {p} | {m["count"]} | {percent(m["recall"])} | {m["fp"]} / {m["ham"]} | {percent(m["hamFpr"])} | {percent(m["hamFprWilson95Upper"])} |\n'
    (out/'report.md').write_text(md)
    print(md, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['prepare', 'score', 'aggregate'], required=True)
    p.add_argument('--corpus', default='.cache/v54-corpus')
    p.add_argument('--development', default='.cache/v60-data')
    p.add_argument('--model', default='.cache/v62')
    p.add_argument('--input')
    p.add_argument('--audit')
    p.add_argument('--shard', type=int, choices=range(SHARDS))
    p.add_argument('--output', required=True)
    args = p.parse_args()
    {'prepare':prepare, 'score':score, 'aggregate':aggregate}[args.phase](args)


if __name__ == '__main__':
    main()
