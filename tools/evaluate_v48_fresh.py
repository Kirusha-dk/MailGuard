#!/usr/bin/env python3
"""Evaluate the exact v48 artifact without fitting or threshold adjustment."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from datasets import load_dataset
from benchmark_v48_calibrated import features
from rspamd_benchmark_audit import load_frozen_v40
from train_v47_phishing_aware import canonical_fields, from_eml, metrics_at


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-root', default='.cache/v48-artifact')
    parser.add_argument('--count', type=int, default=50000)
    parser.add_argument('--max-fp', type=int, default=100)
    parser.add_argument('--output', default='reports/v48-fresh')
    parser.add_argument('--model-subdir', default='v48-full')
    parser.add_argument('--min-recall', type=float, default=.90)
    args = parser.parse_args()
    if args.count <= 0 or args.max_fp < 0:
        parser.error('Invalid count or FP budget')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    files = list(Path(args.artifact_root).rglob(args.model_subdir + '/model.joblib'))
    if len(files) != 1:
        raise RuntimeError(f'Expected one frozen full v48 model, found {files}')
    model_path = files[0]
    artifact = joblib.load(model_path)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    frozen, audit = load_frozen_v40('.cache/v40-fresh50k')
    blocked = set()
    for split in ('train', 'val', 'test'):
        for r in frozen[split]:
            blocked.add(from_eml(r['path'].read_bytes())['identity'])
    candidates, conflicts = {}, set()
    excluded = Counter()

    def add(subject, body, sender, label, source):
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f'Invalid label in {source}')
        c = canonical_fields(subject, body, sender)
        key = c['identity']
        if key in blocked:
            excluded['oldContent'] += 1
            return
        if len(str(body or '').strip()) < 20:
            excluded['shortContent'] += 1
            return
        if key in candidates:
            excluded['duplicateCandidate'] += 1
            if candidates[key]['y'] != label:
                conflicts.add(key)
            return
        candidates[key] = dict(c, y=label, source=source)

    ds = load_dataset('JinqiangDing/seven-phishing-email-datasets', split='train')
    revisions = {'seven': str(ds._fingerprint)}
    for r in ds:
        # Match the neutral fallback used in frozen v40's sanitized renderer.
        add(r.get('subject') or '', r.get('text') or '',
            r.get('sender') or 'unknown@example.invalid', int(r['label']),
            str(r.get('dataset_name') or 'unknown'))
    extra = load_dataset('SetFit/enron_spam')
    for split, rows in extra.items():
        revisions['SetFit/' + split] = str(rows._fingerprint)
        for r in rows:
            add('', r['text'], 'unknown@example.invalid', int(r['label']), 'SetFit-Enron')
    pool = [r for k, r in candidates.items() if k not in conflicts]
    # Selection uses only a seeded content hash; labels and model scores do not rank rows.
    pool.sort(key=lambda r: hashlib.sha256(('v48-fresh-20261010:' + r['identity']).encode()).hexdigest())
    rows = pool[:args.count]
    manifest = dict(modelSha256=model_hash, threshold=float(artifact['threshold']),
                    oldFingerprint=audit, datasetFingerprints=revisions,
                    available=len(pool), selected=len(rows), exclusions=dict(excluded),
                    conflictingCandidates=len(conflicts),
                    selection='seeded content hash; no score-based or class-based selection',
                    warning='Unseen by v48 after exact-content exclusion; historical sources, not proof of production stability. Near-duplicates may remain.',
                    rows=[{'identity':r['identity'], 'source':r['source'], 'label':r['y']} for r in rows])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if not rows or len(set(r['y'] for r in rows)) != 2:
        raise RuntimeError('Fresh evaluation lacks usable rows of both classes')
    complement = None
    if artifact.get('version') == 'v53-complement':
        import torch
        from train_v53_complement import load_components, component_scores
        from train_v51_residual import scored_union
        torch.set_num_threads(4)
        complement = load_components(model_path.parent)
    neural = None
    if complement is None and artifact.get('neuralWeight', 0) > 0:
        import torch
        from train_v50_neural import ByteCNN, predict_neural, mix
        torch.set_num_threads(4)
        if artifact.get('neuralArchitecture') == 'word-hash-v52':
            from train_v52_wordnet import WordNet, predict_words
            neural = WordNet()
            predict_neural = predict_words
        else:
            neural = ByteCNN()
        checkpoint = torch.load(model_path.parent / artifact['neuralCheckpoint'], map_location='cpu', weights_only=True)
        neural.load_state_dict(checkpoint['state'])
    scores = []
    for start in range(0, len(rows), 1000):
        batch = rows[start:start + 1000]
        x = artifact['tfidf'].transform(features([r['ngram_text'] for r in batch]))
        if artifact.get('nbRatio') is not None:
            x = x.multiply(artifact['nbRatio']).tocsr()
        prediction = artifact['model'].predict_proba(x)[:, 1]
        if neural is not None:
            npred = predict_neural(neural, [r['ngram_text'] for r in batch])
            if artifact.get('version') == 'v51-residual-neural':
                from train_v51_residual import predict_scores
                prediction = predict_scores(artifact, prediction, npred)
            else:
                prediction = mix(prediction, npred, artifact['neuralWeight'])
        if complement is not None:
            protected, lexical = component_scores(*complement, [r['ngram_text'] for r in batch])
            prediction = scored_union(protected, lexical, artifact['threshold'])
        scores.extend(prediction.tolist())
        print('fresh v48 scored', start + len(batch), flush=True)
    y = np.asarray([r['y'] for r in rows])
    sources = [r['source'] for r in rows]
    p = np.asarray(scores)
    metrics = metrics_at(y, p, artifact['threshold'], sources)
    smoke = metrics_at(y[:1000], p[:1000], artifact['threshold'], sources[:1000])
    passed = len(rows) == args.count and metrics['recall'] >= args.min_recall and metrics['fp'] <= args.max_fp
    report = dict(minRecall=args.min_recall, requestedCount=args.count, maxFp=args.max_fp, count=len(rows), available=len(pool), modelSha256=model_hash,
                  thresholdUnchanged=True, modelRefit=False, testLabelsUsedForTuning=False,
                  smoke1000=smoke, metrics=metrics, goalPassed=passed,
                  warning=manifest['warning'])
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    with (out / 'predictions.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['identity', 'source', 'label', 'score', 'predictedSpam'])
        for r, score in zip(rows, p):
            w.writerow([r['identity'], r['source'], r['y'], float(score), int(score >= artifact['threshold'])])
    text = ('# Frozen v48 on unused public mail\n\n' + manifest['warning'] + '\n\n'
            '| N | Recall | FP | FN |\n|---:|---:|---:|---:|\n'
            f"| {len(rows)} | {metrics['recall']:.2%} | {metrics['fp']} | {metrics['fn']} |\n\n"
            f'{args.count}-message quality goal passed: {passed}\n')
    if len(rows) < args.count:
        text += f'Insufficient unique unseen mail: short by {args.count-len(rows)}. No duplicated refill.\n'
    (out / 'report.md').write_text(text)
    print(text, flush=True)


if __name__ == '__main__':
    main()
