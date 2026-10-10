#!/usr/bin/env python3
"""Offline evaluation of frozen v51 on locally labelled .eml files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from benchmark_v48_calibrated import features
from train_v47_phishing_aware import from_eml
from train_v50_neural import ByteCNN, predict_neural
from train_v51_residual import predict_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--spam', required=True)
    parser.add_argument('--ham', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    roots = [(Path(args.ham).resolve(), 0), (Path(args.spam).resolve(), 1)]
    if roots[0][0] == roots[1][0] or any(a in b.parents for a, _ in roots for b, _ in roots if a != b):
        parser.error('Spam and ham folders must be separate, not nested')
    files = []
    for root, label in roots:
        if not root.is_dir():
            parser.error(f'Folder not found: {root}')
        group = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.eml')
        if not group:
            parser.error(f'No .eml files in {root}')
        files.extend((p, label, str(p.relative_to(root))) for p in group)
    out = Path(args.output).resolve()
    if any(out == root or root in out.parents for root, _ in roots):
        parser.error('Output folder must be outside input folders')
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ('summary.json', 'summary.txt', 'predictions.csv', 'errors.csv')):
        parser.error('Output already contains a report; choose a new output folder')
    modeldir = Path(args.model_dir)
    modelpath, checkpoint = modeldir / 'model.joblib', modeldir / 'neural.pt'
    artifact = joblib.load(modelpath)
    if artifact.get('version') != 'v51-residual-neural':
        parser.error('This checker requires the frozen v51 model')
    torch.set_num_threads(4)
    neural = ByteCNN()
    neural.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state'])
    threshold = artifact['threshold']
    counts = dict(tp=0, fn=0, fp=0, tn=0)
    failures = 0
    seen, duplicates, conflicting = {}, 0, 0
    with (out / 'predictions.csv').open('w', newline='', encoding='utf-8') as pf, (out / 'errors.csv').open('w', newline='', encoding='utf-8') as ef:
        predictions, errors = csv.writer(pf), csv.writer(ef)
        predictions.writerow(['folder', 'file', 'label', 'predicted', 'decision_score', 'identity'])
        errors.writerow(['folder', 'file', 'type'])
        for start in range(0, len(files), args.batch_size):
            batch = []
            for path, label, relative in files[start:start + args.batch_size]:
                folder = 'spam' if label else 'ham'
                try:
                    raw = path.read_bytes()
                    if not raw.strip():
                        raise ValueError('Empty message')
                    row = from_eml(raw)
                except (OSError, ValueError):
                    failures += 1
                    errors.writerow([folder, relative, 'unreadable_or_empty'])
                    continue
                identity = row['identity']
                if identity in seen:
                    duplicates += 1
                    conflicting += int(seen[identity] != label)
                else:
                    seen[identity] = label
                batch.append((row, label, relative))
            if batch:
                texts = [r['ngram_text'] for r, _, _ in batch]
                sparse = artifact['model'].predict_proba(artifact['tfidf'].transform(features(texts)))[:, 1]
                scores = predict_scores(artifact, sparse, predict_neural(neural, texts))
                for (row, label, relative), score in zip(batch, scores):
                    predicted = int(score >= threshold)
                    key = ('tp' if predicted else 'fn') if label else ('fp' if predicted else 'tn')
                    counts[key] += 1
                    folder = 'spam' if label else 'ham'
                    predictions.writerow([folder, relative, label, predicted, float(score), row['identity']])
                    if key in ('fp', 'fn'):
                        errors.writerow([folder, relative, key])
            print(f'Processed {min(start + args.batch_size, len(files))}/{len(files)}', flush=True)
    spam, ham = counts['tp'] + counts['fn'], counts['tn'] + counts['fp']
    summary = dict(version=51, found=len(files), scored=spam + ham, failed=failures,
                   spam=spam, ham=ham, **counts,
                   spamRecall=counts['tp'] / spam if spam else None,
                   falsePositiveRate=counts['fp'] / ham if ham else None,
                   duplicateContent=duplicates, conflictingDuplicateLabels=conflicting,
                   threshold=float(threshold), thresholdChanged=False,
                   modelSha256=hashlib.sha256(modelpath.read_bytes()).hexdigest(),
                   checkpointSha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    recall = f"{summary['spamRecall']:.2%}" if spam else 'n/a'
    fpr = f"{summary['falsePositiveRate']:.4%}" if ham else 'n/a'
    text = (f'MailGuard v51\nScored: {spam + ham}/{len(files)}; failed: {failures}\n'
            f'Spam: {spam}; ham: {ham}\nSpam recall: {recall}\n'
            f"FP: {counts['fp']}; FPR (FP/ham): {fpr}\nFN: {counts['fn']}\n"
            f'Duplicate content: {duplicates}; conflicting labels: {conflicting}\n'
            'All readable files are counted, including duplicates. Threshold unchanged.\n'
            'Decision scores are internal classifier values, not calibrated probabilities.\n')
    (out / 'summary.txt').write_text(text, encoding='utf-8')
    print(text)
    return 2 if failures or not spam or not ham else 0


if __name__ == '__main__':
    raise SystemExit(main())
