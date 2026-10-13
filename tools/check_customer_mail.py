#!/usr/bin/env python3
"""Offline evaluation of a frozen MailGuard bundle on locally labelled .eml files."""
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
    parser.add_argument('--profile', choices=('conservative', 'high-recall'), default='conservative')
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
    torch.set_num_threads(4)
    if artifact.get('version') in ('v56-consensus', 'v57-preserved-consensus', 'v58-calibrated-protected-consensus', 'v59-multilingual-ham', 'v60-unicode-ensemble', 'v61-semantic-neighbour-ensemble', 'v62-script-nb-margin'):
        if artifact['version'] == 'v62-script-nb-margin':
            from v62_runtime import load_bundle, predict_texts
            from v59_mail import from_eml as parse_v59
        elif artifact['version'] == 'v61-semantic-neighbour-ensemble':
            from v61_runtime import load_bundle, predict_texts
            from v59_mail import from_eml as parse_v59
        elif artifact['version'] == 'v60-unicode-ensemble':
            from v60_runtime import load_bundle, predict_texts
            from v59_mail import from_eml as parse_v59
        elif artifact['version'] == 'v59-multilingual-ham':
            from v59_runtime import load_bundle, predict_texts
            from v59_mail import from_eml as parse_v59
        elif artifact['version'] == 'v58-calibrated-protected-consensus':
            from v58_runtime import load_bundle, predict_texts
        elif artifact['version'] == 'v57-preserved-consensus':
            from v57_runtime import load_bundle, predict_texts
        else:
            from train_v56_consensus import load_bundle, predict_texts
        bundle = load_bundle(modeldir)
        threshold = artifact['profiles'][args.profile]['threshold']
        version, profile = ({'v56-consensus':56,'v57-preserved-consensus':57,'v58-calibrated-protected-consensus':58,'v59-multilingual-ham':59,'v60-unicode-ensemble':60,'v61-semantic-neighbour-ensemble':61,'v62-script-nb-margin':62}[artifact['version']]), args.profile
        checkpoint = modelpath if version == 62 else modeldir / ('encoder/model_quint8_avx2.onnx' if version == 61 else 'unicode.pt' if version == 60 else 'v54/neural.pt')
        component_hashes = artifact['components']
        def score_texts(texts, originals=None):
            if version in (59, 60, 61, 62):
                return predict_texts(bundle, texts, profile, original_texts=originals)
            return predict_texts(bundle, texts, profile)
    elif artifact.get('version') == 'v51-residual-neural':
        if args.profile != 'conservative':
            parser.error('--profile high-recall is available only for v56')
        neural = ByteCNN()
        neural.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state'])
        threshold = artifact['threshold']
        version, profile, component_hashes = 51, None, None
        def score_texts(texts, originals=None):
            sparse = artifact['model'].predict_proba(artifact['tfidf'].transform(features(texts)))[:, 1]
            return predict_scores(artifact, sparse, predict_neural(neural, texts))
    else:
        parser.error('This checker requires a frozen v51, v56, v57, v58, v59, v60, v61 or v62 bundle')
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
                    row = parse_v59(raw) if version in (59, 60, 61, 62) else from_eml(raw)
                    if version in (59, 60, 61, 62):
                        row['original_text'] = from_eml(raw)['ngram_text']
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
                originals = [r['original_text'] for r, _, _ in batch] if version in (59, 60, 61, 62) else None
                scores = np.asarray(score_texts(texts, originals), dtype=np.float64)
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
    summary = dict(version=version, profile=profile, found=len(files), scored=spam + ham, failed=failures,
                   spam=spam, ham=ham, **counts,
                   spamRecall=counts['tp'] / spam if spam else None,
                   falsePositiveRate=counts['fp'] / ham if ham else None,
                   duplicateContent=duplicates, conflictingDuplicateLabels=conflicting,
                   threshold=float(threshold), thresholdChanged=False,
                   modelSha256=hashlib.sha256(modelpath.read_bytes()).hexdigest(),
                   checkpointSha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   componentHashes=component_hashes)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    recall = f"{summary['spamRecall']:.2%}" if spam else 'n/a'
    fpr = f"{summary['falsePositiveRate']:.4%}" if ham else 'n/a'
    text = (f'MailGuard v{version}' + (f' / {profile}' if profile else '') + f'\nScored: {spam + ham}/{len(files)}; failed: {failures}\n'
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


