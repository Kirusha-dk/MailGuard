#!/usr/bin/env python3
"""Calibrate a lexical rescue while preserving frozen v51 decisions."""
import argparse
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
import torch
from benchmark_v48_calibrated import prepare, arrays, features
from rspamd_benchmark_audit import load_frozen_v40
from train_v47_phishing_aware import metrics_at
from train_v50_neural import ByteCNN, predict_neural
from train_v51_residual import predict_scores, residual_threshold, scored_union
from train_v52_wordnet import WordNet, predict_words


def load_components(root):
    artifact = joblib.load(Path(root) / 'model.joblib')
    byte, word = ByteCNN(), WordNet()
    for model, name in ((byte, 'neural.pt'), (word, 'word.pt')):
        model.load_state_dict(torch.load(Path(root) / name, map_location='cpu', weights_only=True)['state'])
    return artifact, byte, word


def component_scores(artifact, byte, word, texts):
    original = artifact['protectedArtifact']
    sparse = original['model'].predict_proba(original['tfidf'].transform(features(texts)))[:, 1]
    protected = predict_scores(original, sparse, predict_neural(byte, texts)) >= original['threshold']
    lexical = predict_words(word, texts)
    return protected, lexical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--output', default='reports/v53-smoke')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    def unique(pattern):
        files = list(Path('.cache').rglob(pattern))
        if len(files) != 1:
            raise RuntimeError(f'Expected one {pattern}, found {len(files)}')
        return files[0]
    basepath = unique('v51-full/model.joblib')
    wordpath = unique('v52-full/neural.pt')
    base = joblib.load(basepath)
    artifact = dict(base, version='v53-complement', protectedArtifact=base)
    shutil.copyfile(basepath.parent / 'neural.pt', out / 'neural.pt')
    shutil.copyfile(wordpath, out / 'word.pt')
    joblib.dump(artifact, out / 'model.joblib')
    artifact, byte, word = load_components(out)
    frozen, audit = load_frozen_v40('.cache/v40-fresh50k')
    data, dedup = prepare(frozen, args.limit)
    # Same calibration half as v51: no threshold adjustment using test labels.
    cal = [r for r in data['val'] if int(r['identity'][:8], 16) % 2 == 1]
    yc, sources = arrays(cal)
    protected, lexical = component_scores(artifact, byte, word, [r['ngram_text'] for r in cal])
    threshold, inherited = residual_threshold(yc, protected, lexical, int(len(cal) * .0012))
    artifact['threshold'] = threshold
    joblib.dump(artifact, out / 'model.joblib', compress=3)
    test = data['test']
    yt, tsources = arrays(test)
    pt, wt = component_scores(artifact, byte, word, [r['ngram_text'] for r in test])
    result = metrics_at(yt, scored_union(pt, wt, threshold), threshold, tsources)
    report = dict(version=53, test=result, calibration=metrics_at(yc, scored_union(protected, lexical, threshold), threshold, sources),
                  inheritedFP=inherited, calibrationSize=len(cal), threshold=threshold,
                  protectedV51=metrics_at(yt, pt.astype(float), .5, tsources),
                  audit=audit, dedup=dedup, baselineDecisionsPreserved=True,
                  warning='Reused development and engineering datasets; not independent production evidence.')
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    summary = ('# MailGuard v53 lexical rescue\n\n' + report['warning'] + '\n\n'
               '| N | Recall | FP | FN |\n|---:|---:|---:|---:|\n'
               f"| {len(test)} | {result['recall']:.2%} | {result['fp']} | {result['fn']} |\n")
    (out / 'report.md').write_text(summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
