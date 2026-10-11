#!/usr/bin/env python3
"""Supervised byte CNN complement to the immutable v48 estimator."""
import argparse
import copy
import json
import random
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from benchmark_v48_calibrated import features, prepare, arrays, threshold_for_budget
from rspamd_benchmark_audit import load_frozen_v40
from train_v47_phishing_aware import metrics_at

WINDOW = 1024


def encode(texts):
    result = np.zeros((len(texts), WINDOW * 2), dtype=np.uint8)
    for i, text in enumerate(texts):
        raw = text.encode('utf-8', 'replace')
        head, tail = raw[:WINDOW], raw[-WINDOW:]
        result[i, :len(head)] = np.frombuffer(head, dtype=np.uint8)
        result[i, WINDOW:WINDOW + len(tail)] = np.frombuffer(tail, dtype=np.uint8)
    return result


class ByteCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 24, padding_idx=0)
        self.branches = nn.ModuleList([nn.Conv1d(24, 48, k) for k in (3, 5, 9)])
        self.classifier = nn.Sequential(nn.Linear(288, 96), nn.ReLU(), nn.Dropout(.25), nn.Linear(96, 1))

    def forward(self, x):
        # Shared filters, independent head and tail windows; no artificial boundary ngrams.
        pooled = []
        for window in (x[:, :WINDOW], x[:, WINDOW:]):
            embedded = self.embedding(window.long()).transpose(1, 2)
            pooled.extend(torch.relu(conv(embedded)).amax(dim=2) for conv in self.branches)
        return self.classifier(torch.cat(pooled, dim=1)).squeeze(1)


def predict_neural(model, texts, batch_size=128):
    loader = DataLoader(TensorDataset(torch.from_numpy(encode(texts))), batch_size=batch_size)
    model.eval()
    scores = []
    with torch.inference_mode():
        for (x,) in loader:
            scores.extend(torch.sigmoid(model(x)).cpu().numpy().tolist())
    return np.asarray(scores)


def mix(sparse, neural, weight):
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    z = (1 - weight) * logit(sparse) + weight * logit(neural)
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--output', default='reports/v50-smoke')
    args = parser.parse_args()
    random.seed(50)
    np.random.seed(50)
    torch.manual_seed(50)
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frozen, audit = load_frozen_v40('.cache/v40-fresh50k')
    data, dedup = prepare(frozen, args.limit)
    train, val = data['train'], data['val']
    selection = [r for r in val if int(r['identity'][:8], 16) % 2 == 0]
    calibration = [r for r in val if int(r['identity'][:8], 16) % 2 == 1]
    text = lambda rows: [r['ngram_text'] for r in rows]
    yt, _ = arrays(train)
    ys, ss = arrays(selection)
    yc, sc = arrays(calibration)
    loader = DataLoader(TensorDataset(torch.from_numpy(encode(text(train))), torch.tensor(yt, dtype=torch.float32)),
                        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(50))
    model = ByteCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss, best_state, best_epoch = float('inf'), None, None
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            train_loss += loss.item() * len(y)
        p = predict_neural(model, text(selection))
        valid_loss = float(np.mean(-(ys * np.log(np.clip(p, 1e-6, 1)) + (1 - ys) * np.log(np.clip(1 - p, 1e-6, 1)))))
        history.append(dict(epoch=epoch + 1, trainLoss=train_loss / len(train), selectionLoss=valid_loss))
        print('v50 epoch', history[-1], flush=True)
        if valid_loss < best_loss:
            best_loss, best_state, best_epoch = valid_loss, copy.deepcopy(model.state_dict()), epoch + 1
    model.load_state_dict(best_state)
    torch.save({'state': best_state, 'window': WINDOW, 'epoch': best_epoch}, out / 'neural.pt')
    files = list(Path('.cache/v48-artifact').rglob('v48-full/model.joblib'))
    if len(files) != 1:
        raise RuntimeError('Missing immutable v48 artifact')
    artifact = joblib.load(files[0])
    def sparse(rows):
        return artifact['model'].predict_proba(artifact['tfidf'].transform(features(text(rows))))[:, 1]
    ps, ns = sparse(selection), predict_neural(model, text(selection))
    # Original v48 development rate: retain it to isolate neural changes.
    rate = .0012
    candidates = []
    for weight in (0., .1, .25, .5, .75, 1.):
        p = mix(ps, ns, weight)
        t = threshold_for_budget(ys, p, int(len(selection) * rate))
        m = metrics_at(ys, p, t, ss)
        candidates.append(dict(weight=weight, metrics=m))
    chosen = max(candidates, key=lambda c: (c['metrics']['recall'], -c['metrics']['fp'], -c['weight']))
    weight = chosen['weight']
    pc = mix(sparse(calibration), predict_neural(model, text(calibration)), weight)
    threshold = threshold_for_budget(yc, pc, int(len(calibration) * rate))
    # Weight zero has exactly the original v48 representation and calibration partition.
    artifact.update(version='v50-byte-cnn-blend', neuralWeight=weight, neuralCheckpoint='neural.pt', threshold=threshold)
    joblib.dump(artifact, out / 'model.joblib', compress=3)
    test = data['test']
    ptest = mix(sparse(test), predict_neural(model, text(test)), weight)
    ytest, stest = arrays(test)
    result = metrics_at(ytest, ptest, threshold, stest)
    report = dict(goal=dict(recall=.98, maxFp=100, count=100000), history=history, selectedEpoch=best_epoch,
                  selectedWeight=weight, candidates=candidates, calibration=metrics_at(yc, pc, threshold, sc),
                  count=len(test), test=result, frozenAudit=audit, dedup=dedup, refitAfterCalibration=False,
                  warning='Supervised CPU CNN; engineering comparison only, not proof of 98% quality.')
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    summary = (f"# MailGuard v50\n\n{report['warning']}\n\nNeural weight: {weight}; epoch: {best_epoch}\n\n"
               '| N | Recall | FP | FN |\n|---:|---:|---:|---:|\n'
               f"| {len(test)} | {result['recall']:.2%} | {result['fp']} | {result['fn']} |\n")
    (out / 'report.md').write_text(summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
