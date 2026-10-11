#!/usr/bin/env python3
"""Supervised word/bigram network using head, middle, and tail context."""
import json
import re
import zlib
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn
import train_v50_neural as trainer

VOCAB = 65536
MAX_WORDS = 768
TOKENS = MAX_WORDS * 2
TOKEN_RE = re.compile(r"(?u)\b\w[\w.-]*\b")


def word_encode(texts):
    result = np.zeros((len(texts), TOKENS), dtype=np.int32)
    for i, text in enumerate(texts):
        words = TOKEN_RE.findall(text.casefold())
        if len(words) > MAX_WORDS:
            # Sample windows rather than joining distant words into fake bigrams.
            width = MAX_WORDS // 3
            mid = max(width, len(words) // 2 - width // 2)
            windows = [words[:width], words[mid:mid + width], words[-width:]]
        else:
            windows = [words]
        tokens = []
        for window in windows:
            tokens.extend('w:' + w for w in window)
            tokens.extend('b:' + a + '\x1f' + b for a, b in zip(window, window[1:]))
        ids = [1 + zlib.crc32(t.encode('utf-8')) % (VOCAB - 1) for t in tokens[:TOKENS]]
        result[i, :len(ids)] = ids
    return result


class WordNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, 48, padding_idx=0)
        self.dropout = nn.Dropout(.15)
        self.classifier = nn.Sequential(nn.Linear(96, 96), nn.ReLU(), nn.Dropout(.3), nn.Linear(96, 1))

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        embedded = self.dropout(self.embedding(x.long()))
        mean = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        maximum = embedded.masked_fill(~mask, -1e4).amax(dim=1)
        maximum = torch.where(mask.any(dim=1), maximum, torch.zeros_like(maximum))
        return self.classifier(torch.cat([mean, maximum], dim=1)).squeeze(1)


def predict_words(model, texts, batch_size=128):
    model.eval()
    x = word_encode(texts)
    result = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tensor = torch.from_numpy(x[start:start + batch_size])
            result.extend(torch.sigmoid(model(tensor)).numpy().tolist())
    return np.asarray(result)


def main():
    # Reuse the audited TRAIN/selection/calibration workflow, changing only
    # the neural representation and predictor. No test-label feedback.
    trainer.ByteCNN = WordNet
    trainer.encode = word_encode
    trainer.predict_neural = predict_words
    trainer.main()
    import sys
    args = sys.argv
    out = Path(args[args.index('--output') + 1] if '--output' in args else 'reports/v50-smoke')
    checkpoint = torch.load(out / 'neural.pt', map_location='cpu', weights_only=True)
    checkpoint.update(architecture='word-hash-v52', vocab=VOCAB, maxWords=MAX_WORDS)
    torch.save(checkpoint, out / 'neural.pt')
    artifact = joblib.load(out / 'model.joblib')
    artifact.update(version='v52-wordnet-blend', neuralArchitecture='word-hash-v52')
    joblib.dump(artifact, out / 'model.joblib', compress=3)
    report = json.loads((out / 'report.json').read_text())
    report.update(version=52, representation='hashed words+bigrams; head/middle/tail; mean+max pooling',
                  warning='Experimental learned lexical model; reused engineering data, not independent 98% evidence.')
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    summary = (out / 'report.md').read_text().replace('MailGuard v50', 'MailGuard v52')
    (out / 'report.md').write_text(summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
