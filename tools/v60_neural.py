"""Small Unicode CNN: learned multilingual character patterns, offline CPU inference."""
import unicodedata
import numpy as np
import torch
from torch import nn

WINDOW = 384
VOCAB = 8192


def encode(rows):
    result = np.zeros((len(rows), WINDOW * 2), dtype=np.int16)
    for i, row in enumerate(rows):
        text = unicodedata.normalize('NFKC', row['focus'] + ' ' + row['context']).casefold()
        for offset, window in ((0, text[:WINDOW]), (WINDOW, text[-WINDOW:])):
            # Stable hash, independent of Python's randomized hash and labels.
            result[i, offset:offset+len(window)] = [1 + (ord(c) * 2654435761 % (VOCAB - 1)) for c in window]
    return result


class UnicodeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, 16, padding_idx=0)
        self.branches = nn.ModuleList(nn.Conv1d(16, 32, kernel) for kernel in (3, 5, 7))
        self.classifier = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, 1))

    def forward(self, x):
        windows = (x[:, :WINDOW], x[:, WINDOW:])
        vectors = []
        for window in windows:
            embedded = self.embedding(window.long()).transpose(1, 2)
            vectors.extend(torch.relu(conv(embedded)).amax(2) for conv in self.branches)
        return self.classifier(torch.cat(vectors, 1)).squeeze(1)


def predict(model, rows, batch_size=128):
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            x = torch.from_numpy(encode(rows[start:start+batch_size]))
            values.append(torch.sigmoid(model(x)).numpy().astype(np.float64))
    return np.concatenate(values) if values else np.empty(0)
