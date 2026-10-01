#!/usr/bin/env python3
import hashlib
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_sample_weight

import benchmark_improved as b
import benchmark_v3 as v3

REPORTS = Path("reports")
SEED = 20261001
VOCAB_SIZE = 1 << 15
EMBED_DIM = 64
NUMERIC_DIM = 7
TOKEN_RE = re.compile(r"[\w@.\-]{2,}", re.UNICODE)
URL_RE = re.compile(r"https?://", re.IGNORECASE)

def stable_hash(token):
    digest = hashlib.blake2b(token.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % VOCAB_SIZE

def token_ids(row):
    text = b.extract_document(row["raw"]).lower()
    words = TOKEN_RE.findall(text)[:1400]

    ids = [stable_hash("w:" + w) for w in words]

    for i in range(min(len(words) - 1, 500)):
        ids.append(stable_hash("b:" + words[i] + "_" + words[i + 1]))

    compact = re.sub(r"\s+", " ", text)[:8000]
    added = 0
    for i in range(0, max(0, len(compact) - 3), 5):
        ids.append(stable_hash("c4:" + compact[i:i+4]))
        added += 1
        if added >= 1200:
            break

    action = re.sub(r"\W+", "_", row["action"].lower())
    ids.append(stable_hash("a:" + action))

    for name, score in row["symbols"][:120]:
        safe = re.sub(r"\W+", "_", name.lower())
        ids.append(stable_hash("s:" + safe))
        if score >= 2:
            ids.append(stable_hash("sp:" + safe))
        elif score <= -2:
            ids.append(stable_hash("sn:" + safe))

    return np.asarray(ids if ids else [0], dtype=np.int64)

def numeric_features(row):
    req = row["required"]
    ratio = row["rscore"] / req if req else 0.0
    text = b.extract_document(row["raw"])
    action = row["action"].lower()

    return np.asarray([
        max(-3.0, min(3.0, row["rscore"] / 15.0)),
        max(-3.0, min(3.0, ratio)),
        math.log1p(len(row["raw"])) / 12.0,
        math.log1p(len(URL_RE.findall(text))) / 4.0,
        1.0 if "<html" in text.lower() else 0.0,
        1.0 if action == "soft reject" else 0.0,
        1.0 if action == "no action" else 0.0,
    ], dtype=np.float32)

def sample_weights(rows):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    w = compute_sample_weight(class_weight="balanced", y=y).astype(np.float32)
    for i, r in enumerate(rows):
        if not r["y"] and "hard_ham" in r["source"]:
            w[i] *= 8.0
        if not r["y"] and r["rscore"] >= 4.0:
            w[i] *= 4.0
        if r["y"] and r["rscore"] <= 4.0:
            w[i] *= 2.0
    return w

class MailDataset(Dataset):
    def __init__(self, rows, weights=None):
        self.rows = rows
        self.tokens = [token_ids(r) for r in rows]
        self.numeric = [numeric_features(r) for r in rows]
        self.labels = np.asarray([r["y"] for r in rows], dtype=np.float32)
        self.weights = (
            np.ones(len(rows), dtype=np.float32)
            if weights is None else np.asarray(weights, dtype=np.float32)
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return (
            self.tokens[index],
            self.numeric[index],
            self.labels[index],
            self.weights[index],
        )

def collate(batch):
    token_arrays, numeric, labels, weights = zip(*batch)
    lengths = [len(x) for x in token_arrays]
    offsets = np.cumsum([0] + lengths[:-1], dtype=np.int64)
    flat = np.concatenate(token_arrays)

    return (
        torch.from_numpy(flat),
        torch.from_numpy(offsets),
        torch.from_numpy(np.stack(numeric)),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )

class MailNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.EmbeddingBag(
            VOCAB_SIZE,
            EMBED_DIM,
            mode="mean",
            include_last_offset=False,
        )
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM + NUMERIC_DIM, 96),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, token_ids, offsets, numeric):
        emb = self.embedding(token_ids, offsets)
        return self.net(torch.cat([emb, numeric], dim=1)).squeeze(1)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def weighted_loss(logits, labels, weights):
    raw = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    return (raw * weights).sum() / torch.clamp(weights.sum(), min=1.0)

def train_one(train_ds, val_ds, seed):
    set_seed(seed)
    model = MailNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-5)

    train_loader = DataLoader(
        train_ds, batch_size=96, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=192, shuffle=False, collate_fn=collate
    )

    best_state = None
    best_loss = float("inf")
    stale = 0

    for epoch in range(1, 19):
        model.train()
        total = 0.0
        batches = 0

        for ids, offsets, numeric, labels, weights in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, offsets, numeric)
            loss = weighted_loss(logits, labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1

        model.eval()
        val_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for ids, offsets, numeric, labels, weights in val_loader:
                logits = model(ids, offsets, numeric)
                loss = weighted_loss(logits, labels, weights)
                val_total += float(loss)
                val_batches += 1

        val_loss = val_total / max(1, val_batches)
        train_loss = total / max(1, batches)
        print(
            f"seed={seed} epoch={epoch:02d} train={train_loss:.5f} val={val_loss:.5f}",
            flush=True,
        )

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model

def predict(model, dataset):
    loader = DataLoader(dataset, batch_size=192, shuffle=False, collate_fn=collate)
    out = []
    with torch.no_grad():
        for ids, offsets, numeric, labels, weights in loader:
            logits = model(ids, offsets, numeric)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out) if out else np.asarray([], dtype=np.float64)

def ensemble_predict(models, dataset):
    return np.mean([predict(m, dataset) for m in models], axis=0)

def metrics(rows, pred):
    y = np.asarray([r["y"] for r in rows], dtype=bool)
    base = np.asarray([r["rspam"] for r in rows], dtype=bool)

    spam_total = int(y.sum())
    ham_total = int((~y).sum())
    tp = int((y & pred).sum())
    fp = int(((~y) & pred).sum())
    base_tp = int((y & base).sum())
    base_fp = int(((~y) & base).sum())

    return {
        "spamTotal": spam_total,
        "hamTotal": ham_total,
        "spamDetected": tp,
        "falsePositives": fp,
        "recall": tp / max(1, spam_total),
        "fpr": fp / max(1, ham_total),
        "baseSpamDetected": base_tp,
        "baseFalsePositives": base_fp,
        "rescuedSpam": tp - base_tp,
        "addedFalsePositives": fp - base_fp,
    }

def choose_threshold(rows, probabilities, max_added_fp):
    y = np.asarray([r["y"] for r in rows], dtype=bool)
    base = np.asarray([r["rspam"] for r in rows], dtype=bool)

    candidates = np.unique(
        np.concatenate([
            probabilities,
            np.asarray([0.0, .5, .7, .8, .9, .95, .97, .98, .99, .995, .999, 1.000001])
        ])
    )
    candidates.sort()
    candidates = candidates[::-1]

    best = None
    for threshold in candidates:
        pred = base | (probabilities >= threshold)
        m = metrics(rows, pred)
        if m["addedFalsePositives"] > max_added_fp:
            continue

        point = {**m, "threshold": float(threshold)}
        if (
            best is None
            or point["recall"] > best["recall"]
            or (
                point["recall"] == best["recall"]
                and point["addedFalsePositives"] < best["addedFalsePositives"]
            )
            or (
                point["recall"] == best["recall"]
                and point["addedFalsePositives"] == best["addedFalsePositives"]
                and point["threshold"] > best["threshold"]
            )
        ):
            best = point

    if best is None:
        pred = base.copy()
        best = {**metrics(rows, pred), "threshold": 1.000001}
    return best

def evaluate(rows, probabilities, gate):
    base = np.asarray([r["rspam"] for r in rows], dtype=bool)
    pred = base | (probabilities >= gate["threshold"])
    return metrics(rows, pred)

def review(rows, probabilities):
    residual = [i for i, r in enumerate(rows) if not r["rspam"]]
    budget = min(max(1, math.ceil(len(rows) * .01)), len(residual))

    top = sorted(residual, key=lambda i: probabilities[i], reverse=True)[:budget]
    top_spam = sum(rows[i]["y"] for i in top)

    rnd = random.Random(SEED)
    random_hits = []
    for _ in range(2000):
        sample = rnd.sample(residual, budget)
        random_hits.append(sum(rows[i]["y"] for i in sample))

    mean = float(np.mean(random_hits))
    return {
        "budget": budget,
        "topRiskSpamFound": int(top_spam),
        "randomSpamFoundMean": mean,
        "lift": top_spam / mean if mean else None,
    }

def main():
    REPORTS.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    b.wait_rspamd()

    groups = b.prepare()
    train, val, test = v3.build_splits(groups)

    print("TRAIN", len(train), "VAL", len(val), "TEST", len(test), flush=True)
    print("v7: neural residual classifier (PyTorch EmbeddingBag + MLP)", flush=True)

    b.reset_bayes()
    b.learn(train)

    tr = b.scan_many(train, "v7-train")
    va = b.scan_many(val, "v7-val")
    te = b.scan_many(test, "v7-test")

    base_test = b.base_metrics(te)

    residual_train = [r for r in tr if not r["rspam"]]
    residual_val = [r for r in va if not r["rspam"]]

    train_ds = MailDataset(residual_train, sample_weights(residual_train))
    val_residual_ds = MailDataset(residual_val, sample_weights(residual_val))
    val_ds = MailDataset(va)
    test_ds = MailDataset(te)

    models = [
        train_one(train_ds, val_residual_ds, SEED + i * 97)
        for i in range(3)
    ]

    pval = ensemble_predict(models, val_ds)
    ptest = ensemble_predict(models, test_ds)

    zero_gate = choose_threshold(va, pval, max_added_fp=0)
    one_gate = choose_threshold(va, pval, max_added_fp=1)
    three_gate = choose_threshold(va, pval, max_added_fp=3)

    zero_test = evaluate(te, ptest, zero_gate)
    one_test = evaluate(te, ptest, one_gate)
    three_test = evaluate(te, ptest, three_gate)

    result = {
        "version": "v7-neural-residual",
        "architecture": {
            "type": "PyTorch EmbeddingBag + MLP ensemble",
            "ensembleSize": 3,
            "vocabSize": VOCAB_SIZE,
            "embeddingDim": EMBED_DIM,
            "usesRspamdContext": True,
            "residualOnlyTraining": True,
        },
        "dataset": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "testSpam": base_test["spamTotal"],
            "testHam": base_test["hamTotal"],
        },
        "rspamdBayes": base_test,
        "zeroAddedFpValidationGate": zero_gate,
        "zeroAddedFpTest": zero_test,
        "oneAddedFpValidationGate": one_gate,
        "oneAddedFpTest": one_test,
        "threeAddedFpValidationGate": three_gate,
        "threeAddedFpTest": three_test,
        "review1pctResidual": review(te, ptest),
    }

    (REPORTS / "v7-benchmark.json").write_text(json.dumps(result, indent=2))

    md = [
        "# MailGuard v7 neural residual benchmark",
        "",
        f"Final untouched test: {base_test['spamTotal']} spam + {base_test['hamTotal']} ham.",
        "",
        "| Mode | Spam recall | Spam rescued over base | FP total | Added FP |",
        "|---|---:|---:|---:|---:|",
        f"| Rspamd + Bayes | {base_test['recall']:.2%} | 0 | "
        f"{base_test['falsePositives']}/{base_test['hamTotal']} | 0 |",
        f"| v7 neural, zero-added-FP gate | {zero_test['recall']:.2%} | "
        f"{zero_test['rescuedSpam']} | {zero_test['falsePositives']}/{zero_test['hamTotal']} | "
        f"{zero_test['addedFalsePositives']} |",
        f"| v7 neural, one-added-FP gate | {one_test['recall']:.2%} | "
        f"{one_test['rescuedSpam']} | {one_test['falsePositives']}/{one_test['hamTotal']} | "
        f"{one_test['addedFalsePositives']} |",
        f"| v7 neural, three-added-FP gate | {three_test['recall']:.2%} | "
        f"{three_test['rescuedSpam']} | {three_test['falsePositives']}/{three_test['hamTotal']} | "
        f"{three_test['addedFalsePositives']} |",
        "",
        "## 1% residual review",
        "",
        f"Random mean {result['review1pctResidual']['randomSpamFoundMean']:.2f}, "
        f"top-risk {result['review1pctResidual']['topRiskSpamFound']}, "
        f"lift {result['review1pctResidual']['lift']}.",
        "",
        "## Validation gates",
        "",
        f"Zero-added-FP: {zero_gate}",
        f"One-added-FP: {one_gate}",
        f"Three-added-FP: {three_gate}",
    ]
    text = "\n".join(md) + "\n"
    (REPORTS / "v7-benchmark.md").write_text(text)
    print(text, flush=True)

if __name__ == "__main__":
    main()
