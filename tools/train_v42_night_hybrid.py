#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from rspamd_benchmark_audit import load_frozen_v40

SEED = 20261021
REPORTS = Path("reports")
MODELS = Path("models/v42-night")
REPORT_JSON = REPORTS / "v42-night-experimental.json"
REPORT_MD = REPORTS / "v42-night-experimental.md"

MAX_BYTES = 4096
BATCH = 48
EPOCHS = 28
MIN_EPOCHS = 10
PATIENCE = 7


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RawMailDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        raw = row["path"].read_bytes()
        # Keep both beginning (headers/subject) and tail (often URLs/signatures).
        if len(raw) > MAX_BYTES:
            head = raw[:3072]
            tail = raw[-1024:]
            raw = head + tail
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64) + 1
        x = np.zeros(MAX_BYTES, dtype=np.int64)
        n = min(MAX_BYTES, len(arr))
        x[:n] = arr[:n]
        return torch.from_numpy(x), torch.tensor(float(row["y"]), dtype=torch.float32)


class ByteHybridNet(nn.Module):
    """
    Intentionally heavyweight experimental mail classifier:
    raw bytes -> multiscale CNN -> Transformer -> BiGRU -> attention pooling.

    This is slower than the normal MailGuard experts and is intentionally kept
    separate from production logic until it proves useful.
    """
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(257, 80, padding_idx=0)

        self.conv3 = nn.Conv1d(80, 96, kernel_size=3, stride=4, padding=1)
        self.conv5 = nn.Conv1d(80, 96, kernel_size=5, stride=4, padding=2)
        self.conv9 = nn.Conv1d(80, 96, kernel_size=9, stride=4, padding=4)
        self.conv_norm = nn.LayerNorm(288)

        self.down = nn.Sequential(
            nn.Conv1d(288, 192, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
        )
        self.pos = nn.Parameter(torch.randn(1, 256, 192) * 0.015)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=192,
            nhead=6,
            dim_feedforward=640,
            dropout=0.14,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.gru = nn.GRU(
            input_size=192,
            hidden_size=128,
            num_layers=2,
            dropout=0.12,
            bidirectional=True,
            batch_first=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.22),
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(96, 1),
        )

    def forward(self, x):
        mask0 = x != 0
        z = self.emb(x).transpose(1, 2)

        a = torch.nn.functional.gelu(self.conv3(z))
        b = torch.nn.functional.gelu(self.conv5(z))
        c = torch.nn.functional.gelu(self.conv9(z))
        z = torch.cat([a, b, c], dim=1).transpose(1, 2)
        z = self.conv_norm(z).transpose(1, 2)
        z = self.down(z).transpose(1, 2)

        seq = z.size(1)
        z = z + self.pos[:, :seq]
        mask = torch.nn.functional.max_pool1d(
            mask0.float().unsqueeze(1), kernel_size=16, stride=16, ceil_mode=True
        ).squeeze(1)
        mask = mask[:, :seq] < 0.5

        z = self.transformer(z, src_key_padding_mask=mask)
        z, _ = self.gru(z)

        logits = self.attn(z).squeeze(-1)
        logits = logits.masked_fill(mask, -1e4)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(z * weights.unsqueeze(-1), dim=1)
        return self.head(pooled).squeeze(-1)


def source_weights(rows):
    counts = Counter((r["source"], int(r["y"])) for r in rows)
    med = float(np.median(list(counts.values())))
    out = []
    for r in rows:
        n = max(1, counts[(r["source"], int(r["y"]))])
        corr = math.sqrt(med / n)
        corr = min(1.8, max(0.70, corr))
        # Slightly favour ham because FP is the expensive error for MailGuard.
        if not r["y"]:
            corr *= 1.25
        out.append(corr)
    return np.asarray(out, dtype=np.float32)


def score_model(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.sigmoid(model(x)).cpu().numpy()
            probs.append(p)
            labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def low_fp_metrics(y, p, fp_budget):
    y = y.astype(bool)
    candidates = np.unique(np.concatenate([
        np.linspace(0.50, 0.995, 120),
        np.quantile(p, np.linspace(0.50, 0.9995, 240)),
        np.asarray([1.000001]),
    ]))
    best = None
    for t in candidates:
        pred = p >= t
        tp = int((y & pred).sum())
        fp = int(((~y) & pred).sum())
        if fp > fp_budget:
            continue
        recall = tp / max(1, int(y.sum()))
        precision = tp / max(1, tp + fp)
        point = dict(
            threshold=float(t), tp=tp, fp=fp,
            fn=int(y.sum()) - tp,
            recall=recall, precision=precision,
            fpr=fp / max(1, int((~y).sum())),
        )
        key = (tp, -fp, t)
        if best is None or key > best[0]:
            best = (key, point)
    return best[1] if best else dict(
        threshold=1.000001, tp=0, fp=0, fn=int(y.sum()),
        recall=0.0, precision=0.0, fpr=0.0,
    )


def main():
    seed_all(SEED)
    torch.set_num_threads(min(6, os.cpu_count() or 2))
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    data, audit = load_frozen_v40(".cache/v40-fresh50k")
    train = data["train"]
    val = data["val"]
    test = data["test"]

    print("v42-night dataset", len(train), len(val), len(test), flush=True)
    print("fingerprints", json.dumps(audit), flush=True)

    train_ds = RawMailDataset(train)
    val_ds = RawMailDataset(val)
    test_ds = RawMailDataset(test)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=96, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=96, shuffle=False, num_workers=0)

    weights = source_weights(train)
    # Loader is shuffled, so use label-aware loss rather than per-row weights here.
    ham = sum(not r["y"] for r in train)
    spam = sum(r["y"] for r in train)
    pos_weight = torch.tensor([ham / max(1, spam)], dtype=torch.float32)

    model = ByteHybridNet()
    params = sum(p.numel() for p in model.parameters())
    print("v42-night parameters", params, flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=4e-5
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_rank = None
    best_meta = None
    stale = 0

    for epoch in range(1, EPOCHS + 1):
        started = time.time()
        model.train()
        losses = []

        for step, (x, y) in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 250 == 0:
                print(
                    f"epoch {epoch:02d} step {step}/{len(train_loader)} "
                    f"loss={np.mean(losses[-100:]):.5f}",
                    flush=True,
                )

        scheduler.step()
        vy, vp = score_model(model, val_loader)
        ap = float(average_precision_score(vy, vp))
        auc = float(roc_auc_score(vy, vp))
        # Validation: 5,300 ham. Rank at <=5 FP, then AP.
        low = low_fp_metrics(vy, vp, fp_budget=5)
        rank = (low["recall"], ap, auc, -float(np.mean(losses)))

        meta = dict(
            epoch=epoch,
            loss=float(np.mean(losses)),
            valAP=ap,
            valAUC=auc,
            valLowFp=low,
            elapsedSeconds=time.time() - started,
        )

        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_state = copy.deepcopy(model.state_dict())
            best_meta = meta
            torch.save({
                "architecture": "ByteHybridNet-v42-night",
                "state_dict": best_state,
                "meta": best_meta,
                "maxBytes": MAX_BYTES,
                "parameters": params,
                "datasetFingerprints": audit,
            }, MODELS / "byte-hybrid-best.pt")
            stale = 0
        else:
            stale += 1

        print(
            f"v42-night epoch={epoch:02d}/{EPOCHS} loss={meta['loss']:.5f} "
            f"val_ap={ap:.5f} val_auc={auc:.5f} "
            f"lowfp_recall={low['recall']:.2%} fp={low['fp']} "
            f"best={best_meta['epoch']} stale={stale} "
            f"elapsed={meta['elapsedSeconds']:.1f}s",
            flush=True,
        )

        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            print("v42-night early stop", epoch, flush=True)
            break

    if best_state is None:
        raise RuntimeError("No v42 checkpoint")
    model.load_state_dict(best_state)

    ty, tp = score_model(model, test_loader)
    # Engineering-only evaluation: same frozen v40 test that has already been inspected.
    results = {
        "safe": low_fp_metrics(ty, tp, fp_budget=5),
        "balanced": low_fp_metrics(ty, tp, fp_budget=25),
        "recall": low_fp_metrics(ty, tp, fp_budget=100),
    }

    report = {
        "version": "v42-night-experimental-byte-hybrid",
        "architecture": (
            "raw-byte embedding + multiscale CNN + 4-layer TransformerEncoder + "
            "2-layer bidirectional GRU + learned attention pooling"
        ),
        "parameters": params,
        "maxBytes": MAX_BYTES,
        "dataset": {
            "train": len(train), "val": len(val), "test": len(test),
            "fingerprints": audit,
            "warning": (
                "Same frozen v40 test; engineering comparison only, not a new lockbox."
            ),
        },
        "bestValidation": best_meta,
        "test": results,
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2))

    lines = [
        "# MailGuard v42 night experimental neural",
        "",
        report["architecture"],
        "",
        f"Parameters: **{params:,}**",
        "",
        "Same frozen v40 50k test: engineering comparison only.",
        "",
        "| Mode | Recall | FN | FP | FP rate | Precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in results.items():
        lines.append(
            f"| {name} | {s['recall']:.2%} | {s['fn']} | {s['fp']} | "
            f"{s['fpr']:.3%} | {s['precision']:.3%} |"
        )
    lines += [
        "",
        f"Best validation epoch: {best_meta['epoch']}",
        f"Validation AP: {best_meta['valAP']:.5f}",
        f"Validation AUC: {best_meta['valAUC']:.5f}",
        f"Validation <=5 FP recall: {best_meta['valLowFp']['recall']:.2%}",
    ]
    text = "\n".join(lines) + "\n"
    REPORT_MD.write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
