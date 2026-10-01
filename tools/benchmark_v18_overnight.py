#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import benchmark_improved as b
import benchmark_v3 as v3
import benchmark_v15_large_enron as enron

REPORTS = Path("reports")
MODELS = Path("models/v18")
SEED = 20261001

TEST_PER_CLASS = 5000
VAL_PER_CLASS = 1500
WORD_VOCAB = 1 << 17
CHAR_VOCAB = 1 << 16
WORD_DIM = 128
CHAR_DIM = 96
NUMERIC_DIM = 12

WORD_RE = re.compile(r"(?u)[\w@.\-]{2,}")
URL_RE = re.compile(r"https?://", re.I)
DATE_RE = re.compile(br"^X-MailGuard-Original-Date:\s*(.*?)\r?$", re.I | re.M)
SYNTH_FROM_RE = re.compile(r"from enron-benchmark-\d+@example\.invalid", re.I)
SYNTH_TO_RE = re.compile(r"to mailguard-benchmark@example\.invalid", re.I)

PROTECTED_ACTIONS = {
    "reject", "add header", "rewrite subject", "quarantine", "discard"
}


@dataclass
class Packed:
    words: np.ndarray
    chars: np.ndarray
    numeric: np.ndarray
    label: float
    weight: float


def stable_hash(text: str, size: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % size


def original_date(path: Path):
    raw = path.read_bytes()[:12000]
    match = DATE_RE.search(raw)
    if not match:
        return datetime.min
    text = match.group(1).decode("utf-8", "ignore").strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S%z",
    ):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=None)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except Exception:
        return datetime.min


def split_enron(rows):
    spam = sorted(
        [r for r in rows if r["y"]],
        key=lambda r: (original_date(r["path"]), str(r["path"])),
    )
    ham = sorted(
        [r for r in rows if not r["y"]],
        key=lambda r: (original_date(r["path"]), str(r["path"])),
    )

    if len(spam) < TEST_PER_CLASS + VAL_PER_CLASS or len(ham) < TEST_PER_CLASS + VAL_PER_CLASS:
        raise RuntimeError("Enron corpus is too small for 10k chronological test")

    test = spam[-TEST_PER_CLASS:] + ham[-TEST_PER_CLASS:]
    val = (
        spam[-(TEST_PER_CLASS + VAL_PER_CLASS):-TEST_PER_CLASS]
        + ham[-(TEST_PER_CLASS + VAL_PER_CLASS):-TEST_PER_CLASS]
    )
    train = (
        spam[:-(TEST_PER_CLASS + VAL_PER_CLASS)]
        + ham[:-(TEST_PER_CLASS + VAL_PER_CLASS)]
    )

    rnd = random.Random(SEED)
    rnd.shuffle(train)
    rnd.shuffle(val)
    rnd.shuffle(test)
    return train, val, test


def is_protected(row):
    return str(row["action"]).lower() in PROTECTED_ACTIONS


def clean_document(row):
    text = b.extract_document(row["raw"])
    if b"X-MailGuard-Dataset: enron-spam" in row["raw"][:4000]:
        text = SYNTH_FROM_RE.sub("from __dataset_sender__", text)
        text = SYNTH_TO_RE.sub("to __dataset_recipient__", text)
    return text


def vectorize(row, weight):
    text = clean_document(row)
    low = text.lower()

    words = WORD_RE.findall(low)[:900]
    word_ids = [stable_hash("w:" + token, WORD_VOCAB) for token in words]

    for i in range(min(len(words) - 1, 320)):
        word_ids.append(
            stable_hash("b:" + words[i] + "_" + words[i + 1], WORD_VOCAB)
        )

    for name, score in row["symbols"][:120]:
        safe = re.sub(r"\W+", "_", name.lower())
        word_ids.append(stable_hash("sym:" + safe, WORD_VOCAB))
        if score >= 2:
            word_ids.append(stable_hash("sympos:" + safe, WORD_VOCAB))
        elif score <= -2:
            word_ids.append(stable_hash("symneg:" + safe, WORD_VOCAB))

    if not word_ids:
        word_ids = [0]

    compact = re.sub(r"\s+", " ", low)[:14000]
    char_ids = []
    step = 9
    for i in range(0, max(0, len(compact) - 5), step):
        char_ids.append(stable_hash("c6:" + compact[i:i + 6], CHAR_VOCAB))
        if len(char_ids) >= 600:
            break
    if not char_ids:
        char_ids = [0]

    required = float(row["required"] or 6.0)
    rscore = float(row["rscore"] or 0.0)
    ratio = rscore / required if required else 0.0
    pos_symbols = sum(1 for _, s in row["symbols"] if s > 0)
    neg_symbols = sum(1 for _, s in row["symbols"] if s < 0)
    strong_pos = sum(1 for _, s in row["symbols"] if s >= 2)
    strong_neg = sum(1 for _, s in row["symbols"] if s <= -2)

    numeric = np.asarray([
        max(-4.0, min(4.0, rscore / 12.0)),
        max(-4.0, min(4.0, ratio)),
        math.log1p(len(row["raw"])) / 14.0,
        math.log1p(len(words)) / 8.0,
        math.log1p(len(URL_RE.findall(text))) / 4.0,
        min(pos_symbols, 50) / 50.0,
        min(neg_symbols, 50) / 50.0,
        min(strong_pos, 20) / 20.0,
        min(strong_neg, 20) / 20.0,
        1.0 if "<html" in low or "text/html" in low else 0.0,
        1.0 if str(row["action"]).lower() == "no action" else 0.0,
        1.0 if str(row["action"]).lower() == "soft reject" else 0.0,
    ], dtype=np.float32)

    return Packed(
        np.asarray(word_ids, dtype=np.int32),
        np.asarray(char_ids, dtype=np.int32),
        numeric,
        float(row["y"]),
        float(weight),
    )


class MailDataset(Dataset):
    def __init__(self, rows, weights):
        started = time.time()
        self.items = []
        for i, (row, weight) in enumerate(zip(rows, weights), 1):
            self.items.append(vectorize(row, weight))
            if i % 1000 == 0 or i == len(rows):
                print(
                    f"vectorize {i}/{len(rows)} elapsed={time.time()-started:.1f}s",
                    flush=True,
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch):
    word_lens = [len(x.words) for x in batch]
    char_lens = [len(x.chars) for x in batch]

    word_offsets = np.cumsum([0] + word_lens[:-1], dtype=np.int64)
    char_offsets = np.cumsum([0] + char_lens[:-1], dtype=np.int64)

    words = np.concatenate([x.words for x in batch]).astype(np.int64, copy=False)
    chars = np.concatenate([x.chars for x in batch]).astype(np.int64, copy=False)
    numeric = np.stack([x.numeric for x in batch])

    return (
        torch.from_numpy(words),
        torch.from_numpy(word_offsets),
        torch.from_numpy(chars),
        torch.from_numpy(char_offsets),
        torch.from_numpy(numeric),
        torch.tensor([x.label for x in batch], dtype=torch.float32),
        torch.tensor([x.weight for x in batch], dtype=torch.float32),
    )


class DeepMailNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.word = nn.EmbeddingBag(
            WORD_VOCAB, WORD_DIM, mode="mean", include_last_offset=False
        )
        self.char = nn.EmbeddingBag(
            CHAR_VOCAB, CHAR_DIM, mode="mean", include_last_offset=False
        )
        self.numeric = nn.Sequential(
            nn.Linear(NUMERIC_DIM, 48),
            nn.GELU(),
            nn.LayerNorm(48),
        )
        self.net = nn.Sequential(
            nn.Linear(WORD_DIM + CHAR_DIM + 48, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.14),
            nn.Linear(128, 48),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(48, 1),
        )

    def forward(self, words, word_offsets, chars, char_offsets, numeric):
        a = self.word(words, word_offsets)
        c = self.char(chars, char_offsets)
        n = self.numeric(numeric)
        return self.net(torch.cat([a, c, n], dim=1)).squeeze(1)


def base_weights(rows, deep=False):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    spam = max(1, int(y.sum()))
    ham = max(1, len(y) - spam)
    out = np.where(
        y == 1,
        len(y) / (2.0 * spam),
        len(y) / (2.0 * ham),
    ).astype(np.float32)

    for i, row in enumerate(rows):
        rscore = float(row["rscore"])
        if row["y"] and rscore <= 3.0:
            out[i] *= 4.0 if deep else 2.5
        if row["y"] and rscore <= 1.5:
            out[i] *= 1.7
        if (not row["y"]) and rscore >= 3.0:
            out[i] *= 4.0
        if (not row["y"]) and rscore >= 5.0:
            out[i] *= 2.0
        if (not row["y"]) and any(name == "BAYES_HAM" for name, _ in row["symbols"]):
            out[i] *= 1.6
    return out


def weighted_focal_loss(logits, labels, weights, gamma=1.3):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    prob = torch.sigmoid(logits)
    pt = torch.where(labels > 0.5, prob, 1.0 - prob)
    focal = torch.pow(torch.clamp(1.0 - pt, min=1e-4), gamma)
    return (bce * focal * weights).sum() / torch.clamp(weights.sum(), min=1.0)


def predict_one(model, dataset, batch_size=256):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    out = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            words, wo, chars, co, numeric, labels, weights = batch
            logits = model(words, wo, chars, co, numeric)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def choose_threshold(rows, probs, max_added_fp):
    labels = np.asarray([bool(r["y"]) for r in rows], dtype=bool)
    base = np.asarray([is_protected(r) for r in rows], dtype=bool)
    base_fp = int(((~labels) & base).sum())

    candidates = np.unique(np.concatenate([
        np.asarray([0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.93,0.95,0.97,0.98,0.99,0.995,0.999,1.000001]),
        np.quantile(probs, np.linspace(.45, 1.0, 120)),
    ]))

    best = None
    for threshold in candidates:
        pred = base | (probs >= threshold)
        tp = int((labels & pred).sum())
        fp = int(((~labels) & pred).sum())
        if fp > base_fp + max_added_fp:
            continue
        point = {
            "threshold": float(threshold),
            "recall": tp / max(1, int(labels.sum())),
            "falsePositives": fp,
            "addedFalsePositives": fp - base_fp,
            "spamDetected": tp,
        }
        key = (
            point["recall"],
            -point["addedFalsePositives"],
            point["threshold"],
        )
        if best is None or key > best[0]:
            best = (key, point)

    if best is None:
        return {
            "threshold": 1.000001,
            "recall": int((labels & base).sum()) / max(1, int(labels.sum())),
            "falsePositives": base_fp,
            "addedFalsePositives": 0,
            "spamDetected": int((labels & base).sum()),
        }
    return best[1]


def train_model(
    name,
    train_ds,
    val_ds,
    val_rows,
    seed,
    max_epochs=32,
    patience=7,
    min_epochs=10,
    max_added_fp=0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = DeepMailNet()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-3, weight_decay=5e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=2e-4
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )

    best_state = None
    best_rank = None
    best_meta = None
    stale = 0

    MODELS.mkdir(parents=True, exist_ok=True)
    checkpoint = MODELS / f"{name}.pt"

    for epoch in range(1, max_epochs + 1):
        started = time.time()
        model.train()
        losses = []

        for batch in train_loader:
            words, wo, chars, co, numeric, labels, weights = batch
            optimizer.zero_grad(set_to_none=True)
            logits = model(words, wo, chars, co, numeric)
            loss = weighted_focal_loss(logits, labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        scheduler.step()

        val_probs = predict_one(model, val_ds)
        gate = choose_threshold(val_rows, val_probs, max_added_fp=max_added_fp)
        rank = (
            gate["recall"],
            -gate["addedFalsePositives"],
            -float(np.mean(losses)),
        )

        improved = best_rank is None or rank > best_rank
        if improved:
            best_rank = rank
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_meta = {
                "epoch": epoch,
                "gate": gate,
                "trainLoss": float(np.mean(losses)),
            }
            torch.save(
                {
                    "state_dict": best_state,
                    "meta": best_meta,
                    "name": name,
                },
                checkpoint,
            )
            stale = 0
        else:
            stale += 1

        print(
            f"{name} epoch={epoch:02d} loss={np.mean(losses):.5f} "
            f"val_recall={gate['recall']:.4%} added_fp={gate['addedFalsePositives']} "
            f"best={best_meta['gate']['recall']:.4%} "
            f"elapsed={time.time()-started:.1f}s",
            flush=True,
        )

        if epoch >= min_epochs and stale >= patience:
            print(f"{name} early stop at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_meta


def ensemble_predict(models, dataset):
    probs = [predict_one(model, dataset) for model in models]
    stacked = np.vstack(probs)
    return np.exp(np.mean(np.log(np.clip(stacked, 1e-7, 1.0)), axis=0)), stacked


def metrics(rows, pred):
    y = np.asarray([bool(r["y"]) for r in rows], dtype=bool)
    spam = int(y.sum())
    ham = int((~y).sum())
    tp = int((y & pred).sum())
    fp = int(((~y) & pred).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, spam)
    return {
        "spamTotal": spam,
        "hamTotal": ham,
        "spamDetected": tp,
        "falseNegatives": spam - tp,
        "falsePositives": fp,
        "recall": recall,
        "precision": precision,
        "fpr": fp / max(1, ham),
    }


def main():
    torch.set_num_threads(min(6, os.cpu_count() or 2))
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    b.wait_rspamd()

    print("[1/11] prepare 30k unique Enron corpus", flush=True)
    all_enron = enron.prepare_enron()
    enron_train, enron_val, enron_test = split_enron(all_enron)

    if len(enron_test) != 10000:
        raise RuntimeError(f"Expected 10000 test emails, got {len(enron_test)}")

    print(
        "ENRON chronological split",
        "train", len(enron_train),
        "val", len(enron_val),
        "test", len(enron_test),
        flush=True,
    )

    print("[2/11] add SpamAssassin training-only diversity", flush=True)
    groups = b.prepare()
    spam_train, _spam_val, _spam_test = v3.build_splits(groups)
    train_paths = list(enron_train) + list(spam_train)

    # Rspamd/Bayes learns from training data only.
    b.reset_bayes()
    b.learn(train_paths)

    print("[3/11] scan training mail", flush=True)
    tr = b.scan_many(train_paths, "v18-train", workers=16)
    print("[4/11] scan validation mail", flush=True)
    va = b.scan_many(enron_val, "v18-val", workers=16)
    print("[5/11] scan locked 10k test mail", flush=True)
    te = b.scan_many(enron_test, "v18-test", workers=16)

    residual_train = [row for row in tr if not is_protected(row)]
    if len({r["y"] for r in residual_train}) < 2:
        raise RuntimeError("Residual training set does not contain both classes")

    # General residual net.
    general_weights = base_weights(residual_train, deep=False)

    # Deep specialist only sees the region where Rspamd itself is weak.
    deep_train = [
        row for row in residual_train
        if float(row["rscore"]) <= 3.5
    ]
    if len({r["y"] for r in deep_train}) < 2:
        deep_train = residual_train
    deep_weights = base_weights(deep_train, deep=True)

    val_weights = np.ones(len(va), dtype=np.float32)
    test_weights = np.ones(len(te), dtype=np.float32)

    print(
        "[6/11] vectorize",
        "general", len(residual_train),
        "deep", len(deep_train),
        "val", len(va),
        "test", len(te),
        flush=True,
    )
    general_ds = MailDataset(residual_train, general_weights)
    deep_ds = MailDataset(deep_train, deep_weights)
    val_ds = MailDataset(va, val_weights)
    test_ds = MailDataset(te, test_weights)

    print("[7/11] train 3 general neural experts", flush=True)
    general_models = []
    general_meta = []
    for i in range(3):
        model, meta = train_model(
            f"general-{i+1}",
            general_ds,
            val_ds,
            va,
            SEED + i * 101,
            max_epochs=32,
            patience=7,
            min_epochs=10,
            max_added_fp=0,
        )
        general_models.append(model)
        general_meta.append(meta)

    print("[8/11] train 3 deep-error specialists", flush=True)
    deep_models = []
    deep_meta = []
    for i in range(3):
        model, meta = train_model(
            f"deep-{i+1}",
            deep_ds,
            val_ds,
            va,
            SEED + 5000 + i * 131,
            max_epochs=36,
            patience=8,
            min_epochs=12,
            max_added_fp=0,
        )
        deep_models.append(model)
        deep_meta.append(meta)

    print("[9/11] select ensemble gate on validation", flush=True)
    pg_val, raw_g_val = ensemble_predict(general_models, val_ds)
    pd_val, raw_d_val = ensemble_predict(deep_models, val_ds)

    # Deep branch gets stronger weight only for low-Rspamd messages.
    low_rspamd_val = np.asarray(
        [float(r["rscore"]) <= 3.5 for r in va], dtype=np.float64
    )
    blend_val = np.exp(
        0.64 * np.log(np.clip(pg_val, 1e-7, 1.0))
        + (0.20 + 0.16 * low_rspamd_val)
          * np.log(np.clip(pd_val, 1e-7, 1.0))
    )

    safe_gate = choose_threshold(va, blend_val, max_added_fp=0)
    balanced_gate = choose_threshold(va, blend_val, max_added_fp=1)

    print("[10/11] evaluate locked 10k test", flush=True)
    pg_test, raw_g_test = ensemble_predict(general_models, test_ds)
    pd_test, raw_d_test = ensemble_predict(deep_models, test_ds)
    low_rspamd_test = np.asarray(
        [float(r["rscore"]) <= 3.5 for r in te], dtype=np.float64
    )
    blend_test = np.exp(
        0.64 * np.log(np.clip(pg_test, 1e-7, 1.0))
        + (0.20 + 0.16 * low_rspamd_test)
          * np.log(np.clip(pd_test, 1e-7, 1.0))
    )

    base_pred = np.asarray([is_protected(r) for r in te], dtype=bool)
    safe_pred = base_pred | (blend_test >= safe_gate["threshold"])
    balanced_pred = base_pred | (blend_test >= balanced_gate["threshold"])

    base_stats = metrics(te, base_pred)
    safe_stats = metrics(te, safe_pred)
    balanced_stats = metrics(te, balanced_pred)

    # Deep-only slice diagnosis.
    labels = np.asarray([bool(r["y"]) for r in te], dtype=bool)
    deep_slice = np.asarray([float(r["rscore"]) <= 3.5 for r in te], dtype=bool)
    deep_spam = labels & deep_slice
    deep_recall_safe = (
        int((deep_spam & safe_pred).sum()) / max(1, int(deep_spam.sum()))
    )

    print("[11/11] save model + report", flush=True)
    package = {
        "version": "v18-overnight-deep-neural",
        "wordVocab": WORD_VOCAB,
        "charVocab": CHAR_VOCAB,
        "generalStates": [m.state_dict() for m in general_models],
        "deepStates": [m.state_dict() for m in deep_models],
        "safeThreshold": safe_gate["threshold"],
        "balancedThreshold": balanced_gate["threshold"],
        "blend": {
            "generalLogWeight": 0.64,
            "deepBaseLogWeight": 0.20,
            "deepLowRspamdBonus": 0.16,
            "lowRspamdCutoff": 3.5,
        },
    }
    torch.save(package, MODELS / "mailguard-v18-overnight.pt")

    result = {
        "version": "v18-overnight-deep-neural",
        "dataset": {
            "enronUnique": len(all_enron),
            "enronTrain": len(enron_train),
            "spamAssassinTrainOnly": len(spam_train),
            "validation": len(va),
            "lockedTest": len(te),
            "testSpam": int(labels.sum()),
            "testHam": int((~labels).sum()),
            "split": "class-stratified chronological Enron split",
        },
        "training": {
            "generalResidualCount": len(residual_train),
            "deepSpecialistCount": len(deep_train),
            "generalModels": general_meta,
            "deepModels": deep_meta,
            "checkpointing": True,
            "earlyStopping": True,
            "selectionMetric": "validation recall under added-FP budget",
        },
        "methodology": {
            "testLabelsUsedForTraining": False,
            "testLabelsUsedForThresholdSelection": False,
            "testSize": 10000,
            "rawProductionMailUploaded": False,
            "warning": (
                "Enron-Spam is old and has synthetic transport headers in this "
                "benchmark. This is a large controlled holdout, not a direct "
                "production estimate."
            ),
        },
        "rspamdBayes": base_stats,
        "safeGate": safe_gate,
        "safeTest": safe_stats,
        "balancedGate": balanced_gate,
        "balancedTest": balanced_stats,
        "deepSlice": {
            "cutoffRspamdScore": 3.5,
            "spamCount": int(deep_spam.sum()),
            "safeRecall": deep_recall_safe,
        },
    }

    (REPORTS / "v18-overnight.json").write_text(json.dumps(result, indent=2))

    md = [
        "# MailGuard v18 overnight deep-neural benchmark",
        "",
        f"Locked test: **{len(te)} unique emails** "
        f"({int(labels.sum())} spam + {int((~labels).sum())} ham).",
        "",
        "| Mode | Recall | FN | FP | FP rate | Precision |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Rspamd + Bayes | {base_stats['recall']:.2%} | "
        f"{base_stats['falseNegatives']} | {base_stats['falsePositives']} | "
        f"{base_stats['fpr']:.3%} | {base_stats['precision']:.3%} |",
        f"| v18 safe | {safe_stats['recall']:.2%} | "
        f"{safe_stats['falseNegatives']} | {safe_stats['falsePositives']} | "
        f"{safe_stats['fpr']:.3%} | {safe_stats['precision']:.3%} |",
        f"| v18 balanced | {balanced_stats['recall']:.2%} | "
        f"{balanced_stats['falseNegatives']} | {balanced_stats['falsePositives']} | "
        f"{balanced_stats['fpr']:.3%} | {balanced_stats['precision']:.3%} |",
        "",
        f"Low-Rspamd deep-spam slice: {int(deep_spam.sum())} messages; "
        f"v18 safe recall {deep_recall_safe:.2%}.",
        "",
        "Training uses 3 general neural experts + 3 deep-error specialists, "
        "per-epoch checkpoints, early stopping, and validation selection under "
        "a strict added-FP budget.",
        "",
        "The 10k test is held out from both training and threshold selection.",
    ]
    report = "\n".join(md) + "\n"
    (REPORTS / "v18-overnight.md").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
