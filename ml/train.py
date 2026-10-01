#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mailguard_nn import (
    MailDataset,
    MailNet,
    MailRow,
    class_weights,
    choose_threshold,
    collate,
    metrics,
    no_rspamd,
    save_artifact,
    scan_rspamd,
)

def files_under(path: str):
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))

def read_rows(spam_dir, ham_dir, rspamd_url, source):
    rows = []
    pairs = [(spam_dir, 1, "spam"), (ham_dir, 0, "ham")]
    for folder, label, kind in pairs:
        paths = files_under(folder)
        print(f"{source} {kind}: {len(paths)}", flush=True)
        for i, path in enumerate(paths, start=1):
            raw = path.read_bytes()
            rspamd = scan_rspamd(raw, rspamd_url) if rspamd_url else no_rspamd()
            rows.append(MailRow(raw, label, f"{source}_{kind}", rspamd))
            if i % 250 == 0 or i == len(paths):
                print(f"  {kind} {i}/{len(paths)}", flush=True)
    return rows

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def weighted_loss(logits, labels, weights):
    raw = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    return (raw * weights).sum() / torch.clamp(weights.sum(), min=1.0)

def train_model(train_ds, val_ds, seed, epochs, patience):
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
    best_val = float("inf")
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = []
        for ids, offsets, numeric, labels, weights in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, offsets, numeric)
            loss = weighted_loss(logits, labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss.append(float(loss.detach()))

        model.eval()
        val_loss = []
        with torch.no_grad():
            for ids, offsets, numeric, labels, weights in val_loader:
                logits = model(ids, offsets, numeric)
                val_loss.append(float(weighted_loss(logits, labels, weights)))

        current = float(np.mean(val_loss)) if val_loss else float("inf")
        print(
            f"seed={seed} epoch={epoch:02d} "
            f"train={np.mean(train_loss):.5f} val={current:.5f}",
            flush=True,
        )

        if current < best_val - 1e-4:
            best_val = current
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model

def predict(models, dataset):
    loader = DataLoader(dataset, batch_size=192, shuffle=False, collate_fn=collate)
    all_probs = []
    for model in models:
        current = []
        with torch.no_grad():
            for ids, offsets, numeric, labels, weights in loader:
                current.append(torch.sigmoid(model(ids, offsets, numeric)).numpy())
        all_probs.append(np.concatenate(current))
    return np.mean(all_probs, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Train MailGuard neural spam classifier")
    parser.add_argument("--train-spam", required=True)
    parser.add_argument("--train-ham", required=True)
    parser.add_argument("--val-spam", required=True)
    parser.add_argument("--val-ham", required=True)
    parser.add_argument("--model", default="models/mailguard-v7.pt")
    parser.add_argument("--report", default="reports/train-v7.json")
    parser.add_argument("--rspamd-url", default="http://rspamd:11333/")
    parser.add_argument("--no-rspamd", action="store_true")
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--max-added-fp", type=int)
    parser.add_argument("--max-added-fpr", type=float, default=0.0001)
    args = parser.parse_args()

    rspamd_url = None if args.no_rspamd else args.rspamd_url
    train_rows = read_rows(
        args.train_spam, args.train_ham, rspamd_url, "train"
    )
    val_rows = read_rows(
        args.val_spam, args.val_ham, rspamd_url, "validation"
    )

    residual_train = [row for row in train_rows if not row.rspamd.is_spam]
    residual_val = [row for row in val_rows if not row.rspamd.is_spam]

    if len({row.label for row in residual_train}) < 2:
        raise RuntimeError("Residual training data must contain both spam and ham.")
    if len({row.label for row in residual_val}) < 2:
        raise RuntimeError("Residual validation data must contain both spam and ham.")

    print(
        f"Residual train: {len(residual_train)} / {len(train_rows)}; "
        f"residual validation: {len(residual_val)} / {len(val_rows)}",
        flush=True,
    )

    train_ds = MailDataset(residual_train, class_weights(residual_train))
    val_residual_ds = MailDataset(residual_val, class_weights(residual_val))
    val_full_ds = MailDataset(val_rows)

    models = [
        train_model(
            train_ds,
            val_residual_ds,
            seed=20261001 + i * 97,
            epochs=args.epochs,
            patience=args.patience,
        )
        for i in range(args.ensemble)
    ]

    probabilities = predict(models, val_full_ds)
    ham_total = sum(row.label == 0 for row in val_rows)
    max_added_fp = (
        args.max_added_fp
        if args.max_added_fp is not None
        else int(np.floor(ham_total * args.max_added_fpr))
    )
    gate = choose_threshold(val_rows, probabilities, max_added_fp=max_added_fp)

    base = np.asarray([row.rspamd.is_spam for row in val_rows], dtype=bool)
    final = base | (probabilities >= gate["threshold"])
    validation = metrics(val_rows, final)

    metadata = {
        "trainCount": len(train_rows),
        "residualTrainCount": len(residual_train),
        "validationCount": len(val_rows),
        "maxAddedFalsePositives": max_added_fp,
        "validation": validation,
        "thresholdSelection": gate,
        "usesRspamd": rspamd_url is not None,
    }
    save_artifact(args.model, models, gate["threshold"], metadata)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps({
        "model": args.model,
        "spamThreshold": gate["threshold"],
        **validation,
    }, indent=2))

if __name__ == "__main__":
    main()
