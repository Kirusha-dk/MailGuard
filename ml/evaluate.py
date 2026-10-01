#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mailguard_nn import (
    MailDataset,
    MailRow,
    collate,
    load_artifact,
    metrics,
    no_rspamd,
    scan_rspamd,
)

def files_under(path):
    return sorted(p for p in Path(path).rglob("*") if p.is_file() and not p.name.startswith("."))

def load_rows(spam_dir, ham_dir, rspamd_url):
    rows = []
    for folder, label, kind in [(spam_dir, 1, "test_spam"), (ham_dir, 0, "test_ham")]:
        paths = files_under(folder)
        print(f"{kind}: {len(paths)}", flush=True)
        for i, path in enumerate(paths, start=1):
            raw = path.read_bytes()
            rspamd = scan_rspamd(raw, rspamd_url) if rspamd_url else no_rspamd()
            rows.append(MailRow(raw, label, kind, rspamd))
            if i % 250 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)}", flush=True)
    return rows

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
    parser = argparse.ArgumentParser(description="Evaluate saved MailGuard neural model")
    parser.add_argument("--spam", required=True)
    parser.add_argument("--ham", required=True)
    parser.add_argument("--model", default="models/mailguard-v7.pt")
    parser.add_argument("--report", default="reports/evaluate-v7.json")
    parser.add_argument("--rspamd-url", default="http://rspamd:11333/")
    parser.add_argument("--no-rspamd", action="store_true")
    args = parser.parse_args()

    rspamd_url = None if args.no_rspamd else args.rspamd_url
    rows = load_rows(args.spam, args.ham, rspamd_url)
    payload, models = load_artifact(args.model)
    probabilities = predict(models, MailDataset(rows))
    threshold = float(payload["spam_threshold"])

    base = np.asarray([row.rspamd.is_spam for row in rows], dtype=bool)
    prediction = base | (probabilities >= threshold)
    result = {
        "modelVersion": payload.get("version"),
        "spamThreshold": threshold,
        **metrics(rows, prediction),
    }

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
