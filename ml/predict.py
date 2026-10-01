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
    no_rspamd,
    scan_rspamd,
)

def neural_probability(models, row):
    dataset = MailDataset([row])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    ids, offsets, numeric, labels, weights = next(iter(loader))
    probabilities = []
    with torch.no_grad():
        for model in models:
            probabilities.append(float(torch.sigmoid(
                model(ids, offsets, numeric)
            )[0]))
    return float(np.mean(probabilities))

def main():
    parser = argparse.ArgumentParser(description="Classify one .eml with MailGuard neural model")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="models/mailguard-v7.pt")
    parser.add_argument("--rspamd-url", default="http://rspamd:11333/")
    parser.add_argument("--no-rspamd", action="store_true")
    args = parser.parse_args()

    raw = Path(args.input).read_bytes()
    rspamd = no_rspamd() if args.no_rspamd else scan_rspamd(raw, args.rspamd_url)

    payload, models = load_artifact(args.model)
    threshold = float(payload["spam_threshold"])
    row = MailRow(raw=raw, label=None, source="inference", rspamd=rspamd)
    probability = neural_probability(models, row)

    final_spam = rspamd.is_spam or probability >= threshold
    result = {
        "modelVersion": payload.get("version"),
        "neuralSpamProbability": probability,
        "spamThreshold": threshold,
        "rspamd": {
            "score": rspamd.score,
            "required": rspamd.required,
            "action": rspamd.action,
            "protectedSpam": rspamd.is_spam,
        },
        "decision": "SPAM" if final_spam else "UNSURE",
        "spam": final_spam,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
