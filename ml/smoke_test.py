#!/usr/bin/env python3
import tempfile
from pathlib import Path

import numpy as np
import torch

from mailguard_nn import (
    MailNet,
    MailRow,
    MailDataset,
    no_rspamd,
    save_artifact,
    load_artifact,
    vectorize,
)

spam = b"""From: promo@bad.example
Subject: WIN MONEY NOW
Content-Type: text/plain; charset=utf-8

Click https://bad.example/win and claim free money now.
"""

row = MailRow(spam, 1, "smoke", no_rspamd())
tokens, numeric = vectorize(row)
assert len(tokens) > 0
assert numeric.shape == (9,)

model = MailNet()
dataset = MailDataset([row])
assert len(dataset) == 1

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "model.pt"
    save_artifact(path, [model], 0.9, {"smoke": True})
    payload, models = load_artifact(path)
    assert payload["spam_threshold"] == 0.9
    assert len(models) == 1

print("MailGuard neural smoke test: OK")
