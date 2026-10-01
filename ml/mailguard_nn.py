#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from html import unescape
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

VERSION = "v7.1-neural"
VOCAB_SIZE = 1 << 16
EMBED_DIM = 96
NUMERIC_DIM = 9

TOKEN_RE = re.compile(r"[\w@.\-]{2,}", re.UNICODE)
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
PROTECTED_ACTIONS = {
    "reject",
    "add header",
    "rewrite subject",
    "quarantine",
    "discard",
}

@dataclass
class RspamdResult:
    score: float = 0.0
    required: float = 6.0
    action: str = "no action"
    symbols: tuple[tuple[str, float], ...] = ()

    @property
    def is_spam(self) -> bool:
        return self.action.strip().lower() in PROTECTED_ACTIONS

@dataclass
class MailRow:
    raw: bytes
    label: int | None
    source: str
    rspamd: RspamdResult

def _decode_part(part) -> str:
    try:
        content = part.get_content()
        if isinstance(content, str):
            return content
        if isinstance(content, bytes):
            charset = part.get_content_charset() or "utf-8"
            return content.decode(charset, "replace")
    except Exception:
        pass
    payload = part.get_payload(decode=True)
    if isinstance(payload, bytes):
        return payload.decode(part.get_content_charset() or "utf-8", "replace")
    return ""

def extract_document(raw: bytes) -> tuple[str, dict[str, float | int | str]]:
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw)
    except Exception:
        text = raw.decode("utf-8", "replace")
        return text, {
            "url_count": len(URL_RE.findall(text)),
            "attachment_count": 0,
            "has_html": 0,
            "sender_domain": "",
        }

    subject = str(msg.get("subject", "") or "")
    sender = str(msg.get("from", "") or "")
    reply_to = str(msg.get("reply-to", "") or "")
    return_path = str(msg.get("return-path", "") or "")

    plain = []
    html = []
    attachments = 0

    parts = msg.walk() if msg.is_multipart() else [msg]
    for part in parts:
        if part.is_multipart():
            continue
        disposition = (part.get_content_disposition() or "").lower()
        ctype = (part.get_content_type() or "").lower()

        if disposition == "attachment":
            attachments += 1
            continue

        if ctype == "text/plain":
            plain.append(_decode_part(part))
        elif ctype == "text/html":
            html.append(_decode_part(part))

    html_text = "\n".join(html)
    html_visible = unescape(TAG_RE.sub(" ", html_text))
    body = "\n".join(plain)
    if not body.strip():
        body = html_visible
    elif html_visible.strip():
        body += "\n" + html_visible[:20000]

    sender_domain = ""
    match = re.search(r"@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", sender)
    if match:
        sender_domain = match.group(1).lower()

    header_context = "\n".join([
        "SUBJECT " + subject,
        "FROM " + sender,
        "REPLY_TO " + reply_to,
        "RETURN_PATH " + return_path,
    ])
    document = SPACE_RE.sub(" ", header_context + "\n" + body).strip()

    return document, {
        "url_count": len(URL_RE.findall(document)),
        "attachment_count": attachments,
        "has_html": 1 if html else 0,
        "sender_domain": sender_domain,
    }

def _endpoint(base: str) -> str:
    base = base.rstrip("/")
    return base if base.endswith("/checkv2") else base + "/checkv2"

def scan_rspamd(raw: bytes, base_url: str, timeout: float = 15.0) -> RspamdResult:
    request = urllib.request.Request(
        _endpoint(base_url),
        data=raw,
        method="POST",
        headers={"Content-Type": "message/rfc822"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8", "replace"))

    score = float(payload.get("score", 0.0) or 0.0)
    required = float(payload.get("required_score", 6.0) or 6.0)
    action = str(payload.get("action", "no action") or "no action")

    parsed_symbols = []
    symbols = payload.get("symbols", {})
    if isinstance(symbols, dict):
        for name, item in symbols.items():
            if isinstance(item, dict):
                value = float(item.get("score", 0.0) or 0.0)
            else:
                try:
                    value = float(item)
                except Exception:
                    value = 0.0
            parsed_symbols.append((str(name), value))
    elif isinstance(symbols, list):
        for item in symbols:
            if isinstance(item, dict):
                name = str(item.get("name", ""))
                value = float(item.get("score", 0.0) or 0.0)
                if name:
                    parsed_symbols.append((name, value))

    parsed_symbols.sort(key=lambda x: abs(x[1]), reverse=True)
    return RspamdResult(score, required, action, tuple(parsed_symbols[:160]))

def no_rspamd() -> RspamdResult:
    return RspamdResult()

def stable_hash(token: str, vocab_size: int = VOCAB_SIZE) -> int:
    digest = hashlib.blake2b(token.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % vocab_size

def vectorize(row: MailRow, vocab_size: int = VOCAB_SIZE) -> tuple[np.ndarray, np.ndarray]:
    text, meta = extract_document(row.raw)
    low = text.lower()
    words = TOKEN_RE.findall(low)[:1800]

    ids = [stable_hash("w:" + word, vocab_size) for word in words]

    for i in range(min(len(words) - 1, 700)):
        ids.append(stable_hash("b:" + words[i] + "_" + words[i + 1], vocab_size))

    compact = SPACE_RE.sub(" ", low)[:12000]
    for n, i in enumerate(range(0, max(0, len(compact) - 4), 7)):
        if n >= 1400:
            break
        ids.append(stable_hash("c5:" + compact[i:i + 5], vocab_size))

    sender_domain = str(meta["sender_domain"])
    if sender_domain:
        ids.append(stable_hash("from_domain:" + sender_domain, vocab_size))

    action = re.sub(r"\W+", "_", row.rspamd.action.lower())
    ids.append(stable_hash("action:" + action, vocab_size))

    for name, score in row.rspamd.symbols:
        safe = re.sub(r"\W+", "_", name.lower())
        ids.append(stable_hash("sym:" + safe, vocab_size))
        if score >= 2:
            ids.append(stable_hash("sym_pos:" + safe, vocab_size))
        elif score <= -2:
            ids.append(stable_hash("sym_neg:" + safe, vocab_size))

    if not ids:
        ids = [0]

    required = row.rspamd.required if row.rspamd.required else 6.0
    ratio = row.rspamd.score / required if required else 0.0
    numeric = np.asarray([
        max(-4.0, min(4.0, row.rspamd.score / 15.0)),
        max(-4.0, min(4.0, ratio)),
        math.log1p(len(row.raw)) / 14.0,
        math.log1p(int(meta["url_count"])) / 4.0,
        math.log1p(int(meta["attachment_count"])) / 3.0,
        float(meta["has_html"]),
        1.0 if row.rspamd.action.lower() == "soft reject" else 0.0,
        1.0 if row.rspamd.action.lower() == "no action" else 0.0,
        min(1.0, len(row.rspamd.symbols) / 40.0),
    ], dtype=np.float32)

    return np.asarray(ids, dtype=np.int64), numeric

class MailDataset(Dataset):
    def __init__(self, rows: list[MailRow], weights: np.ndarray | None = None):
        self.rows = rows
        self.features = [vectorize(row) for row in rows]
        self.labels = np.asarray([
            0.0 if row.label is None else float(row.label) for row in rows
        ], dtype=np.float32)
        self.weights = (
            np.ones(len(rows), dtype=np.float32)
            if weights is None
            else np.asarray(weights, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        tokens, numeric = self.features[index]
        return tokens, numeric, self.labels[index], self.weights[index]

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
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        numeric_dim: int = NUMERIC_DIM,
    ):
        super().__init__()
        self.embedding = nn.EmbeddingBag(
            vocab_size,
            embed_dim,
            mode="mean",
            include_last_offset=False,
        )
        self.network = nn.Sequential(
            nn.Linear(embed_dim + numeric_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.18),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(48, 1),
        )

    def forward(self, token_ids, offsets, numeric):
        embedded = self.embedding(token_ids, offsets)
        merged = torch.cat([embedded, numeric], dim=1)
        return self.network(merged).squeeze(1)

def class_weights(rows: list[MailRow]) -> np.ndarray:
    labels = np.asarray([int(row.label or 0) for row in rows], dtype=np.int32)
    positives = max(1, int(labels.sum()))
    negatives = max(1, len(labels) - positives)
    total = max(1, len(labels))

    spam_weight = total / (2.0 * positives)
    ham_weight = total / (2.0 * negatives)
    weights = np.where(labels == 1, spam_weight, ham_weight).astype(np.float32)

    for i, row in enumerate(rows):
        source = row.source.lower()
        if row.label == 0 and ("hard" in source or row.rspamd.score >= 4.0):
            weights[i] *= 5.0
        if row.label == 1 and row.rspamd.score <= 4.0:
            weights[i] *= 2.0
    return weights

def metrics(rows: list[MailRow], prediction: np.ndarray) -> dict:
    labels = np.asarray([bool(row.label) for row in rows], dtype=bool)
    base = np.asarray([row.rspamd.is_spam for row in rows], dtype=bool)

    spam_total = int(labels.sum())
    ham_total = int((~labels).sum())
    spam_detected = int((labels & prediction).sum())
    false_positives = int(((~labels) & prediction).sum())
    base_spam = int((labels & base).sum())
    base_fp = int(((~labels) & base).sum())

    return {
        "spamTotal": spam_total,
        "hamTotal": ham_total,
        "spamDetected": spam_detected,
        "falsePositives": false_positives,
        "recall": spam_detected / max(1, spam_total),
        "falsePositiveRate": false_positives / max(1, ham_total),
        "baseSpamDetected": base_spam,
        "baseFalsePositives": base_fp,
        "rescuedSpam": spam_detected - base_spam,
        "addedFalsePositives": false_positives - base_fp,
    }

def choose_threshold(
    rows: list[MailRow],
    probabilities: np.ndarray,
    max_added_fp: int,
) -> dict:
    base = np.asarray([row.rspamd.is_spam for row in rows], dtype=bool)
    candidates = np.unique(np.concatenate([
        probabilities,
        np.asarray([0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 1.000001]),
    ]))
    candidates = np.sort(candidates)[::-1]

    best = None
    for threshold in candidates:
        prediction = base | (probabilities >= threshold)
        current = metrics(rows, prediction)
        if current["addedFalsePositives"] > max_added_fp:
            continue

        point = {**current, "threshold": float(threshold)}
        key = (
            point["recall"],
            -point["addedFalsePositives"],
            point["threshold"],
        )
        if best is None or key > (
            best["recall"],
            -best["addedFalsePositives"],
            best["threshold"],
        ):
            best = point

    if best is None:
        best = {**metrics(rows, base.copy()), "threshold": 1.000001}
    return best

def save_artifact(
    path: str | Path,
    models: Iterable[MailNet],
    spam_threshold: float,
    metadata: dict,
) -> None:
    payload = {
        "version": VERSION,
        "vocab_size": VOCAB_SIZE,
        "embed_dim": EMBED_DIM,
        "numeric_dim": NUMERIC_DIM,
        "spam_threshold": float(spam_threshold),
        "metadata": metadata,
        "state_dicts": [model.state_dict() for model in models],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)

def load_artifact(path: str | Path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    models = []
    for state in payload["state_dicts"]:
        model = MailNet(
            int(payload.get("vocab_size", VOCAB_SIZE)),
            int(payload.get("embed_dim", EMBED_DIM)),
            int(payload.get("numeric_dim", NUMERIC_DIM)),
        )
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    return payload, models
