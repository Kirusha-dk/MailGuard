#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import benchmark_v18_overnight as v18
import benchmark_v25_soft_veto as v25

CACHE = Path(".cache/trec07-fresh50k")
EML_DIR = CACHE / "eml"
MANIFEST = CACHE / "manifest.json"
TARGET_HAM = 22000
TARGET_SPAM = 28000
TARGET_TOTAL = TARGET_HAM + TARGET_SPAM
SEED = 20261007


def clean_header(value):
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"[\r\n\x00]+", " ", text)
    return " ".join(text.split())[:1000]


def render_email(item, idx):
    subject = clean_header(item.get("subject"))
    sender = clean_header(item.get("sender")) or "trec07-unknown-sender@example.invalid"
    receiver = clean_header(item.get("receiver")) or "mailguard-benchmark@example.invalid"
    date = clean_header(item.get("date"))
    body = str(item.get("text") or "")

    headers = [
        f"Subject: {subject}",
        f"From: {sender}",
        f"To: {receiver}",
        "X-MailGuard-Dataset: trec07-fresh50k",
        f"X-MailGuard-Original-Date: {date}",
        f"X-MailGuard-Trec-Row: {idx}",
        "MIME-Version: 1.0",
        "Content-Type: text/plain; charset=utf-8",
        "Content-Transfer-Encoding: 8bit",
        "",
        "",
    ]
    return ("\r\n".join(headers)).encode("utf-8", "replace") + body.encode("utf-8", "replace")


def normalized_key(item):
    label = int(item.get("label") or 0)
    subject = clean_header(item.get("subject")).lower()
    sender = clean_header(item.get("sender")).lower()
    receiver = clean_header(item.get("receiver")).lower()
    body = " ".join(str(item.get("text") or "").lower().split())
    payload = "\n".join([str(label), sender, receiver, subject, body]).encode("utf-8", "ignore")
    return hashlib.sha256(payload).hexdigest()


def load_cached():
    if not MANIFEST.exists() or not EML_DIR.exists():
        return None
    try:
        items = json.loads(MANIFEST.read_text())
    except Exception:
        return None
    rows = [
        {
            "path": EML_DIR / x["file"],
            "y": int(x["y"]),
            "source": x["source"],
        }
        for x in items
    ]
    if len(rows) != TARGET_TOTAL:
        return None
    if not all(r["path"].exists() for r in rows[:100]):
        return None
    print(
        "TREC-07 fresh cache",
        len(rows),
        "spam", sum(r["y"] for r in rows),
        "ham", sum(not r["y"] for r in rows),
        flush=True,
    )
    return rows


def prepare_trec50k():
    cached = load_cached()
    if cached is not None:
        return cached

    from datasets import load_dataset

    print("Downloading fresh TREC-07 source from Hugging Face", flush=True)
    ds = load_dataset("JinqiangDing/seven-phishing-email-datasets")

    candidates = {0: [], 1: []}
    seen = set()
    global_idx = 0

    for split_name, split in ds.items():
        print("scan HF split", split_name, len(split), flush=True)
        for item in split:
            global_idx += 1
            if str(item.get("dataset_name") or "").strip().upper() != "TREC-07":
                continue
            y = int(item.get("label") or 0)
            if y not in (0, 1):
                continue
            key = normalized_key(item)
            if key in seen:
                continue
            seen.add(key)
            candidates[y].append((key, global_idx, item))

    print(
        "TREC-07 unique candidates",
        "spam", len(candidates[1]),
        "ham", len(candidates[0]),
        flush=True,
    )

    if len(candidates[0]) < TARGET_HAM or len(candidates[1]) < TARGET_SPAM:
        raise RuntimeError(
            f"Not enough unique TREC-07 mail for 50k lockbox: "
            f"ham={len(candidates[0])}, spam={len(candidates[1])}"
        )

    # Deterministic hash ordering prevents hand-picking easy/hard test messages.
    candidates[0].sort(key=lambda x: x[0])
    candidates[1].sort(key=lambda x: x[0])
    chosen = candidates[0][:TARGET_HAM] + candidates[1][:TARGET_SPAM]

    rnd = random.Random(SEED)
    rnd.shuffle(chosen)

    EML_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = []

    for out_idx, (_key, src_idx, item) in enumerate(chosen):
        y = int(item.get("label") or 0)
        path = EML_DIR / f"{out_idx:05d}.eml"
        path.write_bytes(render_email(item, src_idx))
        source = "trec07_spam" if y else "trec07_ham"
        rows.append({"path": path, "y": y, "source": source})
        manifest.append({"file": path.name, "y": y, "source": source})

        if (out_idx + 1) % 5000 == 0:
            print("prepared TREC-07", out_idx + 1, "/", TARGET_TOTAL, flush=True)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest))

    print(
        "Fresh locked test prepared:",
        len(rows),
        "spam", sum(r["y"] for r in rows),
        "ham", sum(not r["y"] for r in rows),
        flush=True,
    )
    return rows


def rewrite_reports():
    src_json = Path("reports/v24-crossfit.json")
    src_md = Path("reports/v24-crossfit.md")

    if src_json.exists():
        obj = json.loads(src_json.read_text())
        obj["version"] = "v26-fresh-trec07-50k"
        obj["dataset"]["lockedTest"] = TARGET_TOTAL
        obj["dataset"]["testCorpus"] = "TREC-07 fresh external lockbox"
        obj["dataset"]["testSelection"] = (
            "50,000 unique TREC-07 messages selected deterministically by content hash; "
            "22,000 ham + 28,000 spam"
        )
        obj["method"]["testLabelsUsedForTrainingOrThresholds"] = False
        obj["method"]["individualTestErrorsInspected"] = False
        obj["method"]["externalTestDomain"] = True
        obj["warning"] = (
            "TREC-07 is a historical public corpus and the Hugging Face distribution "
            "normalizes message fields. This is a fresh external-domain engineering "
            "lockbox, not a direct estimate for modern customer traffic."
        )
        Path("reports/v26-trec50k.json").write_text(json.dumps(obj, indent=2))

    if src_md.exists():
        txt = src_md.read_text()
        txt = txt.replace(
            "# MailGuard v24 hard-ham-veto benchmark",
            "# MailGuard v26 fresh TREC-07 50k lockbox",
        )
        txt = txt.replace("v24 ", "v26 ")
        txt += (
            "\nFresh external test corpus: TREC-07-derived Hugging Face distribution. "
            "Exactly 50,000 unique messages (28,000 spam + 22,000 ham). "
            "Test labels were not used for model fitting or gate selection.\n"
        )
        Path("reports/v26-trec50k.md").write_text(txt)


def main():
    fresh_test = prepare_trec50k()
    original_split = v18.split_enron

    def split_with_fresh_test(rows):
        train, val, _old_test = original_split(rows)
        return train, val, fresh_test

    # benchmark_v24 and benchmark_v25 share this imported v18 module object.
    v18.split_enron = split_with_fresh_test

    # Reuse the v25 soft two-tier gate logic unchanged; only the final lockbox changes.
    v25.base.select_dual_guard = v25.select_dual_guard
    v25.base.apply_dual_guard = v25.apply_dual_guard
    v25.base.MODELS = Path("models/v26")
    v25.base.SEED = 20261006
    v25.base.main()

    rewrite_reports()


if __name__ == "__main__":
    main()
