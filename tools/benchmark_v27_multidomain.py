#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import benchmark_v18_overnight as v18
import benchmark_v25_soft_veto as v25

base = v25.base
v3 = base.v3

CACHE = Path(".cache/v27-multidomain")
EML_DIR = CACHE / "eml"
MANIFEST = CACHE / "manifest.json"
REPORT_JSON = Path("reports/v27-multidomain-50k.json")
REPORT_MD = Path("reports/v27-multidomain-50k.md")

TRAIN_PLAN = {
    ("TREC-07", 0): 4000, ("TREC-07", 1): 5500,
    ("CEAS-08", 0): 4000, ("CEAS-08", 1): 4000,
    ("Enron", 0): 4000, ("Enron", 1): 4000,
    ("Assassin", 0): 2000, ("Assassin", 1): 1300,
    ("Ling", 0): 1000, ("Ling", 1): 200,
}
VAL_PLAN = {
    ("TREC-07", 0): 800, ("TREC-07", 1): 880,
    ("CEAS-08", 0): 800, ("CEAS-08", 1): 820,
    ("Enron", 0): 800, ("Enron", 1): 800,
    ("Assassin", 0): 500, ("Assassin", 1): 150,
    ("Ling", 0): 400, ("Ling", 1): 50,
}
TEST_PLAN = {
    ("TREC-05", 0): 21000, ("TREC-05", 1): 21500,
    ("TREC-06", 0): 4000, ("TREC-06", 1): 3500,
}

EXPECTED_TRAIN = 30000
EXPECTED_VAL = 6000
EXPECTED_TEST = 50000


def clean(value):
    if value is None:
        return ""
    return " ".join(re.sub(r"[\r\n\x00]+", " ", str(value)).split())


def key_for(item):
    payload = "\n".join([
        str(int(item.get("label") or 0)),
        clean(item.get("sender")).lower(),
        clean(item.get("receiver")).lower(),
        clean(item.get("subject")).lower(),
        " ".join(str(item.get("text") or "").lower().split()),
    ]).encode("utf-8", "ignore")
    return hashlib.sha256(payload).hexdigest()


def render(item, source, idx):
    sender = clean(item.get("sender")) or f"unknown@{source.lower().replace('-', '')}.invalid"
    receiver = clean(item.get("receiver")) or "mailguard-benchmark@example.invalid"
    subject = clean(item.get("subject"))
    date = clean(item.get("date"))
    body = str(item.get("text") or "")
    raw = (
        f"Subject: {subject}\r\n"
        f"From: {sender}\r\n"
        f"To: {receiver}\r\n"
        f"X-MailGuard-Dataset: v27-{source}\r\n"
        f"X-MailGuard-Original-Date: {date}\r\n"
        f"X-MailGuard-Source-Row: {idx}\r\n"
        "MIME-Version: 1.0\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "Content-Transfer-Encoding: 8bit\r\n"
        "\r\n"
    ).encode("utf-8", "replace")
    return raw + body.encode("utf-8", "replace")


def load_cached():
    if not MANIFEST.exists() or not EML_DIR.exists():
        return None
    try:
        obj = json.loads(MANIFEST.read_text())
    except Exception:
        return None
    if obj.get("version") != 1:
        return None

    out = {}
    for split in ("train", "val", "test"):
        rows = [
            {
                "path": EML_DIR / x["file"],
                "y": int(x["y"]),
                "source": x["source"],
            }
            for x in obj.get(split, [])
        ]
        if not rows or not all(r["path"].exists() for r in rows[:100]):
            return None
        out[split] = rows

    if (
        len(out["train"]) != EXPECTED_TRAIN
        or len(out["val"]) != EXPECTED_VAL
        or len(out["test"]) != EXPECTED_TEST
    ):
        return None
    return out


def prepare():
    cached = load_cached()
    if cached is not None:
        print(
            "v27 cache",
            len(cached["train"]), len(cached["val"]), len(cached["test"]),
            flush=True,
        )
        return cached

    from datasets import load_dataset

    ds = load_dataset("JinqiangDing/seven-phishing-email-datasets")["train"]
    print("HF rows", len(ds), flush=True)

    # Drop every exact normalized duplicate that occurs more than once anywhere
    # in the aggregate. This prevents cross-source exact leakage.
    counts = Counter()
    for i, item in enumerate(ds):
        counts[key_for(item)] += 1
        if (i + 1) % 25000 == 0:
            print("dedupe pass", i + 1, "/", len(ds), flush=True)

    buckets = defaultdict(list)
    dropped = 0
    for i, item in enumerate(ds):
        k = key_for(item)
        if counts[k] != 1:
            dropped += 1
            continue
        source = clean(item.get("dataset_name"))
        y = int(item.get("label") or 0)
        if y in (0, 1):
            buckets[(source, y)].append((k, i))
    for b in buckets.values():
        b.sort(key=lambda x: x[0])

    print("exact duplicate rows dropped", dropped, flush=True)
    for key in sorted(set(TRAIN_PLAN) | set(VAL_PLAN) | set(TEST_PLAN)):
        print("bucket", key, len(buckets[key]), flush=True)

    cursor = defaultdict(int)
    picked = {"train": [], "val": [], "test": []}

    def take(split, plan):
        for bucket_key, n in plan.items():
            arr = buckets[bucket_key]
            start = cursor[bucket_key]
            end = start + n
            if end > len(arr):
                raise RuntimeError(
                    f"Not enough unique rows for {bucket_key}: need through {end}, have {len(arr)}"
                )
            for _, idx in arr[start:end]:
                picked[split].append((bucket_key[0], bucket_key[1], idx))
            cursor[bucket_key] = end

    # Train/validation are only from five source families.
    take("train", TRAIN_PLAN)
    take("val", VAL_PLAN)
    # Test uses two completely unseen source families.
    take("test", TEST_PLAN)

    if len(picked["train"]) != EXPECTED_TRAIN:
        raise RuntimeError("bad train size")
    if len(picked["val"]) != EXPECTED_VAL:
        raise RuntimeError("bad val size")
    if len(picked["test"]) != EXPECTED_TEST:
        raise RuntimeError("bad test size")

    EML_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "train": [], "val": [], "test": []}
    out = {"train": [], "val": [], "test": []}

    for split in ("train", "val", "test"):
        # Stable order, independent of labels.
        picked[split].sort(
            key=lambda x: hashlib.sha256(f"{x[0]}:{x[2]}".encode()).hexdigest()
        )
        for j, (source, y, idx) in enumerate(picked[split]):
            filename = f"{split}-{j:05d}.eml"
            path = EML_DIR / filename
            path.write_bytes(render(ds[idx], source, idx))
            out[split].append({"path": path, "y": y, "source": source})
            manifest[split].append({"file": filename, "y": y, "source": source})
            if (j + 1) % 5000 == 0:
                print("render", split, j + 1, "/", len(picked[split]), flush=True)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest))

    for split in ("train", "val", "test"):
        print(
            split,
            len(out[split]),
            "spam", sum(r["y"] for r in out[split]),
            "ham", sum(not r["y"] for r in out[split]),
            "sources", dict(Counter(r["source"] for r in out[split])),
            flush=True,
        )
    return out


def source_stratified_folds(rows, n_folds=5):
    fold_id = np.full(len(rows), -1, dtype=np.int32)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row.get("source", "unknown"), int(bool(row["y"])))].append(i)

    for (source, label), idxs in groups.items():
        idxs.sort(
            key=lambda i: (
                v18.original_date(rows[i]["path"]),
                hashlib.sha256(str(rows[i]["path"]).encode()).hexdigest(),
            )
        )
        chunks = np.array_split(np.asarray(idxs, dtype=np.int64), n_folds)
        for fid, chunk in enumerate(chunks):
            fold_id[chunk] = fid

    if np.any(fold_id < 0):
        raise RuntimeError("unassigned validation fold")
    return fold_id


def write_v27_report():
    src_json = Path("reports/v24-crossfit.json")
    src_md = Path("reports/v24-crossfit.md")

    if src_json.exists():
        obj = json.loads(src_json.read_text())
        obj["version"] = "v27-multidomain-fresh50k"
        obj["dataset"]["training"] = EXPECTED_TRAIN
        obj["dataset"]["validation"] = EXPECTED_VAL
        obj["dataset"]["lockedTest"] = EXPECTED_TEST
        obj["dataset"]["trainingSources"] = ["TREC-07", "CEAS-08", "Enron", "Assassin", "Ling"]
        obj["dataset"]["testSources"] = ["TREC-05", "TREC-06"]
        obj["dataset"]["split"] = (
            "global exact-dedupe; source-separated train/validation vs fresh test domains; "
            "source+label-stratified 5-fold OOF meta calibration"
        )
        obj["method"]["testLabelsUsedForTrainingOrThresholds"] = False
        obj["method"]["individualTestErrorsInspected"] = False
        obj["method"]["sourceSeparatedExternalTest"] = True
        obj["warning"] = (
            "Historical public corpora with normalized message fields. This measures "
            "cross-corpus generalization, not current customer production accuracy."
        )
        REPORT_JSON.write_text(json.dumps(obj, indent=2))

    if src_md.exists():
        txt = src_md.read_text()
        txt = txt.replace(
            "# MailGuard v24 hard-ham-veto benchmark",
            "# MailGuard v27 multidomain fresh 50k benchmark",
        )
        txt = txt.replace("v24 ", "v27 ")
        txt += (
            "\nTraining/validation sources: TREC-07, CEAS-08, Enron, Assassin, Ling. "
            "Fresh test sources: TREC-05 + TREC-06 only. Exact normalized duplicates "
            "are removed globally before splitting. Test labels are not used for fitting "
            "or threshold selection.\n"
        )
        REPORT_MD.write_text(txt)


def main():
    data = prepare()

    # Feed v27 data into the mature v25 architecture without changing its
    # classifier logic. This isolates the effect of multi-domain training.
    v18.enron.prepare_enron = lambda: data["train"] + data["val"] + data["test"]
    v18.split_enron = lambda _rows: (data["train"], data["val"], data["test"])
    v3.build_splits = lambda _groups: ([], [], [])

    base.select_dual_guard = v25.select_dual_guard
    base.apply_dual_guard = v25.apply_dual_guard
    base.chronological_folds = source_stratified_folds
    base.MODELS = Path("models/v27")
    base.SEED = 20261008

    base.main()
    write_v27_report()


if __name__ == "__main__":
    main()
