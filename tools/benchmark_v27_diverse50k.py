#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import benchmark_v18_overnight as v18
import benchmark_v25_soft_veto as v25

CACHE = Path(".cache/v27-diverse50k")
EML_DIR = CACHE / "eml"
MANIFEST = CACHE / "manifest.json"

TREC_TRAIN_PER_CLASS = 15000
TREC_VAL_PER_CLASS = 2500
TEST_PER_CLASS = 25000
SEED = 20261008


def clean_header(value):
    if value is None:
        return ""
    text = re.sub(r"[\r\n\x00]+", " ", str(value))
    return " ".join(text.split())[:1000]


def norm_name(value):
    return re.sub(r"[^A-Z0-9]+", "_", str(value or "").upper()).strip("_")


def normalized_key(item):
    label = int(item.get("label") or 0)
    subject = clean_header(item.get("subject")).lower()
    sender = clean_header(item.get("sender")).lower()
    receiver = clean_header(item.get("receiver")).lower()
    body = " ".join(str(item.get("text") or "").lower().split())
    payload = "\n".join([str(label), sender, receiver, subject, body]).encode(
        "utf-8", "ignore"
    )
    return hashlib.sha256(payload).hexdigest()


def render_email(item, dataset_name, source_row):
    subject = clean_header(item.get("subject"))
    sender = clean_header(item.get("sender")) or "dataset-sender@example.invalid"
    receiver = clean_header(item.get("receiver")) or "mailguard-benchmark@example.invalid"
    date = clean_header(item.get("date"))
    body = str(item.get("text") or "")
    headers = [
        f"Subject: {subject}",
        f"From: {sender}",
        f"To: {receiver}",
        f"X-MailGuard-Dataset: {dataset_name}",
        f"X-MailGuard-Original-Date: {date}",
        f"X-MailGuard-Source-Row: {source_row}",
        "MIME-Version: 1.0",
        "Content-Type: text/plain; charset=utf-8",
        "Content-Transfer-Encoding: 8bit",
        "",
        "",
    ]
    return ("\r\n".join(headers)).encode("utf-8", "replace") + body.encode(
        "utf-8", "replace"
    )


def write_partition(name, records):
    folder = EML_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = []
    for i, rec in enumerate(records):
        key, source_row, dataset_name, item = rec
        y = int(item.get("label") or 0)
        path = folder / f"{i:05d}.eml"
        path.write_bytes(render_email(item, dataset_name, source_row))
        rows.append({"path": path, "y": y, "source": dataset_name})
        manifest.append({
            "file": str(path.relative_to(EML_DIR)),
            "y": y,
            "source": dataset_name,
            "key": key,
        })
        if (i + 1) % 5000 == 0:
            print("prepared", name, i + 1, "/", len(records), flush=True)
    return rows, manifest


def load_cached():
    if not MANIFEST.exists() or not EML_DIR.exists():
        return None
    try:
        obj = json.loads(MANIFEST.read_text())
    except Exception:
        return None

    required = {"trecTrain", "trecVal", "freshTest"}
    if set(obj) != required:
        return None

    expected = {
        "trecTrain": 2 * TREC_TRAIN_PER_CLASS,
        "trecVal": 2 * TREC_VAL_PER_CLASS,
        "freshTest": 2 * TEST_PER_CLASS,
    }
    out = {}
    for name, items in obj.items():
        if len(items) != expected[name]:
            return None
        rows = [
            {
                "path": EML_DIR / x["file"],
                "y": int(x["y"]),
                "source": x["source"],
            }
            for x in items
        ]
        if not all(r["path"].exists() for r in rows[:100]):
            return None
        out[name] = rows

    print(
        "v27 cache ready",
        {k: len(v) for k, v in out.items()},
        flush=True,
    )
    return out


def prepare_diverse_data():
    cached = load_cached()
    if cached is not None:
        return cached

    from datasets import load_dataset

    print("Loading multi-corpus email dataset", flush=True)
    ds = load_dataset("JinqiangDing/seven-phishing-email-datasets")

    trec = {0: [], 1: []}
    external = {0: [], 1: []}
    seen = set()
    counts = Counter()
    source_row = 0

    for split_name, split in ds.items():
        print("scan HF split", split_name, len(split), flush=True)
        for item in split:
            source_row += 1
            y = int(item.get("label") or 0)
            if y not in (0, 1):
                continue

            dataset_name = norm_name(item.get("dataset_name"))
            key = normalized_key(item)
            if key in seen:
                continue
            seen.add(key)
            counts[(dataset_name, y)] += 1
            rec = (key, source_row, dataset_name, item)

            if "TREC" in dataset_name and "07" in dataset_name:
                trec[y].append(rec)
            elif (
                "ENRON" not in dataset_name
                and "SPAMASSASSIN" not in dataset_name
                and "TREC" not in dataset_name
            ):
                external[y].append(rec)

    print("unique source counts", flush=True)
    for (name, y), n in sorted(counts.items()):
        print(name, "spam" if y else "ham", n, flush=True)

    for y in (0, 1):
        trec[y].sort(key=lambda x: x[0])
        external[y].sort(key=lambda x: (x[2], x[0]))

        need = TREC_TRAIN_PER_CLASS + TREC_VAL_PER_CLASS + TEST_PER_CLASS
        if len(trec[y]) + len(external[y]) < need:
            raise RuntimeError(
                f"Not enough unique data for class {y}: "
                f"trec={len(trec[y])}, external={len(external[y])}, need={need}"
            )
        if len(trec[y]) < TREC_TRAIN_PER_CLASS + TREC_VAL_PER_CLASS:
            raise RuntimeError(
                f"Not enough TREC-07 for train+validation class {y}: {len(trec[y])}"
            )

    train = []
    val = []
    fresh = []

    for y in (0, 1):
        train_y = trec[y][:TREC_TRAIN_PER_CLASS]
        val_y = trec[y][
            TREC_TRAIN_PER_CLASS:
            TREC_TRAIN_PER_CLASS + TREC_VAL_PER_CLASS
        ]
        trec_holdout = trec[y][
            TREC_TRAIN_PER_CLASS + TREC_VAL_PER_CLASS:
        ]

        # Prefer different corpora for the final lockbox. TREC holdout only tops
        # up a class when the external corpora do not provide 25k unique items.
        test_y = list(external[y][:TEST_PER_CLASS])
        if len(test_y) < TEST_PER_CLASS:
            need = TEST_PER_CLASS - len(test_y)
            test_y.extend(trec_holdout[:need])

        if len(test_y) != TEST_PER_CLASS:
            raise RuntimeError(
                f"Could not build {TEST_PER_CLASS} fresh test rows for class {y}"
            )

        train.extend(train_y)
        val.extend(val_y)
        fresh.extend(test_y)

    rnd = random.Random(SEED)
    rnd.shuffle(train)
    rnd.shuffle(val)
    rnd.shuffle(fresh)

    train_rows, train_manifest = write_partition("trec_train", train)
    val_rows, val_manifest = write_partition("trec_val", val)
    test_rows, test_manifest = write_partition("fresh_test", fresh)

    train_keys = {x["key"] for x in train_manifest}
    val_keys = {x["key"] for x in val_manifest}
    test_keys = {x["key"] for x in test_manifest}
    if train_keys & val_keys or train_keys & test_keys or val_keys & test_keys:
        raise RuntimeError("Content-hash leakage detected between v27 partitions")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "trecTrain": train_manifest,
        "trecVal": val_manifest,
        "freshTest": test_manifest,
    }))

    print(
        "v27 partitions:",
        "TREC train", len(train_rows),
        "TREC val", len(val_rows),
        "fresh test", len(test_rows),
        flush=True,
    )
    return {
        "trecTrain": train_rows,
        "trecVal": val_rows,
        "freshTest": test_rows,
    }


def rewrite_reports(parts):
    src_json = Path("reports/v24-crossfit.json")
    src_md = Path("reports/v24-crossfit.md")

    if src_json.exists():
        obj = json.loads(src_json.read_text())
        obj["version"] = "v27-diverse-training-fresh50k"
        obj["dataset"]["lockedTest"] = len(parts["freshTest"])
        obj["dataset"]["testSpam"] = sum(r["y"] for r in parts["freshTest"])
        obj["dataset"]["testHam"] = sum(not r["y"] for r in parts["freshTest"])
        obj["dataset"]["extraTrainingTrec07"] = len(parts["trecTrain"])
        obj["dataset"]["extraValidationTrec07"] = len(parts["trecVal"])
        obj["dataset"]["testDesign"] = (
            "Fresh 50k content-hash-disjoint lockbox. Prefer non-TREC external "
            "corpora; unused TREC-07 rows only top up a class if necessary."
        )
        obj["method"]["domainDiversityTraining"] = (
            "Original Enron training + SpamAssassin training + 30k disjoint TREC-07"
        )
        obj["method"]["testLabelsUsedForTrainingOrThresholds"] = False
        obj["method"]["individualTestErrorsInspected"] = False
        obj["warning"] = (
            "Public historical corpora are useful for engineering robustness tests "
            "but do not directly estimate current customer traffic."
        )
        Path("reports/v27-diverse50k.json").write_text(json.dumps(obj, indent=2))

    if src_md.exists():
        txt = src_md.read_text()
        txt = txt.replace(
            "# MailGuard v24 hard-ham-veto benchmark",
            "# MailGuard v27 diverse-training fresh 50k lockbox",
        )
        txt = txt.replace("v24 ", "v27 ")
        txt += (
            "\nTraining diversity: original Enron training + SpamAssassin + "
            "30,000 content-unique TREC-07 messages. Validation adds 5,000 disjoint "
            "TREC-07 messages. Final test is exactly 50,000 content-hash-disjoint "
            "messages and its labels are not used for fitting or gate selection.\n"
        )
        Path("reports/v27-diverse50k.md").write_text(txt)


def main():
    parts = prepare_diverse_data()
    original_split = v18.split_enron

    def split_with_diversity(rows):
        enron_train, enron_val, _old_test = original_split(rows)
        train = list(enron_train) + [r["path"] for r in parts["trecTrain"]]
        val = list(enron_val) + [r["path"] for r in parts["trecVal"]]
        test = [r["path"] for r in parts["freshTest"]]
        print(
            "v27 split",
            "train before SpamAssassin", len(train),
            "validation", len(val),
            "fresh test", len(test),
            flush=True,
        )
        return train, val, test

    v18.split_enron = split_with_diversity

    # Keep the v25 two-tier low-FP gate, but relearn every model on the more
    # diverse training distribution and evaluate once on a disjoint lockbox.
    v25.base.select_dual_guard = v25.select_dual_guard
    v25.base.apply_dual_guard = v25.apply_dual_guard
    v25.base.MODELS = Path("models/v27")
    v25.base.SEED = SEED
    v25.base.main()

    rewrite_reports(parts)


if __name__ == "__main__":
    main()
