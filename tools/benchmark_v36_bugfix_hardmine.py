#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import benchmark_v18_overnight as v18
import benchmark_v25_soft_veto as v25
import benchmark_v27_multidomain as md
import benchmark_v28_crosssource as v28
import benchmark_v30_balanced_lowfp as v30
import benchmark_v31_sanitized_crosssource as v31
import benchmark_v33_overnight_neural as v33
import benchmark_v34_hybrid_rescue as v34

base = v25.base
v3 = base.v3

SEED = 20261016
CACHE = Path(".cache/v36-corrected-crosssource")
EML_DIR = CACHE / "eml"
MANIFEST = CACHE / "manifest.json"
OUT_MODELS = Path("models/v36")
REPORT_JSON = Path("reports/v36-bugfix-hardmine-50k.json")
REPORT_MD = Path("reports/v36-bugfix-hardmine-50k.md")

GENERAL_MODELS = []


def content_key(item):
    """
    Content-only identity.

    BUG FIX: v27-v35 included the label in the dedupe hash. The same normalized
    message with conflicting labels therefore received two different hashes and
    could survive the global dedupe. Labels must never be part of content identity.
    """
    payload = "\n".join([
        md.clean(item.get("sender")).lower(),
        md.clean(item.get("receiver")).lower(),
        md.clean(item.get("subject")).lower(),
        " ".join(str(item.get("text") or "").lower().split()),
    ]).encode("utf-8", "ignore")
    return hashlib.sha256(payload).hexdigest()


def _load_cache():
    if not MANIFEST.exists() or not EML_DIR.exists():
        return None
    try:
        obj = json.loads(MANIFEST.read_text())
    except Exception:
        return None
    if obj.get("version") != 36:
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
        len(out["train"]) != v28.EXPECTED_TRAIN
        or len(out["val"]) != v28.EXPECTED_VAL
        or len(out["test"]) != v28.EXPECTED_TEST
    ):
        return None
    return out


def prepare_corrected():
    cached = _load_cache()
    if cached is not None:
        print(
            "v36 corrected cache",
            len(cached["train"]), len(cached["val"]), len(cached["test"]),
            flush=True,
        )
        return cached

    from datasets import load_dataset

    ds = load_dataset("JinqiangDing/seven-phishing-email-datasets")["train"]
    print("HF rows", len(ds), flush=True)

    counts = Counter()
    labels_by_key = defaultdict(set)

    for i, item in enumerate(ds):
        k = content_key(item)
        counts[k] += 1
        labels_by_key[k].add(int(item.get("label") or 0))
        if (i + 1) % 25000 == 0:
            print("v36 content-dedupe pass", i + 1, "/", len(ds), flush=True)

    conflict_keys = {k for k, labels in labels_by_key.items() if len(labels) > 1}
    duplicate_keys = {k for k, n in counts.items() if n > 1}

    buckets = defaultdict(list)
    dropped_duplicate = 0
    dropped_conflict = 0

    for i, item in enumerate(ds):
        k = content_key(item)
        if k in conflict_keys:
            dropped_conflict += 1
            continue
        if k in duplicate_keys:
            dropped_duplicate += 1
            continue

        source = md.clean(item.get("dataset_name"))
        y = int(item.get("label") or 0)
        if y in (0, 1):
            buckets[(source, y)].append((k, i))

    for arr in buckets.values():
        arr.sort(key=lambda x: x[0])

    print(
        "v36 dedupe",
        "duplicate_rows_dropped", dropped_duplicate,
        "conflicting_label_rows_dropped", dropped_conflict,
        "conflict_keys", len(conflict_keys),
        flush=True,
    )

    for key in sorted(set(v28.TRAIN_PLAN) | set(v28.VAL_PLAN) | set(v28.TEST_PLAN)):
        print("v36 bucket", key, len(buckets[key]), flush=True)

    cursor = defaultdict(int)
    picked = {"train": [], "val": [], "test": []}

    def take(split, plan):
        for bucket_key, n in plan.items():
            arr = buckets[bucket_key]
            start = cursor[bucket_key]
            end = start + n
            if end > len(arr):
                raise RuntimeError(
                    f"v36 not enough content-unique rows for {bucket_key}: "
                    f"need through {end}, have {len(arr)}"
                )
            for _, idx in arr[start:end]:
                picked[split].append((bucket_key[0], bucket_key[1], idx))
            cursor[bucket_key] = end

    take("train", v28.TRAIN_PLAN)
    take("val", v28.VAL_PLAN)
    take("test", v28.TEST_PLAN)

    EML_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 36,
        "contentKeyExcludesLabel": True,
        "conflictKeysDropped": len(conflict_keys),
        "train": [], "val": [], "test": [],
    }
    out = {"train": [], "val": [], "test": []}

    for split in ("train", "val", "test"):
        picked[split].sort(
            key=lambda x: hashlib.sha256(f"{x[0]}:{x[2]}".encode()).hexdigest()
        )
        for j, (source, y, idx) in enumerate(picked[split]):
            filename = f"{split}-{j:05d}.eml"
            path = EML_DIR / filename
            # Keep v31 source-neutral raw EML.
            path.write_bytes(v31.sanitized_render(ds[idx], source, idx))
            out[split].append({"path": path, "y": y, "source": source})
            manifest[split].append({"file": filename, "y": y, "source": source})
            if (j + 1) % 5000 == 0:
                print("v36 render", split, j + 1, "/", len(picked[split]), flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest))

    for split in ("train", "val", "test"):
        print(
            "v36", split, len(out[split]),
            "spam", sum(r["y"] for r in out[split]),
            "ham", sum(not r["y"] for r in out[split]),
            "sources", dict(Counter(r["source"] for r in out[split])),
            flush=True,
        )

    return out


class ItemDataset(Dataset):
    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def hard_mining_train_model(
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
    """
    BUG FIX: on the sanitized public corpora Rspamd scores are near zero, so the
    old condition rscore <= 3.5 selected ~all 40k rows for the "deep specialist".
    That made deep experts almost duplicates of general experts.

    V36 mines actual hard examples using the already-trained general neural
    ensemble. Deep models then see spam that general models score lowest and ham
    that they score highest, with a small anchor set for stability.
    """
    global GENERAL_MODELS

    if "general" in name.lower():
        model, meta = v33.overnight_train_model(
            name, train_ds, val_ds, val_rows, seed,
            max_epochs=max_epochs,
            patience=patience,
            min_epochs=min_epochs,
            max_added_fp=max_added_fp,
        )
        GENERAL_MODELS.append(model)
        return model, meta

    if "deep" not in name.lower() or not GENERAL_MODELS:
        return v33.overnight_train_model(
            name, train_ds, val_ds, val_rows, seed,
            max_epochs=max_epochs,
            patience=patience,
            min_epochs=min_epochs,
            max_added_fp=max_added_fp,
        )

    probs, _ = v18.ensemble_predict(GENERAL_MODELS, train_ds)
    labels = np.asarray(
        [int(item.label > 0.5) for item in train_ds.items],
        dtype=np.int32,
    )

    spam_idx = np.where(labels == 1)[0]
    ham_idx = np.where(labels == 0)[0]

    # Hard spam = lowest general probability; hard ham = highest general probability.
    spam_sorted = spam_idx[np.argsort(probs[spam_idx])]
    ham_sorted = ham_idx[np.argsort(-probs[ham_idx])]

    # Keep most hard spam because recall is our bottleneck; fewer hard ham are enough
    # to preserve the FP discipline. Add random anchors so specialists do not collapse.
    n_spam_hard = max(1500, int(round(len(spam_sorted) * 0.72)))
    n_ham_hard = max(1200, int(round(len(ham_sorted) * 0.38)))

    selected = set(spam_sorted[:n_spam_hard].tolist())
    selected.update(ham_sorted[:n_ham_hard].tolist())

    rng = np.random.default_rng(seed)
    spam_anchor = rng.choice(
        spam_idx, size=min(len(spam_idx), max(500, len(spam_idx) // 12)), replace=False
    )
    ham_anchor = rng.choice(
        ham_idx, size=min(len(ham_idx), max(500, len(ham_idx) // 16)), replace=False
    )
    selected.update(spam_anchor.tolist())
    selected.update(ham_anchor.tolist())

    selected = sorted(selected)
    hard_ds = ItemDataset([train_ds.items[i] for i in selected])
    hard_labels = np.asarray([int(x.label > 0.5) for x in hard_ds.items])

    print(
        f"v36 hard mining {name}: original={len(train_ds)} hard={len(hard_ds)} "
        f"spam={int(hard_labels.sum())} ham={int((hard_labels == 0).sum())} "
        f"general_p_mean={float(probs.mean()):.4f} "
        f"hard_spam_cut={float(probs[spam_sorted[min(n_spam_hard-1, len(spam_sorted)-1)]]) if len(spam_sorted) else 0:.4f} "
        f"hard_ham_cut={float(probs[ham_sorted[min(n_ham_hard-1, len(ham_sorted)-1)]]) if len(ham_sorted) else 0:.4f}",
        flush=True,
    )

    model, meta = v33.overnight_train_model(
        name, hard_ds, val_ds, val_rows, seed,
        max_epochs=max_epochs,
        patience=patience,
        min_epochs=min_epochs,
        max_added_fp=max_added_fp,
    )
    meta = dict(meta or {})
    meta["hardMining"] = {
        "originalCount": len(train_ds),
        "hardCount": len(hard_ds),
        "hardSpam": int(hard_labels.sum()),
        "hardHam": int((hard_labels == 0).sum()),
    }
    return model, meta


def write_report():
    src_json = Path("reports/v24-crossfit.json")
    src_md = Path("reports/v24-crossfit.md")

    if src_json.exists():
        obj = json.loads(src_json.read_text())
        obj["version"] = "v36-bugfix-hardmine-lowfp"
        obj["dataset"]["training"] = v28.EXPECTED_TRAIN
        obj["dataset"]["validation"] = v28.EXPECTED_VAL
        obj["dataset"]["lockedTest"] = v28.EXPECTED_TEST
        obj["method"]["dedupeIdentity"] = (
            "normalized sender+receiver+subject+body SHA256; label excluded"
        )
        obj["method"]["conflictingLabels"] = (
            "all content keys occurring with both labels are dropped before splitting"
        )
        obj["method"]["sourceArtifactSanitization"] = True
        obj["method"]["deepSpecialists"] = (
            "hard-example mined from general-neural probabilities instead of "
            "Rspamd rscore<=3.5, which selected nearly the whole stripped corpus"
        )
        obj["method"]["forensicHybrid"] = "v34 78/22 subject + source committee"
        obj["method"]["gate"] = "v30 hard low-FP two-tier gate"
        obj["method"]["testLabelsUsedForTrainingOrThresholds"] = False
        obj["warning"] = (
            "This fixes benchmark/training bugs but still uses the previously observed "
            "TREC-05 + CEAS-08 engineering test pair. Use a fresh untouched corpus for "
            "the final frozen-model claim."
        )
        REPORT_JSON.write_text(json.dumps(obj, indent=2))

    if src_md.exists():
        txt = src_md.read_text()
        txt = txt.replace(
            "# MailGuard v24 hard-ham-veto benchmark",
            "# MailGuard v36 bugfix + hard-mining 50k benchmark",
        )
        txt = txt.replace("v24 ", "v36 ")
        txt = txt.replace(
            "The 10k test labels are not used",
            "The 50k test labels are not used",
        )
        txt += (
            "\nV36 fixes two concrete bugs. First, global dedupe is now content-only: "
            "the label is no longer part of the message hash, and contradictory-label "
            "copies are removed before splitting. Second, the old deep-neural selector "
            "used Rspamd <= 3.5; on stripped public corpora this selected essentially "
            "the entire training set, so the 'deep specialists' duplicated the general "
            "nets. V36 mines actual hard spam/ham from general-neural predictions.\n"
        )
        REPORT_MD.write_text(txt)


def main():
    data = prepare_corrected()

    v33.SOURCE_BY_PATH = {
        str(row["path"]): row.get("source", "unknown")
        for split in ("train", "val", "test")
        for row in data[split]
    }

    v18.enron.prepare_enron = lambda: data["train"] + data["val"] + data["test"]
    v18.split_enron = lambda _rows: (data["train"], data["val"], data["test"])
    v3.build_splits = lambda _groups: ([], [], [])

    if not hasattr(v18, "_v33_original_base_weights"):
        v18._v33_original_base_weights = v18.base_weights

    v18.base_weights = v33.mild_source_weights
    v18.DeepMailNet = v33.OvernightMailNet
    v18.train_model = hard_mining_train_model
    v18.MODELS = OUT_MODELS
    v33.OUT_MODELS = OUT_MODELS

    # Start from v34, which was the best recall/FP trade-off before these fixes.
    base.fit_forensic_text_experts = v34.fit_hybrid_forensic
    base.select_dual_guard = v30.select_balanced_guard
    base.apply_dual_guard = v30.apply_balanced_guard
    base.chronological_folds = v28.source_heldout_folds
    base.MODELS = OUT_MODELS
    base.SEED = SEED

    base.main()
    write_report()


if __name__ == "__main__":
    main()
