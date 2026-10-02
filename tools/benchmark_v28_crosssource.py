#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import benchmark_v18_overnight as v18
import benchmark_v25_soft_veto as v25
import benchmark_v27_multidomain as md

base = v25.base
v3 = base.v3

SEED = 20261009

TRAIN_PLAN = {
    ("TREC-07", 0): 7000, ("TREC-07", 1): 8700,
    ("Enron", 0): 6000, ("Enron", 1): 6000,
    ("TREC-06", 0): 5000, ("TREC-06", 1): 2500,
    ("Assassin", 0): 2000, ("Assassin", 1): 1000,
    ("Ling", 0): 1500, ("Ling", 1): 300,
}

VAL_PLAN = {
    ("TREC-07", 0): 1200, ("TREC-07", 1): 1600,
    ("Enron", 0): 800, ("Enron", 1): 800,
    ("TREC-06", 0): 700, ("TREC-06", 1): 300,
    ("Assassin", 0): 500, ("Assassin", 1): 150,
    ("Ling", 0): 300, ("Ling", 1): 50,
}

# Completely unseen source families for the final 50k lockbox.
TEST_PLAN = {
    ("TREC-05", 0): 13000, ("TREC-05", 1): 13000,
    ("CEAS-08", 0): 12000, ("CEAS-08", 1): 12000,
}

EXPECTED_TRAIN = 40000
EXPECTED_VAL = 6400
EXPECTED_TEST = 50000

REPORT_JSON = Path("reports/v28-crosssource-50k.json")
REPORT_MD = Path("reports/v28-crosssource-50k.md")


def source_heldout_folds(rows, n_folds=5):
    """
    Each OOF fold is a whole source family, not a random/within-source slice.
    The meta model therefore has to predict a source it did not see in the
    other folds. This is the key change versus v27.
    """
    source_names = sorted({str(r.get("source", "unknown")) for r in rows})
    if len(source_names) != n_folds:
        raise RuntimeError(
            f"Expected {n_folds} validation source families, got {source_names}"
        )

    source_to_fold = {name: i for i, name in enumerate(source_names)}
    fold_id = np.asarray(
        [source_to_fold[str(r.get("source", "unknown"))] for r in rows],
        dtype=np.int32,
    )

    # Every held-out source must contain both ham and spam, otherwise recall/FP
    # stability for that fold would be meaningless.
    by_fold = defaultdict(lambda: [0, 0])
    for fid, row in zip(fold_id, rows):
        by_fold[int(fid)][int(bool(row["y"]))] += 1
    for fid, (ham, spam) in sorted(by_fold.items()):
        if ham == 0 or spam == 0:
            raise RuntimeError(
                f"Source-held-out fold {fid} lacks a class: ham={ham} spam={spam}"
            )
        print(
            "v28 source-held-out fold",
            fid,
            source_names[fid],
            "ham", ham,
            "spam", spam,
            flush=True,
        )
    return fold_id


def write_report():
    src_json = Path("reports/v24-crossfit.json")
    src_md = Path("reports/v24-crossfit.md")

    if src_json.exists():
        obj = json.loads(src_json.read_text())
        obj["version"] = "v28-crosssource-fresh50k"
        obj["dataset"]["training"] = EXPECTED_TRAIN
        obj["dataset"]["validation"] = EXPECTED_VAL
        obj["dataset"]["lockedTest"] = EXPECTED_TEST
        obj["dataset"]["trainingSources"] = [
            "TREC-07", "Enron", "TREC-06", "Assassin", "Ling"
        ]
        obj["dataset"]["testSources"] = ["TREC-05", "CEAS-08"]
        obj["dataset"]["split"] = (
            "global normalized exact-dedupe; train/validation source families are "
            "disjoint from the two final test families"
        )
        obj["method"]["metaValidation"] = (
            "5-fold leave-one-source-family-out validation"
        )
        obj["method"]["testLabelsUsedForTrainingOrThresholds"] = False
        obj["method"]["individualTestErrorsInspected"] = False
        obj["method"]["sourceSeparatedExternalTest"] = True
        obj["warning"] = (
            "Historical public corpora with normalized message fields. This is a "
            "cross-corpus robustness benchmark, not an estimate of current customer traffic."
        )
        REPORT_JSON.write_text(json.dumps(obj, indent=2))

    if src_md.exists():
        txt = src_md.read_text()
        txt = txt.replace(
            "# MailGuard v24 hard-ham-veto benchmark",
            "# MailGuard v28 cross-source fresh 50k benchmark",
        )
        txt = txt.replace("v24 ", "v28 ")
        txt += (
            "\nKey methodology change: meta/gate validation is leave-one-source-family-out, "
            "not within-source cross-validation. Training/validation sources are TREC-07, "
            "Enron, TREC-06, Assassin and Ling. Final test uses only unseen TREC-05 + CEAS-08. "
            "Exact normalized duplicates are removed globally. Test labels are not used for "
            "training or threshold selection.\n"
        )
        REPORT_MD.write_text(txt)


def main():
    # Reconfigure the reusable multi-domain loader with a source-separated plan.
    md.CACHE = Path(".cache/v28-crosssource")
    md.EML_DIR = md.CACHE / "eml"
    md.MANIFEST = md.CACHE / "manifest.json"
    md.TRAIN_PLAN = TRAIN_PLAN
    md.VAL_PLAN = VAL_PLAN
    md.TEST_PLAN = TEST_PLAN
    md.EXPECTED_TRAIN = EXPECTED_TRAIN
    md.EXPECTED_VAL = EXPECTED_VAL
    md.EXPECTED_TEST = EXPECTED_TEST

    data = md.prepare()

    print(
        "v28 prepared",
        "train", len(data["train"]),
        "val", len(data["val"]),
        "test", len(data["test"]),
        flush=True,
    )

    # Feed the source-separated data into the mature v25 architecture.
    v18.enron.prepare_enron = lambda: data["train"] + data["val"] + data["test"]
    v18.split_enron = lambda _rows: (data["train"], data["val"], data["test"])
    v3.build_splits = lambda _groups: ([], [], [])

    base.select_dual_guard = v25.select_dual_guard
    base.apply_dual_guard = v25.apply_dual_guard
    base.chronological_folds = source_heldout_folds
    base.MODELS = Path("models/v28")
    base.SEED = SEED

    base.main()
    write_report()


if __name__ == "__main__":
    main()
