#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

import benchmark_v18_overnight as v18
import benchmark_v25_soft_veto as v25
import benchmark_v28_crosssource as v28
import benchmark_v30_balanced_lowfp as v30
import benchmark_v33_overnight_neural as v33
import benchmark_v34_hybrid_rescue as v34
import benchmark_v36_bugfix_hardmine as v36

base = v25.base
v3 = base.v3

SEED = 20261018
OUT_MODELS = Path("models/v38")
REPORT_JSON = Path("reports/v38-corrected-recall-50k.json")
REPORT_MD = Path("reports/v38-corrected-recall-50k.md")


def recall_sample_weights(rows):
    """
    V37 is very conservative on the corrected corpus. Keep class balancing and
    ham protection, but reduce the old 9x ham penalty so the meta-model can rank
    more cross-domain spam above the decision boundary.
    """
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    w = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)

    for i, row in enumerate(rows):
        symbols = {name.upper() for name, _ in row["symbols"]}
        rscore = float(row["rscore"] or 0.0)

        if not row["y"]:
            w[i] *= 6.0
            if rscore >= 2.0:
                w[i] *= 1.45
            if "BAYES_HAM" in symbols:
                w[i] *= 1.30
            if "MIME_GOOD" in symbols:
                w[i] *= 1.10
        elif rscore <= 3.5:
            # On stripped public corpora most spam has a low Rspamd score.
            # Give missed-spam examples more influence because recall is now
            # the bottleneck, but do not touch the test set.
            w[i] *= 2.35

    return w


def fit_recall_ham_veto(x, rows, seed):
    """
    Same ham-veto idea as v24/v37, but less aggressive.

    The old veto was useful when FP was the bottleneck. V37 is only at 0.528% FP,
    so we can spend some of that headroom to recover false negatives.
    """
    y_ham = np.asarray([0 if r["y"] else 1 for r in rows], dtype=np.int32)
    w = compute_sample_weight(class_weight="balanced", y=y_ham).astype(np.float64)

    for i, row in enumerate(rows):
        symbols = {name.upper() for name, _ in row["symbols"]}
        rscore = float(row["rscore"] or 0.0)

        if y_ham[i]:
            w[i] *= 2.35
            if rscore >= 1.5:
                w[i] *= 1.45
            if "BAYES_HAM" in symbols:
                w[i] *= 1.20
            if "MIME_GOOD" in symbols:
                w[i] *= 1.12
        elif rscore <= 3.5:
            w[i] *= 1.55

    model = HistGradientBoostingClassifier(
        learning_rate=0.040,
        max_iter=240,
        max_leaf_nodes=15,
        min_samples_leaf=24,
        l2_regularization=2.1,
        random_state=seed,
    )
    model.fit(x, y_ham, sample_weight=w)
    return model


def recall_profile(name):
    # V37 target-95: 66.44% recall, 132 FP / 25k ham = 0.528%.
    # We explicitly allow more validation FP in the high-recall modes because
    # the user wants recall now and ~1% FP is acceptable for this engineering
    # benchmark. Per-source caps remain, so one corpus cannot consume the budget.
    if name == "ultra-safe":
        return {"fp": 0, "fold_fp": 0, "target": 0.70, "worst": 0.35}
    if name == "safe":
        return {"fp": 3, "fold_fp": 1, "target": 0.88, "worst": 0.50}
    if name == "target-93":
        return {"fp": 8, "fold_fp": 3, "target": 0.92, "worst": 0.58}
    if name == "target-95":
        return {"fp": 16, "fold_fp": 4, "target": 0.95, "worst": 0.62}
    return {"fp": 8, "fold_fp": 3, "target": 0.92, "worst": 0.58}


def write_report():
    src_json = Path("reports/v24-crossfit.json")
    src_md = Path("reports/v24-crossfit.md")

    if src_json.exists():
        obj = json.loads(src_json.read_text())
        obj["version"] = "v38-corrected-recall"
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
        obj["method"]["neural"] = (
            "v33 larger two-tower ensemble; no v36 hard-mining"
        )
        obj["method"]["forensicHybrid"] = "v34 78/22 subject + source committee"
        obj["method"]["metaHamPenalty"] = 6.0
        obj["method"]["hamVeto"] = "relaxed recall-oriented HistGradientBoosting veto"
        obj["method"]["validationFpBudgets"] = {
            "ultraSafe": 0,
            "safe": 3,
            "target93": 8,
            "target95": 16,
        }
        obj["method"]["testLabelsUsedForTrainingOrThresholds"] = False
        obj["method"]["individualTestErrorsInspected"] = False
        obj["warning"] = (
            "TREC-05 + CEAS-08 is an already-observed engineering benchmark, not a "
            "pristine final lockbox. Freeze the design and use a new untouched corpus "
            "before final generalization claims."
        )
        REPORT_JSON.write_text(json.dumps(obj, indent=2))

    if src_md.exists():
        txt = src_md.read_text()
        txt = txt.replace(
            "# MailGuard v24 hard-ham-veto benchmark",
            "# MailGuard v38 corrected recall-first 50k benchmark",
        )
        txt = txt.replace("v24 ", "v38 ")
        txt = txt.replace(
            "The 10k test labels are not used",
            "The 50k test labels are not used",
        )
        txt += (
            "\nV38 starts from the corrected-data v37 path. It does not restore the "
            "old label-in-hash bug and does not use v36 hard-mining. Instead it spends "
            "some FP headroom on recall: the meta ham penalty is reduced from 9x to 6x, "
            "the ham veto is relaxed, and target-95 may use up to 16 validation FP with "
            "hard per-source caps.\n"
        )
        REPORT_MD.write_text(txt)


def main():
    data = v36.prepare_corrected()

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

    # Keep the controlled v37 neural path: corrected data, v33 net, no hard-mining.
    v18.base_weights = v33.mild_source_weights
    v18.DeepMailNet = v33.OvernightMailNet
    v18.train_model = v33.overnight_train_model
    v18.MODELS = OUT_MODELS
    v33.OUT_MODELS = OUT_MODELS

    base.fit_forensic_text_experts = v34.fit_hybrid_forensic

    # These functions are looked up from the v24 module globals at runtime.
    base.sample_weights = recall_sample_weights
    base.fit_ham_veto = fit_recall_ham_veto

    v30.profile = recall_profile
    base.select_dual_guard = v30.select_balanced_guard
    base.apply_dual_guard = v30.apply_balanced_guard
    base.chronological_folds = v28.source_heldout_folds
    base.MODELS = OUT_MODELS
    base.SEED = SEED

    base.main()
    write_report()


if __name__ == "__main__":
    main()
