#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import benchmark_v18_overnight as v18
import benchmark_v25_soft_veto as v25
import benchmark_v28_crosssource as v28
import benchmark_v30_balanced_lowfp as v30
import benchmark_v33_overnight_neural as v33
import benchmark_v34_hybrid_rescue as v34
import benchmark_v36_bugfix_hardmine as v36

base = v25.base
v3 = base.v3

SEED = 20261017
OUT_MODELS = Path("models/v37")
REPORT_JSON = Path("reports/v37-corrected-v34-50k.json")
REPORT_MD = Path("reports/v37-corrected-v34-50k.md")


def write_report():
    src_json = Path("reports/v24-crossfit.json")
    src_md = Path("reports/v24-crossfit.md")

    if src_json.exists():
        obj = json.loads(src_json.read_text())
        obj["version"] = "v37-corrected-data-v34-ablation"
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
            "v33/v34 neural training; NO v36 hard-mining"
        )
        obj["method"]["forensicHybrid"] = "v34 78/22 subject + source committee"
        obj["method"]["gate"] = "v30 hard low-FP two-tier gate"
        obj["method"]["purpose"] = (
            "ablation: isolate corrected-data effect from v36 hard-mining effect"
        )
        obj["method"]["testLabelsUsedForTrainingOrThresholds"] = False
        obj["warning"] = (
            "This uses the already-observed TREC-05 + CEAS-08 engineering test pair. "
            "It is for debugging/ablation only, not a pristine final lockbox."
        )
        REPORT_JSON.write_text(json.dumps(obj, indent=2))

    if src_md.exists():
        txt = src_md.read_text()
        txt = txt.replace(
            "# MailGuard v24 hard-ham-veto benchmark",
            "# MailGuard v37 corrected-data v34 ablation 50k benchmark",
        )
        txt = txt.replace("v24 ", "v37 ")
        txt = txt.replace(
            "The 10k test labels are not used",
            "The 50k test labels are not used",
        )
        txt += (
            "\nV37 is a controlled ablation. It keeps the corrected content-only "
            "dedupe/conflicting-label removal from v36, but restores the v34 neural "
            "training path with no v36 hard-mining. This tells us whether the v36 "
            "recall drop came mainly from benchmark correction or from hard-mining.\n"
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

    # Restore the exact v34-style neural path: mild source weights + larger v33 net,
    # but NOT v36 hard-example mining.
    v18.base_weights = v33.mild_source_weights
    v18.DeepMailNet = v33.OvernightMailNet
    v18.train_model = v33.overnight_train_model
    v18.MODELS = OUT_MODELS
    v33.OUT_MODELS = OUT_MODELS

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
