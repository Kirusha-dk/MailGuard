#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
ORIGINAL_META_SCORE = base.meta_score

SEED = 20261015
OUT_MODELS = Path("models/v35")
REPORT_JSON = Path("reports/v35-recall-boost-50k.json")
REPORT_MD = Path("reports/v35-recall-boost-50k.md")


def recall_profile(name):
    # V34 already showed that FP around 1% on this public stress test is acceptable
    # for the current engineering goal. Spend a little more validation FP budget
    # specifically on the high-recall mode, but keep hard per-source caps.
    if name == "ultra-safe":
        return {"fp": 0, "fold_fp": 0, "target": 0.70, "worst": 0.35}
    if name == "safe":
        return {"fp": 2, "fold_fp": 1, "target": 0.90, "worst": 0.55}
    if name == "target-93":
        return {"fp": 5, "fold_fp": 2, "target": 0.94, "worst": 0.62}
    if name == "target-95":
        return {"fp": 10, "fold_fp": 2, "target": 0.96, "worst": 0.68}
    return {"fp": 5, "fold_fp": 2, "target": 0.92, "worst": 0.60}


def recall_meta_score(models, x, raw_probs):
    """
    Preserve the v24/v34 stack, then add a conservative rescue channel.

    The rescue is activated only when both neural towers agree with several
    independent text branches and the ham expert is not strongly confident.
    This is aimed at the false negatives that v34 still leaves behind.
    """
    score, pl, pt, agree = ORIGINAL_META_SCORE(models, x, raw_probs)

    eps = 1e-8
    neural = np.sqrt(
        np.clip(raw_probs[5], eps, 1.0) * np.clip(raw_probs[6], eps, 1.0)
    )
    text = np.median(
        raw_probs[[0, 1, 4, 7, 8, 9], :],
        axis=0,
    )
    not_ham = np.clip(raw_probs[2], eps, 1.0)
    agree72 = np.sum(raw_probs >= 0.72, axis=0)

    rescue = np.exp(
        0.52 * np.log(np.clip(neural, eps, 1.0))
        + 0.38 * np.log(np.clip(text, eps, 1.0))
        + 0.10 * np.log(not_ham)
    )
    rescue *= 0.90 + 0.10 * np.clip(agree72 / raw_probs.shape[0], 0.0, 1.0)

    rescue_mask = (
        (neural >= 0.70)
        & (text >= 0.56)
        & (not_ham >= 0.20)
        & (agree72 >= 5)
    )

    boosted = np.where(rescue_mask, 0.98 * rescue, 0.0)
    score = np.maximum(score, boosted)

    print(
        "v35 rescue",
        "eligible", int(rescue_mask.sum()),
        "mean_neural", float(neural.mean()),
        "mean_text", float(text.mean()),
        flush=True,
    )
    return np.clip(score, 0.0, 1.0), pl, pt, agree


def lighter_hybrid_forensic(train_rows, val_rows, test_rows):
    """
    Reuse v34's source-committee protection but make it only a light veto.

    V34 used a 22% committee share and cut FP very effectively, but also cost
    recall. V35 keeps only 8% committee weight and 92% of the original subject
    expert, because the user explicitly wants to spend some FP headroom on recall.
    """
    original = v34.ORIGINAL_FORENSIC(train_rows, val_rows, test_rows)
    pword_val, pchar_val, psubject_val = original["val"]
    pword_test, pchar_test, psubject_test = original["test"]

    word = original["vectorizers"]["word"]
    tr_docs, _ = base.tfidf_documents(train_rows)
    va_docs, _ = base.tfidf_documents(val_rows)
    te_docs, _ = base.tfidf_documents(test_rows)

    xtr = word.transform(tr_docs)
    xval = word.transform(va_docs)
    xtest = word.transform(te_docs)
    y = np.asarray([int(bool(r["y"])) for r in train_rows], dtype=np.int32)

    from sklearn.svm import LinearSVC
    from scipy.special import expit

    val_votes = []
    test_votes = []
    committee_models = []
    sources = sorted({r.get("source", "unknown") for r in train_rows})

    for k, source in enumerate(sources):
        idx = np.asarray(
            [i for i, r in enumerate(train_rows) if r.get("source", "unknown") == source],
            dtype=np.int64,
        )
        sy = y[idx]
        if len(idx) < 100 or len(np.unique(sy)) < 2:
            continue

        ham = sy == 0
        spam = sy == 1
        w = np.ones(len(idx), dtype=np.float64)
        w[ham] = len(idx) / (2.0 * max(1, int(ham.sum()))) * 1.8
        w[spam] = len(idx) / (2.0 * max(1, int(spam.sum())))

        clf = LinearSVC(C=0.95, random_state=SEED + 100 + 17 * k)
        clf.fit(xtr[idx], sy, sample_weight=w)
        val_votes.append(
            expit(np.asarray(clf.decision_function(xval), dtype=np.float64))
        )
        test_votes.append(
            expit(np.asarray(clf.decision_function(xtest), dtype=np.float64))
        )
        committee_models.append((source, clf))

    if val_votes:
        committee_val = np.median(np.vstack(val_votes), axis=0)
        committee_test = np.median(np.vstack(test_votes), axis=0)
    else:
        committee_val = pword_val.copy()
        committee_test = pword_test.copy()

    eps = 1e-7
    hybrid_val = np.exp(
        0.92 * np.log(np.clip(psubject_val, eps, 1.0))
        + 0.08 * np.log(np.clip(committee_val, eps, 1.0))
    )
    hybrid_test = np.exp(
        0.92 * np.log(np.clip(psubject_test, eps, 1.0))
        + 0.08 * np.log(np.clip(committee_test, eps, 1.0))
    )

    return {
        "models": list(original["models"]) + [("source-committee", committee_models)],
        "vectorizers": dict(original["vectorizers"]),
        "val": [pword_val, pchar_val, hybrid_val],
        "test": [pword_test, pchar_test, hybrid_test],
    }


def write_report():
    src_json = Path("reports/v24-crossfit.json")
    src_md = Path("reports/v24-crossfit.md")

    if src_json.exists():
        obj = json.loads(src_json.read_text())
        obj["version"] = "v35-recall-boost-lowfp"
        obj["dataset"]["training"] = v28.EXPECTED_TRAIN
        obj["dataset"]["validation"] = v28.EXPECTED_VAL
        obj["dataset"]["lockedTest"] = v28.EXPECTED_TEST
        obj["method"]["sourceArtifactSanitization"] = True
        obj["method"]["neural"] = "v33 larger two-tower ensemble"
        obj["method"]["forensicHybridCommitteeWeight"] = 0.08
        obj["method"]["neuralTextRescue"] = (
            "neural geometric consensus + median independent text consensus; "
            "requires >=5 branches above 0.72 and weak ham evidence"
        )
        obj["method"]["validationFpBudgets"] = {
            "ultraSafe": 0, "safe": 2, "target93": 5, "target95": 10
        }
        obj["method"]["testLabelsUsedForTrainingOrThresholds"] = False
        obj["warning"] = (
            "The 50k TREC-05 + CEAS-08 pair is now an engineering benchmark, not a "
            "pristine final lockbox. Use a new untouched corpus after architecture freeze."
        )
        REPORT_JSON.write_text(json.dumps(obj, indent=2))

    if src_md.exists():
        txt = src_md.read_text()
        txt = txt.replace(
            "# MailGuard v24 hard-ham-veto benchmark",
            "# MailGuard v35 recall-boost 50k benchmark",
        )
        txt = txt.replace("v24 ", "v35 ")
        txt += (
            "\nV35 intentionally spends some of the FP headroom gained by v34 to raise "
            "recall. The v34 source committee is reduced from 22% to 8%, and a new "
            "neural+text rescue score is allowed only when the big v33 neural towers "
            "and at least five independent branches agree. The high-recall validation "
            "FP budget is raised from 8 to 10 with per-source caps retained.\n"
        )
        REPORT_MD.write_text(txt)


def main():
    md.render = v31.sanitized_render
    md.CACHE = Path(".cache/v31-sanitized-crosssource")
    md.EML_DIR = md.CACHE / "eml"
    md.MANIFEST = md.CACHE / "manifest.json"
    md.TRAIN_PLAN = v28.TRAIN_PLAN
    md.VAL_PLAN = v28.VAL_PLAN
    md.TEST_PLAN = v28.TEST_PLAN
    md.EXPECTED_TRAIN = v28.EXPECTED_TRAIN
    md.EXPECTED_VAL = v28.EXPECTED_VAL
    md.EXPECTED_TEST = v28.EXPECTED_TEST

    data = md.prepare()

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
    v18.train_model = v33.overnight_train_model
    v18.MODELS = OUT_MODELS
    v33.OUT_MODELS = OUT_MODELS

    base.fit_forensic_text_experts = lighter_hybrid_forensic
    base.meta_score = recall_meta_score

    # v30 selector calls its module-global profile(); swap in the recall-oriented
    # profile while keeping the same hard two-tier gating implementation.
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
