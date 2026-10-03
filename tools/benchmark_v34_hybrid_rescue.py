#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.svm import LinearSVC

import benchmark_v18_overnight as v18
import benchmark_v25_soft_veto as v25
import benchmark_v27_multidomain as md
import benchmark_v28_crosssource as v28
import benchmark_v30_balanced_lowfp as v30
import benchmark_v31_sanitized_crosssource as v31
import benchmark_v33_overnight_neural as v33

base = v25.base
v3 = base.v3
ORIGINAL_FORENSIC = base.fit_forensic_text_experts

SEED = 20261014
OUT_MODELS = Path("models/v34")
REPORT_JSON = Path("reports/v34-hybrid-rescue-50k.json")
REPORT_MD = Path("reports/v34-hybrid-rescue-50k.md")


def fit_hybrid_forensic(train_rows, val_rows, test_rows):
    """
    Keep the strong v31 word/char/subject experts, but add a small source-robust
    correction to the subject slot.

    V32 replaced the subject signal entirely with a source committee and lost too
    much recall. V34 instead blends the original subject expert with a median vote
    from one word-SVM per training corpus. The committee acts as a conservative
    cross-domain sanity check, not as the main classifier.
    """
    original = ORIGINAL_FORENSIC(train_rows, val_rows, test_rows)
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

        pv = expit(np.asarray(clf.decision_function(xval), dtype=np.float64))
        pe = expit(np.asarray(clf.decision_function(xtest), dtype=np.float64))
        val_votes.append(pv)
        test_votes.append(pe)
        committee_models.append((source, clf))

        print(
            f"v34 committee {source} train={len(idx)} "
            f"val_mean={float(pv.mean()):.4f} test_mean={float(pe.mean()):.4f}",
            flush=True,
        )

    if val_votes:
        committee_val = np.median(np.vstack(val_votes), axis=0)
        committee_test = np.median(np.vstack(test_votes), axis=0)
    else:
        committee_val = pword_val.copy()
        committee_test = pword_test.copy()

    # Mostly preserve the proven subject signal. The committee only suppresses
    # predictions that look source-specific across unrelated training corpora.
    eps = 1e-7
    hybrid_val = np.exp(
        0.78 * np.log(np.clip(psubject_val, eps, 1.0))
        + 0.22 * np.log(np.clip(committee_val, eps, 1.0))
    )
    hybrid_test = np.exp(
        0.78 * np.log(np.clip(psubject_test, eps, 1.0))
        + 0.22 * np.log(np.clip(committee_test, eps, 1.0))
    )

    print(
        "v34 hybrid forensic means",
        f"subject={float(psubject_val.mean()):.4f}/{float(psubject_test.mean()):.4f}",
        f"committee={float(committee_val.mean()):.4f}/{float(committee_test.mean()):.4f}",
        f"hybrid={float(hybrid_val.mean()):.4f}/{float(hybrid_test.mean()):.4f}",
        flush=True,
    )

    models = list(original["models"]) + [("source-committee", committee_models)]
    vectorizers = dict(original["vectorizers"])

    return {
        "models": models,
        "vectorizers": vectorizers,
        "val": [pword_val, pchar_val, hybrid_val],
        "test": [pword_test, pchar_test, hybrid_test],
    }


def write_report():
    src_json = Path("reports/v24-crossfit.json")
    src_md = Path("reports/v24-crossfit.md")

    if src_json.exists():
        obj = json.loads(src_json.read_text())
        obj["version"] = "v34-hybrid-neural-rescue-lowfp"
        obj["dataset"]["training"] = v28.EXPECTED_TRAIN
        obj["dataset"]["validation"] = v28.EXPECTED_VAL
        obj["dataset"]["lockedTest"] = v28.EXPECTED_TEST
        obj["dataset"]["trainingSources"] = [
            "TREC-07", "Enron", "TREC-06", "Assassin", "Ling"
        ]
        obj["dataset"]["testSources"] = ["TREC-05", "CEAS-08"]
        obj["method"]["sourceArtifactSanitization"] = True
        obj["method"]["neural"] = (
            "v33 larger two-tower PyTorch ensemble with internal train-only "
            "checkpoint holdouts"
        )
        obj["method"]["forensicHybrid"] = (
            "v31 word+char retained; subject signal blended 78/22 in log-space "
            "with median committee of one word-SVM per training source"
        )
        obj["method"]["sourceCommitteeRole"] = (
            "conservative auxiliary sanity-check, not replacement classifier"
        )
        obj["method"]["gate"] = "v30 hard low-FP two-tier gate"
        obj["method"]["testLabelsUsedForTrainingOrThresholds"] = False
        obj["method"]["individualTestErrorsInspected"] = False
        obj["warning"] = (
            "TREC-05 + CEAS-08 is an engineering comparison set already seen at "
            "aggregate level in earlier versions. Freeze the final design and use "
            "a new untouched corpus before final generalization claims."
        )
        REPORT_JSON.write_text(json.dumps(obj, indent=2))

    if src_md.exists():
        txt = src_md.read_text()
        txt = txt.replace(
            "# MailGuard v24 hard-ham-veto benchmark",
            "# MailGuard v34 hybrid neural-rescue low-FP 50k benchmark",
        )
        txt = txt.replace("v24 ", "v34 ")
        txt += (
            "\nV34 combines the useful parts of v31/v32/v33 instead of replacing "
            "one with another: v33's larger neural ensemble is retained, v31's strong "
            "word/char/subject forensic branch is retained, and only 22% of the third "
            "forensic signal comes from v32-style cross-source committee consensus. "
            "The goal is to keep v33 recall while using committee disagreement to "
            "suppress source-specific false positives.\n"
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

    base.fit_forensic_text_experts = fit_hybrid_forensic
    base.select_dual_guard = v30.select_balanced_guard
    base.apply_dual_guard = v30.apply_balanced_guard
    base.chronological_folds = v28.source_heldout_folds
    base.MODELS = OUT_MODELS
    base.SEED = SEED

    base.main()
    write_report()


if __name__ == "__main__":
    main()
