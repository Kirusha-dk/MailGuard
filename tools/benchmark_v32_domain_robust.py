#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import benchmark_v18_overnight as v18
import benchmark_v25_soft_veto as v25
import benchmark_v27_multidomain as md
import benchmark_v28_crosssource as v28
import benchmark_v30_balanced_lowfp as v30
import benchmark_v31_sanitized_crosssource as v31

base = v25.base
v3 = base.v3

SEED = 20261012
REPORT_JSON = Path("reports/v32-domain-robust-50k.json")
REPORT_MD = Path("reports/v32-domain-robust-50k.md")


def source_balanced_weights(rows):
    """
    Equalize source/label groups before applying the low-FP cost bias.

    Without this, the largest corpora dominate the stacked model and it can
    learn source-specific decision surfaces that do not transfer well.
    """
    counts = Counter((r.get("source", "unknown"), int(bool(r["y"]))) for r in rows)
    groups = max(1, len(counts))
    n = max(1, len(rows))

    w = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        key = (row.get("source", "unknown"), int(bool(row["y"])))
        # Every source/label bucket contributes approximately equal total mass.
        w[i] = n / (groups * max(1, counts[key]))

        symbols = {name.upper() for name, _ in row["symbols"]}
        rscore = float(row["rscore"] or 0.0)
        if not row["y"]:
            # Keep a strong ham penalty, but less extreme than v24's 9x because
            # source balancing itself already protects minority ham domains.
            w[i] *= 7.0
            if rscore >= 2.0:
                w[i] *= 1.5
            if "BAYES_HAM" in symbols:
                w[i] *= 1.35
            if "MIME_GOOD" in symbols:
                w[i] *= 1.10
        elif rscore <= 3.5:
            # Focus on the spam Rspamd did not already make easy.
            w[i] *= 2.0

    return w


def forensic_weights(rows):
    counts = Counter((r.get("source", "unknown"), int(bool(r["y"]))) for r in rows)
    groups = max(1, len(counts))
    n = max(1, len(rows))
    w = np.empty(len(rows), dtype=np.float64)

    for i, row in enumerate(rows):
        key = (row.get("source", "unknown"), int(bool(row["y"])))
        w[i] = n / (groups * max(1, counts[key]))
        if not row["y"]:
            w[i] *= 2.2
        elif float(row["rscore"] or 0.0) <= 3.5:
            w[i] *= 1.35
    return w


def fit_domain_robust_text_experts(train_rows, val_rows, test_rows):
    """
    Three domain-robust text signals:
      1) globally source-balanced word SVM
      2) globally source-balanced char SVM
      3) median vote of one word-SVM per training source

    The third signal must be supported by several independent corpus-specific
    models, which suppresses patterns that are strong only in one corpus.
    """
    tr_docs, _ = base.tfidf_documents(train_rows)
    va_docs, _ = base.tfidf_documents(val_rows)
    te_docs, _ = base.tfidf_documents(test_rows)

    y = np.asarray([int(bool(r["y"])) for r in train_rows], dtype=np.int32)
    w = forensic_weights(train_rows)

    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        min_df=2,
        max_df=0.997,
        max_features=200000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[\w@.\-]{2,}\b",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        lowercase=True,
        sublinear_tf=True,
        min_df=2,
        max_features=180000,
        analyzer="char_wb",
        ngram_range=(3, 5),
        dtype=np.float32,
    )

    xw = word.fit_transform(tr_docs)
    xv_w = word.transform(va_docs)
    xe_w = word.transform(te_docs)

    xc = char.fit_transform(tr_docs)
    xv_c = char.transform(va_docs)
    xe_c = char.transform(te_docs)

    global_word = LinearSVC(C=1.10, random_state=SEED)
    global_word.fit(xw, y, sample_weight=w)
    pword_val = expit(np.asarray(global_word.decision_function(xv_w), dtype=np.float64))
    pword_test = expit(np.asarray(global_word.decision_function(xe_w), dtype=np.float64))

    global_char = LinearSVC(C=0.95, random_state=SEED + 17)
    global_char.fit(xc, y, sample_weight=w)
    pchar_val = expit(np.asarray(global_char.decision_function(xv_c), dtype=np.float64))
    pchar_test = expit(np.asarray(global_char.decision_function(xe_c), dtype=np.float64))

    source_models = []
    val_votes = []
    test_votes = []
    sources = sorted({r.get("source", "unknown") for r in train_rows})

    for k, source in enumerate(sources):
        idx = np.asarray(
            [i for i, row in enumerate(train_rows) if row.get("source", "unknown") == source],
            dtype=np.int64,
        )
        sy = y[idx]
        if len(np.unique(sy)) < 2:
            continue

        local_w = np.ones(len(idx), dtype=np.float64)
        ham = sy == 0
        spam = sy == 1
        # Balance classes inside every corpus, then bias mildly toward ham.
        local_w[ham] = len(idx) / (2.0 * max(1, int(ham.sum()))) * 2.0
        local_w[spam] = len(idx) / (2.0 * max(1, int(spam.sum())))

        clf = LinearSVC(C=0.90, random_state=SEED + 100 + k * 13)
        clf.fit(xw[idx], sy, sample_weight=local_w)
        pv = expit(np.asarray(clf.decision_function(xv_w), dtype=np.float64))
        pe = expit(np.asarray(clf.decision_function(xe_w), dtype=np.float64))
        val_votes.append(pv)
        test_votes.append(pe)
        source_models.append((source, clf))
        print(
            f"v32 source expert {source} train={len(idx)} "
            f"val_mean={float(pv.mean()):.4f} test_mean={float(pe.mean()):.4f}",
            flush=True,
        )

    if val_votes:
        # Median requires agreement across unrelated training corpora and is
        # deliberately more robust than a max/mean vote.
        pcommittee_val = np.median(np.vstack(val_votes), axis=0)
        pcommittee_test = np.median(np.vstack(test_votes), axis=0)
    else:
        pcommittee_val = pword_val.copy()
        pcommittee_test = pword_test.copy()

    print(
        "v32 robust text means",
        f"word={float(pword_val.mean()):.4f}/{float(pword_test.mean()):.4f}",
        f"char={float(pchar_val.mean()):.4f}/{float(pchar_test.mean()):.4f}",
        f"committee={float(pcommittee_val.mean()):.4f}/{float(pcommittee_test.mean()):.4f}",
        flush=True,
    )

    return {
        "models": [
            ("source-balanced-word", global_word),
            ("source-balanced-char", global_char),
            ("source-committee", source_models),
        ],
        "vectorizers": {"word": word, "char": char},
        # Reuse the existing three forensic feature slots. The old subject-only
        # score is replaced by the more transferable source committee.
        "val": [pword_val, pchar_val, pcommittee_val],
        "test": [pword_test, pchar_test, pcommittee_test],
    }


def write_report():
    src_json = Path("reports/v24-crossfit.json")
    src_md = Path("reports/v24-crossfit.md")

    if src_json.exists():
        obj = json.loads(src_json.read_text())
        obj["version"] = "v32-domain-robust-lowfp"
        obj["dataset"]["training"] = v28.EXPECTED_TRAIN
        obj["dataset"]["validation"] = v28.EXPECTED_VAL
        obj["dataset"]["lockedTest"] = v28.EXPECTED_TEST
        obj["dataset"]["trainingSources"] = [
            "TREC-07", "Enron", "TREC-06", "Assassin", "Ling"
        ]
        obj["dataset"]["testSources"] = ["TREC-05", "CEAS-08"]
        obj["method"]["sourceArtifactSanitization"] = True
        obj["method"]["sourceBalancedMetaTraining"] = True
        obj["method"]["domainRobustTextExperts"] = (
            "source-balanced global word SVM + source-balanced global char SVM + "
            "median committee of one word SVM per training corpus"
        )
        obj["method"]["gate"] = "v30 hard low-FP two-tier gate"
        obj["method"]["testLabelsUsedForTrainingOrThresholds"] = False
        obj["method"]["individualTestErrorsInspected"] = False
        obj["warning"] = (
            "The v28/v30/v31 test source pair is reused for engineering comparison, "
            "so this is not a pristine final lockbox. Validate the chosen design on a "
            "new untouched corpus before making a final generalization claim."
        )
        REPORT_JSON.write_text(json.dumps(obj, indent=2))

    if src_md.exists():
        txt = src_md.read_text()
        txt = txt.replace(
            "# MailGuard v24 hard-ham-veto benchmark",
            "# MailGuard v32 domain-robust low-FP 50k benchmark",
        )
        txt = txt.replace("v24 ", "v32 ")
        txt += (
            "\nV32 changes the model, not only thresholds: training weights are "
            "balanced by source+label, global word/char SVMs use those weights, and "
            "the third forensic signal is a median committee of source-specific SVMs. "
            "V31 source-artifact sanitization and the v30 low-FP gate are retained.\n"
        )
        REPORT_MD.write_text(txt)


def main():
    # Keep v31's sanitized EML. Reuse cache if available, but make fresh runners
    # render the same source-neutral representation.
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

    v18.enron.prepare_enron = lambda: data["train"] + data["val"] + data["test"]
    v18.split_enron = lambda _rows: (data["train"], data["val"], data["test"])
    v3.build_splits = lambda _groups: ([], [], [])

    base.fit_forensic_text_experts = fit_domain_robust_text_experts
    base.sample_weights = source_balanced_weights
    base.select_dual_guard = v30.select_balanced_guard
    base.apply_dual_guard = v30.apply_balanced_guard
    base.chronological_folds = v28.source_heldout_folds
    base.MODELS = Path("models/v32")
    base.SEED = SEED

    base.main()
    write_report()


if __name__ == "__main__":
    main()
