#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import benchmark_v18_overnight as v18
import benchmark_v25_soft_veto as v25
import benchmark_v27_multidomain as md

base = v25.base
v3 = base.v3

SEED = 20261010

TRAIN_PLAN = {
    ("TREC-07", 0): 10000, ("TREC-07", 1): 12000,
    ("Enron", 0): 8000, ("Enron", 1): 8000,
    ("Assassin", 0): 2500, ("Assassin", 1): 1000,
    ("Ling", 0): 1500, ("Ling", 1): 300,
}

# Entire validation source is unseen by every base classifier.
VAL_PLAN = {
    ("TREC-06", 0): 8000, ("TREC-06", 1): 3000,
}

# Final lockbox is again completely unseen by training and validation.
TEST_PLAN = {
    ("TREC-05", 0): 13000, ("TREC-05", 1): 13000,
    ("CEAS-08", 0): 12000, ("CEAS-08", 1): 12000,
}

EXPECTED_TRAIN = 51300
EXPECTED_VAL = 11000
EXPECTED_TEST = 50000

REPORT_JSON = Path("reports/v29-lowfp-50k.json")
REPORT_MD = Path("reports/v29-lowfp-50k.md")


def _profile(name):
    # Goal scaled from ~50 FP / 500k to the current balanced 50k benchmark.
    # With 25k ham here, ~5 FP corresponds to the same 0.02% ham-side rate.
    if name == "ultra-safe":
        return 0
    if name == "safe":
        return 1
    if name == "target-93":
        return 1
    if name == "target-95":
        return 2
    return 1


def select_precision_guard(
    rows,
    spam_score,
    ham_risk,
    agreement,
    fold_id,
    target_recall,
    max_total_fp,
    max_fold_fp,
    name,
):
    y = np.asarray([bool(r["y"]) for r in rows], dtype=bool)
    fp_budget = _profile(name)

    score_candidates = np.unique(np.concatenate([
        np.asarray([
            .40,.45,.50,.55,.60,.65,.70,.75,.80,.84,.88,.90,.92,.94,
            .95,.96,.97,.98,.985,.99,.995,.999,1.000001
        ]),
        np.quantile(spam_score, np.linspace(.60, .9995, 140)),
    ]))
    ham_candidates = np.unique(np.concatenate([
        np.asarray([
            .005,.01,.015,.02,.025,.03,.04,.05,.06,.075,.09,.11,.13,.16,.20
        ]),
        np.quantile(ham_risk, np.linspace(.005, .35, 45)),
    ]))

    best = None
    fold_values = sorted(set(fold_id.tolist()))

    for min_agree in (2, 3, 4, 5, 6, 7, 8):
        for max_ham in ham_candidates:
            eligible = (agreement >= min_agree) & (ham_risk <= max_ham)
            if not np.any(eligible):
                continue

            for threshold in score_candidates:
                pred = eligible & (spam_score >= threshold)
                fp_total = int(((~y) & pred).sum())
                if fp_total > fp_budget:
                    continue

                fold_stats = []
                recalls = []
                stable = True
                for fid in fold_values:
                    fm = fold_id == fid
                    fy = y[fm]
                    fpred = pred[fm]
                    fp = int(((~fy) & fpred).sum())
                    if fp > 1:
                        stable = False
                        break
                    spam = int(fy.sum())
                    tp = int((fy & fpred).sum())
                    fr = tp / max(1, spam)
                    recalls.append(fr)
                    fold_stats.append({
                        "fold": int(fid),
                        "spam": spam,
                        "tp": tp,
                        "fp": fp,
                        "recall": fr,
                    })

                if not stable:
                    continue

                tp_total = int((y & pred).sum())
                recall = tp_total / max(1, int(y.sum()))
                worst = min(recalls) if recalls else 0.0
                point = {
                    "name": name,
                    "threshold": float(threshold),
                    "maxHamRisk": float(max_ham),
                    "minAgreement": int(min_agree),
                    "spamDetected": tp_total,
                    "falsePositives": fp_total,
                    "recall": recall,
                    "worstFoldRecall": worst,
                    "meanFoldRecall": float(np.mean(recalls)) if recalls else 0.0,
                    "foldStats": fold_stats,
                    "fpBudget": fp_budget,
                }

                # Precision first: never exceed the hard budget. Inside it,
                # maximize recall, then prefer more stable / stricter rules.
                key = (
                    tp_total,
                    round(worst, 8),
                    -fp_total,
                    min_agree,
                    threshold,
                    -max_ham,
                )
                if best is None or key > best[0]:
                    best = (key, point)

    if best is not None:
        return best[1]

    return {
        "name": name,
        "threshold": 1.000001,
        "maxHamRisk": 0.0,
        "minAgreement": 10,
        "spamDetected": 0,
        "falsePositives": 0,
        "recall": 0.0,
        "worstFoldRecall": 0.0,
        "meanFoldRecall": 0.0,
        "foldStats": [],
        "fpBudget": fp_budget,
    }


def apply_precision_guard(base_rspamd, spam_score, ham_risk, agreement, gate):
    return base_rspamd | (
        (spam_score >= gate["threshold"])
        & (ham_risk <= gate["maxHamRisk"])
        & (agreement >= gate["minAgreement"])
    )


def rewrite_report():
    src_json = Path("reports/v24-crossfit.json")
    src_md = Path("reports/v24-crossfit.md")

    if src_json.exists():
        obj = json.loads(src_json.read_text())
        obj["version"] = "v29-lowfp-source-separated"
        obj["dataset"]["training"] = EXPECTED_TRAIN
        obj["dataset"]["validation"] = EXPECTED_VAL
        obj["dataset"]["lockedTest"] = EXPECTED_TEST
        obj["dataset"]["trainingSources"] = ["TREC-07", "Enron", "Assassin", "Ling"]
        obj["dataset"]["validationSources"] = ["TREC-06"]
        obj["dataset"]["testSources"] = ["TREC-05", "CEAS-08"]
        obj["dataset"]["split"] = (
            "base-model train sources, calibration source, and final test sources are "
            "fully source-separated after global normalized exact dedupe"
        )
        obj["method"]["lowFpObjective"] = (
            "hard validation FP budget scaled from the project target of about 50 FP / 500k"
        )
        obj["method"]["softOverrideRemoved"] = True
        obj["method"]["testLabelsUsedForTrainingOrThresholds"] = False
        obj["method"]["individualTestErrorsInspected"] = False
        obj["warning"] = (
            "Historical public corpora with normalized fields. The final 50k remains "
            "a cross-corpus robustness test, not production accuracy."
        )
        REPORT_JSON.write_text(json.dumps(obj, indent=2))

    if src_md.exists():
        txt = src_md.read_text()
        txt = txt.replace(
            "# MailGuard v24 hard-ham-veto benchmark",
            "# MailGuard v29 low-FP source-separated 50k benchmark",
        )
        txt = txt.replace("v24 ", "v29 ")
        txt += (
            "\nFixes in v29: base classifiers never see the validation source; the old "
            "soft override path is removed; gate selection uses a hard FP budget before "
            "optimizing recall. Training sources: TREC-07, Enron, Assassin, Ling. "
            "Calibration: TREC-06 only. Final test: unseen TREC-05 + CEAS-08.\n"
        )
        REPORT_MD.write_text(txt)


def main():
    md.CACHE = Path(".cache/v29-lowfp")
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
        "v29 source-separated",
        "train", len(data["train"]),
        "val", len(data["val"]),
        "test", len(data["test"]),
        flush=True,
    )

    v18.enron.prepare_enron = lambda: data["train"] + data["val"] + data["test"]
    v18.split_enron = lambda _rows: (data["train"], data["val"], data["test"])
    v3.build_splits = lambda _groups: ([], [], [])

    base.select_dual_guard = select_precision_guard
    base.apply_dual_guard = apply_precision_guard
    base.MODELS = Path("models/v29")
    base.SEED = SEED

    base.main()
    rewrite_report()


if __name__ == "__main__":
    main()
