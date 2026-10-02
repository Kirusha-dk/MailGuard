#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import benchmark_v18_overnight as v18
import benchmark_v25_soft_veto as v25
import benchmark_v27_multidomain as md
import benchmark_v28_crosssource as v28

base = v25.base
v3 = base.v3

SEED = 20261011
REPORT_JSON = Path("reports/v30-balanced-lowfp-50k.json")
REPORT_MD = Path("reports/v30-balanced-lowfp-50k.md")


def profile(name):
    # Validation has 3,500 ham spread across five whole source families.
    # v28 used 6-11 validation FP and produced 389-659 FP on the unseen 25k ham.
    # v30 cuts those budgets sharply while still allowing a high-confidence
    # rescue path so recall does not collapse like v29.
    if name == "ultra-safe":
        return {"fp": 0, "fold_fp": 0, "target": 0.70, "worst": 0.35}
    if name == "safe":
        return {"fp": 2, "fold_fp": 1, "target": 0.90, "worst": 0.55}
    if name == "target-93":
        return {"fp": 4, "fold_fp": 1, "target": 0.93, "worst": 0.60}
    if name == "target-95":
        return {"fp": 8, "fold_fp": 2, "target": 0.95, "worst": 0.65}
    return {"fp": 4, "fold_fp": 1, "target": 0.90, "worst": 0.55}


def select_balanced_guard(
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
    cfg = profile(name)
    y = np.asarray([bool(r["y"]) for r in rows], dtype=bool)
    fold_values = sorted(set(fold_id.tolist()))

    primary_scores = np.asarray([
        .40, .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .93, .95
    ])
    primary_ham = np.asarray([
        .015, .025, .04, .055, .075, .10, .13, .16
    ])
    primary_agree = (2, 3, 4, 5)

    override_scores = np.asarray([
        .72, .78, .84, .88, .92, .95, .97, .985, .995
    ])
    override_ham = np.asarray([
        .04, .07, .10, .16, .25, .40
    ])
    override_agree = (4, 5, 6, 7, 8)

    primary = []
    for agree in primary_agree:
        for hmax in primary_ham:
            eligible = (agreement >= agree) & (ham_risk <= hmax)
            for threshold in primary_scores:
                m = eligible & (spam_score >= threshold)
                if int(((~y) & m).sum()) <= cfg["fp"]:
                    primary.append((threshold, hmax, agree, m))

    # Include an empty primary branch so a pure high-confidence rule is always possible.
    primary.append((1.000001, 0.0, 99, np.zeros(len(rows), dtype=bool)))

    override = []
    for agree in override_agree:
        for hmax in override_ham:
            eligible = (agreement >= agree) & (ham_risk <= hmax)
            for threshold in override_scores:
                m = eligible & (spam_score >= threshold)
                if int(((~y) & m).sum()) <= cfg["fp"]:
                    override.append((threshold, hmax, agree, m))

    override.append((1.000001, 0.0, 99, np.zeros(len(rows), dtype=bool)))

    best_feasible = None
    best_fallback = None

    for ps, ph, pa, pm in primary:
        for os, oh, oa, om in override:
            pred = pm | om
            fp_total = int(((~y) & pred).sum())
            if fp_total > cfg["fp"]:
                continue

            stats = []
            recalls = []
            stable = True
            for fid in fold_values:
                fm = fold_id == fid
                fy = y[fm]
                fpred = pred[fm]
                fp = int(((~fy) & fpred).sum())
                if fp > cfg["fold_fp"]:
                    stable = False
                    break
                spam = int(fy.sum())
                tp = int((fy & fpred).sum())
                recall = tp / max(1, spam)
                recalls.append(recall)
                stats.append({
                    "fold": int(fid),
                    "spam": spam,
                    "tp": tp,
                    "fp": fp,
                    "recall": recall,
                })

            if not stable:
                continue

            tp_total = int((y & pred).sum())
            recall = tp_total / max(1, int(y.sum()))
            worst = min(recalls) if recalls else 0.0
            mean = float(np.mean(recalls)) if recalls else 0.0

            point = {
                "name": name,
                "primaryThreshold": float(ps),
                "primaryMaxHamRisk": float(ph),
                "primaryMinAgreement": int(pa),
                "overrideThreshold": float(os),
                "overrideMaxHamRisk": float(oh),
                "overrideMinAgreement": int(oa),
                "spamDetected": tp_total,
                "falsePositives": fp_total,
                "recall": recall,
                "worstFoldRecall": worst,
                "meanFoldRecall": mean,
                "foldStats": stats,
                "validationFpBudget": cfg["fp"],
                "validationFoldFpBudget": cfg["fold_fp"],
            }

            # Main engineering objective: highest recall inside a much tighter FP
            # budget.  Source-held worst-fold recall is the first tie-breaker.
            fallback_key = (
                tp_total,
                round(worst, 8),
                -fp_total,
                pa + oa,
                ps + os,
            )
            if best_fallback is None or fallback_key > best_fallback[0]:
                best_fallback = (fallback_key, point)

            if recall >= cfg["target"] and worst >= cfg["worst"]:
                feasible_key = (
                    -fp_total,
                    round(worst, 8),
                    recall,
                    pa + oa,
                    ps + os,
                )
                if best_feasible is None or feasible_key > best_feasible[0]:
                    best_feasible = (feasible_key, point)

    if best_feasible is not None:
        return best_feasible[1]
    if best_fallback is not None:
        return best_fallback[1]

    return {
        "name": name,
        "primaryThreshold": 1.000001,
        "primaryMaxHamRisk": 0.0,
        "primaryMinAgreement": 99,
        "overrideThreshold": 1.000001,
        "overrideMaxHamRisk": 0.0,
        "overrideMinAgreement": 99,
        "spamDetected": 0,
        "falsePositives": 0,
        "recall": 0.0,
        "worstFoldRecall": 0.0,
        "meanFoldRecall": 0.0,
        "foldStats": [],
        "validationFpBudget": cfg["fp"],
        "validationFoldFpBudget": cfg["fold_fp"],
    }


def apply_balanced_guard(base_rspamd, spam_score, ham_risk, agreement, gate):
    primary = (
        (spam_score >= gate["primaryThreshold"])
        & (ham_risk <= gate["primaryMaxHamRisk"])
        & (agreement >= gate["primaryMinAgreement"])
    )
    override = (
        (spam_score >= gate["overrideThreshold"])
        & (ham_risk <= gate["overrideMaxHamRisk"])
        & (agreement >= gate["overrideMinAgreement"])
    )
    return base_rspamd | primary | override


def write_report():
    src_json = Path("reports/v24-crossfit.json")
    src_md = Path("reports/v24-crossfit.md")

    if src_json.exists():
        obj = json.loads(src_json.read_text())
        obj["version"] = "v30-balanced-lowfp-crosssource"
        obj["dataset"]["training"] = v28.EXPECTED_TRAIN
        obj["dataset"]["validation"] = v28.EXPECTED_VAL
        obj["dataset"]["lockedTest"] = v28.EXPECTED_TEST
        obj["dataset"]["trainingSources"] = [
            "TREC-07", "Enron", "TREC-06", "Assassin", "Ling"
        ]
        obj["dataset"]["testSources"] = ["TREC-05", "CEAS-08"]
        obj["method"]["metaValidation"] = (
            "5-fold leave-one-source-family-out validation"
        )
        obj["method"]["gate"] = (
            "two-tier high-precision gate with hard total and per-source FP budgets"
        )
        obj["method"]["validationFpBudgets"] = {
            "ultraSafe": 0,
            "safe": 2,
            "target93": 4,
            "target95": 8,
        }
        obj["method"]["testLabelsUsedForTrainingOrThresholds"] = False
        obj["method"]["individualTestErrorsInspected"] = False
        obj["warning"] = (
            "This reuses the v28 external source pair for engineering comparison, so "
            "it is no longer a pristine final lockbox. A new untouched corpus is still "
            "required before a final generalization claim."
        )
        REPORT_JSON.write_text(json.dumps(obj, indent=2))

    if src_md.exists():
        txt = src_md.read_text()
        txt = txt.replace(
            "# MailGuard v24 hard-ham-veto benchmark",
            "# MailGuard v30 balanced low-FP cross-source 50k benchmark",
        )
        txt = txt.replace("v24 ", "v30 ")
        txt += (
            "\nV30 keeps the v28 cross-source training design but replaces the gate "
            "with much tighter FP budgets (0/2/4/8 total validation FP) and a "
            "high-confidence override branch. The aim is to reduce v28 false positives "
            "several-fold without repeating v29's recall collapse. Test labels are not "
            "used to fit models or choose thresholds.\n"
        )
        REPORT_MD.write_text(txt)


def main():
    # Reuse the exact v28 source-separated data design and cached rendered split.
    md.CACHE = Path(".cache/v28-crosssource")
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

    base.select_dual_guard = select_balanced_guard
    base.apply_dual_guard = apply_balanced_guard
    base.chronological_folds = v28.source_heldout_folds
    base.MODELS = Path("models/v30")
    base.SEED = SEED

    base.main()
    write_report()


if __name__ == "__main__":
    main()
