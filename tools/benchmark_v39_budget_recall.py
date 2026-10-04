#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import benchmark_v18_overnight as v18
import benchmark_v25_soft_veto as v25
import benchmark_v28_crosssource as v28
import benchmark_v30_balanced_lowfp as v30
import benchmark_v33_overnight_neural as v33
import benchmark_v34_hybrid_rescue as v34
import benchmark_v36_bugfix_hardmine as v36
import benchmark_v38_corrected_recall as v38

base = v25.base
v3 = base.v3

SEED = 20261019
OUT_MODELS = Path("models/v39")
REPORT_JSON = Path("reports/v39-budget-recall-50k.json")
REPORT_MD = Path("reports/v39-budget-recall-50k.md")


def transfer_meta_score(models, x, raw_probs):
    """
    Cross-domain recall score.

    V38's meta score can become conservative on unseen source families because it
    leans on the fitted linear/tree pair plus median consensus. Keep that core,
    but add a rescue rank from the upper quartile / strongest experts. The final
    ham veto and source-held-out gate still control false positives.
    """
    linear, tree = models
    pl = np.clip(linear.predict_proba(x)[:, 1], 1e-8, 1.0)
    pt = np.clip(tree.predict_proba(x)[:, 1], 1e-8, 1.0)

    pair = np.sqrt(pl * pt)
    median = np.median(raw_probs, axis=0)
    q75 = np.quantile(raw_probs, 0.75, axis=0)
    top3 = np.mean(np.sort(raw_probs, axis=0)[-3:, :], axis=0)

    agree = np.sum(raw_probs >= 0.72, axis=0).astype(np.int32)

    core = pair * (0.68 + 0.18 * median + 0.14 * q75)
    expert_rescue = np.sqrt(np.clip(q75 * top3, 1e-10, 1.0))
    rescue_scale = 0.80 + 0.20 * np.minimum(1.0, agree / 6.0)
    rescue = 0.90 * expert_rescue * rescue_scale
    rescue *= 0.88 + 0.12 * np.sqrt(pair)
    score = np.maximum(core, rescue)

    return np.clip(score, 0.0, 1.0), pl, pt, agree


def recall_budget_profile(name):
    if name == "ultra-safe":
        return {"fp": 0, "fold_fp": 0, "worst": 0.35}
    if name == "safe":
        return {"fp": 3, "fold_fp": 1, "worst": 0.50}
    if name == "target-93":
        return {"fp": 8, "fold_fp": 3, "worst": 0.58}
    if name == "target-95":
        return {"fp": 16, "fold_fp": 4, "worst": 0.62}
    return {"fp": 8, "fold_fp": 3, "worst": 0.58}


def select_budget_recall_guard(
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
    """
    Select a two-tier gate by recall under a hard FP budget.

    V38 inherited v30's "once target recall is reached, minimize FP first"
    objective. On target-95 it selected 95.45% validation recall with only 6 FP
    even though 16 FP were allowed. That leaves recall headroom unused.

    V39 keeps the exact hard total/per-source FP caps, but ranks feasible gates by:
      1) total spam caught
      2) worst held-out-source recall
      3) mean held-out-source recall
      4) fewer FP
    """
    cfg = recall_budget_profile(name)
    y = np.asarray([bool(r["y"]) for r in rows], dtype=bool)
    fold_values = sorted(set(fold_id.tolist()))

    primary_scores = np.asarray([
        .24, .28, .32, .36, .40, .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .93, .95
    ])
    primary_ham = np.asarray([
        .012, .020, .030, .040, .055, .075, .10, .13, .16, .20
    ])
    primary_agree = (2, 3, 4, 5, 6)

    override_scores = np.asarray([
        .50, .56, .62, .68, .72, .78, .84, .88, .92, .95, .97, .985, .995
    ])
    override_ham = np.asarray([
        .025, .04, .07, .10, .16, .25, .40
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

    best_stable = None
    best_any = None

    for ps, ph, pa, pm in primary:
        for os, oh, oa, om in override:
            pred = pm | om
            fp_total = int(((~y) & pred).sum())
            if fp_total > cfg["fp"]:
                continue

            stats = []
            recalls = []
            per_source_ok = True

            for fid in fold_values:
                fm = fold_id == fid
                fy = y[fm]
                fpred = pred[fm]
                fp = int(((~fy) & fpred).sum())
                if fp > cfg["fold_fp"]:
                    per_source_ok = False
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

            if not per_source_ok:
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
                "selectionObjective": "max recall under hard total/per-source FP budget",
            }

            key = (
                tp_total,
                round(worst, 8),
                round(mean, 8),
                -fp_total,
                pa + oa,
                ps + os,
            )
            if best_any is None or key > best_any[0]:
                best_any = (key, point)

            if worst >= cfg["worst"]:
                if best_stable is None or key > best_stable[0]:
                    best_stable = (key, point)

    if best_stable is not None:
        return best_stable[1]
    if best_any is not None:
        return best_any[1]

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
        "selectionObjective": "max recall under hard total/per-source FP budget",
    }


def write_report():
    src_json = Path("reports/v24-crossfit.json")
    src_md = Path("reports/v24-crossfit.md")

    if src_json.exists():
        obj = json.loads(src_json.read_text())
        obj["version"] = "v39-budget-recall-crossdomain"
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
        obj["method"]["neural"] = "v33 larger two-tower ensemble; corrected v36 data"
        obj["method"]["forensicHybrid"] = "v34 78/22 subject + source committee"
        obj["method"]["metaHamPenalty"] = 6.0
        obj["method"]["metaScore"] = (
            "v38 pair/median core plus upper-quartile/top-3 cross-domain rescue"
        )
        obj["method"]["agreementThresholdForScore"] = 0.72
        obj["method"]["gateSelection"] = (
            "maximize recall under unchanged hard total/per-source validation FP budgets"
        )
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
            "# MailGuard v39 budget-first cross-domain recall 50k benchmark",
        )
        txt = txt.replace("v24 ", "v39 ")
        txt = txt.replace(
            "The 10k test labels are not used",
            "The 50k test labels are not used",
        )
        txt += (
            "\nV39 keeps v38's corrected-data path, ham protection and exact FP "
            "budgets. The main fix is gate selection: after reaching the named recall "
            "target, v38 minimized validation FP and left most of the target-95 FP "
            "budget unused. V39 instead maximizes caught spam under the same hard total "
            "and per-source FP caps. It also adds a cross-domain rescue component to the "
            "meta score when several independent experts agree on spam even if the fitted "
            "meta pair is conservative on a new source family.\n"
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

    v18.base_weights = v33.mild_source_weights
    v18.DeepMailNet = v33.OvernightMailNet
    v18.train_model = v33.overnight_train_model
    v18.MODELS = OUT_MODELS
    v33.OUT_MODELS = OUT_MODELS

    base.fit_forensic_text_experts = v34.fit_hybrid_forensic
    base.sample_weights = v38.recall_sample_weights
    base.fit_ham_veto = v38.fit_recall_ham_veto
    base.meta_score = transfer_meta_score
    base.select_dual_guard = select_budget_recall_guard
    base.apply_dual_guard = v30.apply_balanced_guard
    base.chronological_folds = v28.source_heldout_folds
    base.MODELS = OUT_MODELS
    base.SEED = SEED

    base.main()
    write_report()


if __name__ == "__main__":
    main()
