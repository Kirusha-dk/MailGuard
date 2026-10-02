#!/usr/bin/env python3
from pathlib import Path
import numpy as np

import benchmark_v24_ham_veto as base

ORIGINAL_APPLY = base.apply_dual_guard


def _profile(name, requested_target, max_total_fp, max_fold_fp):
    if name == "ultra-safe":
        return 0.90, 0.80, min(max_total_fp + 2, 6), min(max_fold_fp + 1, 3)
    if name == "safe":
        return 0.955, 0.90, max(max_total_fp, 8), max(max_fold_fp, 3)
    if name == "target-93":
        return 0.975, 0.945, max(max_total_fp, 16), max(max_fold_fp, 5)
    if name == "target-95":
        return 0.985, 0.955, max(max_total_fp, 22), max(max_fold_fp, 7)
    return requested_target, 0.0, max_total_fp, max_fold_fp


def select_two_tier_guard(
    rows,
    spam_score,
    ham_risk,
    agreement,
    fold_id,
    target_recall,
    min_worst_recall,
    max_total_fp,
    max_fold_fp,
    name,
):
    y = np.asarray([bool(r["y"]) for r in rows], dtype=bool)

    primary_scores = np.asarray([.40,.45,.50,.55,.60,.65,.70,.75])
    primary_ham = np.asarray([.025,.04,.055,.075,.10,.13])
    primary_agree = (1, 2, 3)

    override_scores = np.asarray([.58,.64,.70,.76,.82,.88,.93,.97])
    override_ham = np.asarray([.08,.12,.16,.22,.30,.45,.70,1.0])
    override_agree = (2, 3, 4, 5)

    primary = []
    for a in primary_agree:
        for h in primary_ham:
            for s in primary_scores:
                m = (spam_score >= s) & (ham_risk <= h) & (agreement >= a)
                if int(((~y) & m).sum()) <= max_total_fp:
                    primary.append((s, h, a, m))

    overrides = []
    for a in override_agree:
        for h in override_ham:
            for s in override_scores:
                m = (spam_score >= s) & (ham_risk <= h) & (agreement >= a)
                if int(((~y) & m).sum()) <= max_total_fp:
                    overrides.append((s, h, a, m))

    feasible = []
    fallback = []
    fold_values = sorted(set(fold_id.tolist()))

    for ps, ph, pa, pm in primary:
        for oscore, oh, oa, om in overrides:
            pred = pm | om
            fp_total = int(((~y) & pred).sum())
            if fp_total > max_total_fp:
                continue

            tp_total = int((y & pred).sum())
            recall = tp_total / max(1, int(y.sum()))
            stats = []
            recalls = []
            stable = True

            for fid in fold_values:
                fm = fold_id == fid
                fy = y[fm]
                fpred = pred[fm]
                fp = int(((~fy) & fpred).sum())
                if fp > max_fold_fp:
                    stable = False
                    break
                tp = int((fy & fpred).sum())
                spam = int(fy.sum())
                fr = tp / max(1, spam)
                recalls.append(fr)
                stats.append({
                    "fold": int(fid),
                    "spam": spam,
                    "tp": tp,
                    "fp": fp,
                    "recall": fr,
                })

            if not stable:
                continue

            worst = min(recalls) if recalls else 0.0
            point = {
                "name": name,
                "primaryThreshold": float(ps),
                "primaryMaxHamRisk": float(ph),
                "primaryMinAgreement": int(pa),
                "overrideThreshold": float(oscore),
                "overrideMaxHamRisk": float(oh),
                "overrideMinAgreement": int(oa),
                "spamDetected": tp_total,
                "falsePositives": fp_total,
                "recall": recall,
                "worstFoldRecall": worst,
                "meanFoldRecall": float(np.mean(recalls)) if recalls else 0.0,
                "foldStats": stats,
            }
            fallback.append(point)
            if recall >= target_recall and worst >= min_worst_recall:
                feasible.append(point)

    if feasible:
        return min(
            feasible,
            key=lambda p: (
                p["falsePositives"],
                -p["worstFoldRecall"],
                -p["recall"],
                p["overrideMaxHamRisk"],
                -p["overrideThreshold"],
            ),
        )

    if fallback:
        return max(
            fallback,
            key=lambda p: (
                p["worstFoldRecall"],
                p["recall"],
                -p["falsePositives"],
            ),
        )

    return {
        "name": name,
        "primaryThreshold": 1.000001,
        "primaryMaxHamRisk": 0.0,
        "primaryMinAgreement": 10,
        "overrideThreshold": 1.000001,
        "overrideMaxHamRisk": 0.0,
        "overrideMinAgreement": 10,
        "spamDetected": 0,
        "falsePositives": 0,
        "recall": 0.0,
        "worstFoldRecall": 0.0,
        "meanFoldRecall": 0.0,
        "foldStats": [],
    }


def select_dual_guard(
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
    target, worst, total_fp, fold_fp = _profile(
        name, target_recall, max_total_fp, max_fold_fp
    )
    return select_two_tier_guard(
        rows,
        spam_score,
        ham_risk,
        agreement,
        fold_id,
        target,
        worst,
        total_fp,
        fold_fp,
        name,
    )


def apply_dual_guard(base_rspamd, spam_score, ham_risk, agreement, gate):
    if "primaryThreshold" not in gate:
        return ORIGINAL_APPLY(base_rspamd, spam_score, ham_risk, agreement, gate)

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


def rewrite_report(path_in, path_out):
    p = Path(path_in)
    if not p.exists():
        return
    txt = p.read_text()
    txt = txt.replace("v24", "v25")
    txt = txt.replace("V24", "V25")
    txt = txt.replace("hard-ham-veto", "soft-two-tier-veto")
    Path(path_out).write_text(txt)


def main():
    base.select_dual_guard = select_dual_guard
    base.apply_dual_guard = apply_dual_guard
    base.MODELS = Path("models/v25")
    # Keep the same seed as v24 so the experiment isolates the veto change.
    base.SEED = 20261006
    base.main()
    rewrite_report("reports/v24-crossfit.json", "reports/v25-crossfit.json")
    rewrite_report("reports/v24-crossfit.md", "reports/v25-crossfit.md")


if __name__ == "__main__":
    main()
