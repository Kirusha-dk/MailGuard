#!/usr/bin/env python3
import json
import math
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

import benchmark_improved as b
import benchmark_v3 as v3
import benchmark_v5 as v5
import benchmark_v15_hard_mining as v15

REPORTS = Path("reports")
SEED = 20261001
TARGET_TEST = 1500


def small_test(rows, target=TARGET_TEST):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    spam_idx = np.where(y == 1)[0].tolist()
    ham_idx = np.where(y == 0)[0].tolist()

    rnd = random.Random(SEED + 1700)
    rnd.shuffle(spam_idx)
    rnd.shuffle(ham_idx)

    spam_target = int(round(target * len(spam_idx) / len(rows)))
    spam_target = min(spam_target, len(spam_idx))
    ham_target = min(target - spam_target, len(ham_idx))

    chosen = spam_idx[:spam_target] + ham_idx[:ham_target]

    # Fill any rounding/capacity gap deterministically.
    if len(chosen) < target:
        used = set(chosen)
        remaining = [i for i in range(len(rows)) if i not in used]
        rnd.shuffle(remaining)
        chosen.extend(remaining[: target - len(chosen)])

    chosen.sort()
    return [rows[i] for i in chosen]


def validation_halves(rows):
    a = np.zeros(len(rows), dtype=bool)
    bmask = np.zeros(len(rows), dtype=bool)
    buckets = {}

    for i, row in enumerate(rows):
        buckets.setdefault(row["source"], []).append(i)

    for idxs in buckets.values():
        for j, idx in enumerate(idxs):
            (a if j % 2 == 0 else bmask)[idx] = True

    return a, bmask


def has_symbol(row, name):
    name = name.upper()
    return any(sym.upper() == name for sym, _ in row["symbols"])


def meta_matrix(rows, probs, feat, pchar):
    ptext, pctx, pspec_text, pspec_ctx, perror_text, perror_ctx, pham = probs

    values = []
    for i, row in enumerate(rows):
        required = row["required"] or 0.0
        ratio = row["rscore"] / required if required else 0.0
        pos_symbols = sum(1 for _, score in row["symbols"] if score > 0)
        neg_symbols = sum(1 for _, score in row["symbols"] if score < 0)

        values.append([
            ptext[i],
            pctx[i],
            pspec_text[i],
            pspec_ctx[i],
            perror_text[i],
            perror_ctx[i],
            1.0 - pham[i],
            feat["score"][i],
            feat["consensus"][i],
            feat["median"][i],
            feat["specialist"][i],
            feat["errorExpert"][i],
            feat["agreement"][i] / 7.0,
            pchar[i],
            max(-3.0, min(3.0, row["rscore"] / 10.0)),
            max(-3.0, min(3.0, ratio)),
            math.log1p(len(row["raw"])) / 12.0,
            min(pos_symbols, 40) / 40.0,
            min(neg_symbols, 40) / 40.0,
            float(has_symbol(row, "BAYES_HAM")),
            float(has_symbol(row, "MIME_GOOD")),
            float(has_symbol(row, "LOCAL_OUTBOUND")),
        ])

    return np.asarray(values, dtype=np.float64)


def fit_meta(x, rows):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    base = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)

    spam_w = base.copy()
    ham_w = base.copy()

    for i, row in enumerate(rows):
        if not row["y"]:
            spam_w[i] *= 5.0
            ham_w[i] *= 8.0
            if "hard_ham" in row["source"]:
                spam_w[i] *= 2.0
                ham_w[i] *= 2.5
            if has_symbol(row, "BAYES_HAM"):
                spam_w[i] *= 1.5
                ham_w[i] *= 1.8
        else:
            if row["rscore"] <= 3.0:
                spam_w[i] *= 2.0

    spam_model = LogisticRegression(
        C=0.45,
        max_iter=800,
        solver="lbfgs",
        random_state=SEED + 17,
    )
    spam_model.fit(x, y, sample_weight=spam_w)

    # Independent ham shield.
    ham_target = 1 - y
    ham_model = LogisticRegression(
        C=0.35,
        max_iter=800,
        solver="lbfgs",
        random_state=SEED + 19,
    )
    ham_model.fit(x, ham_target, sample_weight=ham_w)

    return spam_model, ham_model


def meta_scores(models, x, rows, pchar):
    spam_model, ham_model = models
    pspam = spam_model.predict_proba(x)[:, 1]
    pham = ham_model.predict_proba(x)[:, 1]

    score = np.sqrt(
        np.clip(pspam, 1e-8, 1.0)
        * np.clip(1.0 - pham, 1e-8, 1.0)
        * np.clip(pchar, 1e-8, 1.0)
    )

    # Error analysis showed BAYES_HAM on many false positives, so it becomes
    # a shield rather than a hard veto.
    bayes_ham = np.asarray(
        [has_symbol(row, "BAYES_HAM") for row in rows],
        dtype=bool,
    )
    score = score * np.where(bayes_ham, 0.55, 1.0)

    return pspam, pham, score, bayes_ham


def v15_fail_count(feat, gate):
    checks = np.vstack([
        feat["score"] >= gate["scoreThreshold"],
        feat["consensus"] >= gate["consensusThreshold"],
        feat["median"] >= gate["medianThreshold"],
        feat["specialist"] >= gate["specialistThreshold"],
        feat["errorExpert"] >= gate["errorExpertThreshold"],
        feat["ham"] <= gate["hamVetoThreshold"],
        feat["agreement"] >= gate["minAgreement"],
    ])
    return (~checks).sum(axis=0)


def stats(rows, pred):
    y = np.asarray([r["y"] for r in rows], dtype=bool)
    spam = int(y.sum())
    ham = int((~y).sum())
    tp = int((y & pred).sum())
    fp = int(((~y) & pred).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, spam)

    return {
        "spamTotal": spam,
        "hamTotal": ham,
        "spamDetected": tp,
        "falseNegatives": spam - tp,
        "falsePositives": fp,
        "recall": recall,
        "precision": precision,
        "fpr": fp / max(1, ham),
    }


def choose_border_gate(rows, base_pred, effective, fail_count, pham_meta, safe):
    y = np.asarray([r["y"] for r in rows], dtype=bool)
    ham = ~y
    hard = np.asarray([
        (not r["y"]) and ("hard_ham" in r["source"])
        for r in rows
    ], dtype=bool)

    base_fp = int((ham & base_pred).sum())
    base_hard_fp = int((hard & base_pred).sum())

    residual = ~base_pred
    residual_scores = effective[residual]

    if len(residual_scores) == 0:
        return {
            "threshold": 1.000001,
            "maxFailCount": 0,
            "maxHamMeta": 0.0,
            "rescued": 0,
            "addedFp": 0,
        }

    candidates = np.unique(np.concatenate([
        np.asarray([0.30,0.40,0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.93,0.95,0.97,0.98,0.99,1.000001]),
        np.quantile(residual_scores, np.linspace(.40, 1.0, 50)),
    ]))

    best = None
    for max_fail in (1, 2):
        for max_ham in (0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.65):
            eligible = residual & (fail_count <= max_fail) & (pham_meta <= max_ham)
            if not np.any(eligible):
                continue

            for threshold in candidates:
                rescue = eligible & (effective >= threshold)
                pred = base_pred | rescue

                fp = int((ham & pred).sum())
                hard_fp = int((hard & pred).sum())

                if hard_fp > base_hard_fp:
                    continue
                if safe and fp > base_fp:
                    continue
                if (not safe) and fp > base_fp + 1:
                    continue

                tp = int((y & pred).sum())
                base_tp = int((y & base_pred).sum())
                point = {
                    "threshold": float(threshold),
                    "maxFailCount": int(max_fail),
                    "maxHamMeta": float(max_ham),
                    "rescued": tp - base_tp,
                    "addedFp": fp - base_fp,
                }

                key = (
                    point["rescued"],
                    -point["addedFp"],
                    point["threshold"],
                    -point["maxFailCount"],
                    -point["maxHamMeta"],
                )
                if best is None or key > best[0]:
                    best = (key, point)

    if best is None:
        return {
            "threshold": 1.000001,
            "maxFailCount": 0,
            "maxHamMeta": 0.0,
            "rescued": 0,
            "addedFp": 0,
        }

    return best[1]


def apply_border(base_pred, effective, fail_count, pham_meta, gate):
    rescue = (
        (~base_pred)
        & (fail_count <= gate["maxFailCount"])
        & (pham_meta <= gate["maxHamMeta"])
        & (effective >= gate["threshold"])
    )
    return base_pred | rescue


def main():
    REPORTS.mkdir(exist_ok=True)
    b.wait_rspamd()

    groups = b.prepare()
    train, val, full_test = v3.build_splits(groups)
    test = small_test(full_test, TARGET_TEST)

    print(
        "TRAIN", len(train),
        "VAL", len(val),
        "SMALL TEST", len(test),
        flush=True,
    )
    print("v17: analysis-driven near-miss rescue + ham shield", flush=True)

    b.reset_bayes()
    b.learn(train)

    print("[1/9] scan train", flush=True)
    tr = b.scan_many(train, "v17-train")
    print("[2/9] scan validation", flush=True)
    va = b.scan_many(val, "v17-val")
    print("[3/9] scan 1500-message test", flush=True)
    te = b.scan_many(test, "v17-test")

    base = b.base_metrics(te)

    xt, xv, xe, ctx_t, ctx_v, ctx_e = v3.matrices(tr, va, te)
    residual_idx = [i for i, row in enumerate(tr) if not row["rspam"]]
    residual_rows = [tr[i] for i in residual_idx]
    xr_text = xt[residual_idx]
    xr_ctx = ctx_t[residual_idx]

    print("[4/9] train v15 hard-example experts", flush=True)
    hardness, _ = v15.crossfit_hardness(xr_text, xr_ctx, residual_rows)

    base_text_models = v3.fit_ensemble(xr_text, residual_rows)
    base_ctx_models = v3.fit_ensemble(xr_ctx, residual_rows)
    ham_models = v5.fit_ham_ensemble(xr_ctx, residual_rows)
    spec_text_models = v15.fit_specialist_ensemble(xr_text, residual_rows, hardness, 2100)
    spec_ctx_models = v15.fit_specialist_ensemble(xr_ctx, residual_rows, hardness, 3100)
    error_text_models, error_ctx_models, _ = v15.fit_error_experts(
        xr_text, xr_ctx, residual_rows, hardness
    )

    # Lightweight independent char-only specialist for obfuscated/near-miss text.
    print("[5/9] train char-only specialist", flush=True)
    hard_idx = v15.hard_subset_indices(residual_rows, hardness)
    hard_rows = [residual_rows[i] for i in hard_idx]
    hard_y = np.asarray([r["y"] for r in hard_rows], dtype=np.int32)
    hard_w = v15.mining_weights(hard_rows, hardness[hard_idx])

    char_train = b.CHAR.transform([b.extract_document(r["raw"]) for r in residual_rows])
    char_val = b.CHAR.transform([b.extract_document(r["raw"]) for r in va])
    char_test = b.CHAR.transform([b.extract_document(r["raw"]) for r in te])

    char_models = [
        v15.fit_one(
            char_train[hard_idx],
            hard_y,
            hard_w,
            alpha,
            SEED + 17000 + j * 101,
        )
        for j, alpha in enumerate((8e-5, 2.5e-4))
    ]
    pchar_val = v15.avg_prob(char_models, char_val)
    pchar_test = v15.avg_prob(char_models, char_test)

    def all_probs(x_text, x_ctx):
        return (
            v3.ensemble_predict(base_text_models, x_text),
            v3.ensemble_predict(base_ctx_models, x_ctx),
            v15.avg_prob(spec_text_models, x_text),
            v15.avg_prob(spec_ctx_models, x_ctx),
            v15.avg_prob(error_text_models, x_text),
            v15.avg_prob(error_ctx_models, x_ctx),
            v5.avg_prob(ham_models, x_ctx),
        )

    print("[6/9] score validation/test", flush=True)
    val_probs = all_probs(xv, ctx_v)
    test_probs = all_probs(xe, ctx_e)

    pbase_text_val, pbase_ctx_val, _, _, _, _, pham_val = val_probs
    pbase_text_test, pbase_ctx_test, _, _, _, _, pham_test = test_probs

    stage1_gate, _ = v5.choose_gate(
        va, pbase_text_val, pbase_ctx_val, pham_val
    )
    stage1_val = v5.predict_gate(
        va,
        pbase_text_val,
        pbase_ctx_val,
        pham_val,
        stage1_gate["textThreshold"],
        stage1_gate["contextThreshold"],
        stage1_gate["hamVetoThreshold"],
    )
    stage1_test = v5.predict_gate(
        te,
        pbase_text_test,
        pbase_ctx_test,
        pham_test,
        stage1_gate["textThreshold"],
        stage1_gate["contextThreshold"],
        stage1_gate["hamVetoThreshold"],
    )

    feat_val = v15.features(*val_probs)
    feat_test = v15.features(*test_probs)

    # Original v15 baseline on the exact same 1500-message test.
    standard_v15_gate = v15.choose_gate(va, stage1_val, feat_val, safe=True)
    standard_v15_test = v15.predict(stage1_test, feat_test, standard_v15_gate)

    half_a, half_b = validation_halves(va)

    # v17 learns its tiny stacker on A and selects rescue thresholds on B.
    va_a = [row for row, use in zip(va, half_a) if use]
    xmeta_val = meta_matrix(va, val_probs, feat_val, pchar_val)
    xmeta_test = meta_matrix(te, test_probs, feat_test, pchar_test)

    stack_idx = np.where(half_a & (~stage1_val))[0]
    if len(np.unique([va[i]["y"] for i in stack_idx])) < 2:
        stack_idx = np.where(half_a)[0]

    print("[7/9] train border stacker + ham shield", flush=True)
    meta_models = fit_meta(
        xmeta_val[stack_idx],
        [va[i] for i in stack_idx],
    )

    pspam_val, pham_meta_val, effective_val, _ = meta_scores(
        meta_models, xmeta_val, va, pchar_val
    )
    pspam_test, pham_meta_test, effective_test, _ = meta_scores(
        meta_models, xmeta_test, te, pchar_test
    )

    # Base v15 gate is selected on validation A only, then border rescue on B.
    va_a_rows = [row for row, use in zip(va, half_a) if use]
    feat_a = {k: v[half_a] for k, v in feat_val.items()}
    stage1_a = stage1_val[half_a]
    v15_a_gate = v15.choose_gate(va_a_rows, stage1_a, feat_a, safe=True)

    base_v17_val = v15.predict(stage1_val, feat_val, v15_a_gate)
    base_v17_test = v15.predict(stage1_test, feat_test, v15_a_gate)

    fail_val = v15_fail_count(feat_val, v15_a_gate)
    fail_test = v15_fail_count(feat_test, v15_a_gate)

    va_b_rows = [row for row, use in zip(va, half_b) if use]

    print("[8/9] select zero-FP border rescue on validation B", flush=True)
    safe_border = choose_border_gate(
        va_b_rows,
        base_v17_val[half_b],
        effective_val[half_b],
        fail_val[half_b],
        pham_meta_val[half_b],
        safe=True,
    )
    balanced_border = choose_border_gate(
        va_b_rows,
        base_v17_val[half_b],
        effective_val[half_b],
        fail_val[half_b],
        pham_meta_val[half_b],
        safe=False,
    )

    safe_pred = apply_border(
        base_v17_test,
        effective_test,
        fail_test,
        pham_meta_test,
        safe_border,
    )
    balanced_pred = apply_border(
        base_v17_test,
        effective_test,
        fail_test,
        pham_meta_test,
        balanced_border,
    )

    stage1_stats = stats(te, stage1_test)
    v15_stats = stats(te, standard_v15_test)
    base_v17_stats = stats(te, base_v17_test)
    safe_stats = stats(te, safe_pred)
    balanced_stats = stats(te, balanced_pred)

    result = {
        "version": "v17-analysis-driven-border-rescue",
        "dataset": {
            "train": len(train),
            "validation": len(val),
            "test": len(te),
            "testSpam": base["spamTotal"],
            "testHam": base["hamTotal"],
        },
        "architecture": {
            "v15HardExampleExperts": True,
            "charOnlySpecialist": True,
            "metaTrain": "validation half A only",
            "gateSelection": "validation half B only",
            "hamShield": "independent ham logistic + BAYES_HAM penalty",
            "borderTarget": "messages failing at most 1-2 v15 rescue checks",
            "testLabelsUsedForTrainingOrThresholds": False,
        },
        "evaluationWarning": (
            "v17 feature design was informed by post-hoc analysis of the older "
            "public test family. Treat this as engineering evidence only; the "
            "33k external benchmark is the more important next check."
        ),
        "rspamdBayes": base,
        "stage1": stage1_stats,
        "v15StandardSame1500": v15_stats,
        "v17BaseFromHalfA": base_v17_stats,
        "safeBorderGate": safe_border,
        "safeTest": safe_stats,
        "balancedBorderGate": balanced_border,
        "balancedTest": balanced_stats,
    }

    (REPORTS / "v17-benchmark.json").write_text(json.dumps(result, indent=2))

    print("[9/9] write report", flush=True)
    md = [
        "# MailGuard v17 analysis-driven benchmark",
        "",
        f"Fast test: {len(te)} messages "
        f"({base['spamTotal']} spam + {base['hamTotal']} ham).",
        "",
        "| Mode | Recall | FN | FP | FP rate |",
        "|---|---:|---:|---:|---:|",
        f"| Rspamd + Bayes | {base['recall']:.2%} | "
        f"{base['spamTotal'] - base['spamDetected']} | "
        f"{base['falsePositives']} | {base['fpr']:.3%} |",
        f"| Stage 1 | {stage1_stats['recall']:.2%} | "
        f"{stage1_stats['falseNegatives']} | {stage1_stats['falsePositives']} | "
        f"{stage1_stats['fpr']:.3%} |",
        f"| v15 standard (same 1500) | {v15_stats['recall']:.2%} | "
        f"{v15_stats['falseNegatives']} | {v15_stats['falsePositives']} | "
        f"{v15_stats['fpr']:.3%} |",
        f"| v17 safe | {safe_stats['recall']:.2%} | "
        f"{safe_stats['falseNegatives']} | {safe_stats['falsePositives']} | "
        f"{safe_stats['fpr']:.3%} |",
        f"| v17 balanced | {balanced_stats['recall']:.2%} | "
        f"{balanced_stats['falseNegatives']} | {balanced_stats['falsePositives']} | "
        f"{balanced_stats['fpr']:.3%} |",
        "",
        f"Safe border gate: {safe_border}",
        f"Balanced border gate: {balanced_border}",
        "",
        "v17 deliberately targets near-miss errors; it does not lower the global "
        "spam threshold for deep low-score messages.",
    ]

    report = "\n".join(md) + "\n"
    (REPORTS / "v17-benchmark.md").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
