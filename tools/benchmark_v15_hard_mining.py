#!/usr/bin/env python3
import json
import math
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

import benchmark_improved as b
import benchmark_v3 as v3
import benchmark_v5 as v5

REPORTS = Path("reports")
SEED = 20261001


def base_weights(rows):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    w = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)

    for i, row in enumerate(rows):
        if not row["y"] and "hard_ham" in row["source"]:
            w[i] *= 8.0
        if not row["y"] and row["rscore"] >= 4.0:
            w[i] *= 4.0
        if row["y"] and row["rscore"] <= 4.0:
            w[i] *= 2.5

    return w


def fit_one(x, y, weights, alpha, seed):
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=300,
        tol=1e-5,
        average=True,
        fit_intercept=True,
        random_state=seed,
    )
    model.fit(x, y, sample_weight=weights)
    return model


def crossfit_hardness(xtext, xctx, rows):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    base_w = base_weights(rows)

    ptext = np.zeros(len(rows), dtype=np.float64)
    pctx = np.zeros(len(rows), dtype=np.float64)

    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)

    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(np.zeros(len(rows)), y), 1):
        print(f"[4/9] cross-fit fold {fold}/4", flush=True)

        mt = fit_one(
            xtext[fit_idx],
            y[fit_idx],
            base_w[fit_idx],
            2e-4,
            SEED + fold * 31,
        )
        mc = fit_one(
            xctx[fit_idx],
            y[fit_idx],
            base_w[fit_idx],
            2e-4,
            SEED + 500 + fold * 31,
        )

        ptext[hold_idx] = mt.predict_proba(xtext[hold_idx])[:, 1]
        pctx[hold_idx] = mc.predict_proba(xctx[hold_idx])[:, 1]

    score = np.sqrt(np.clip(ptext, 1e-8, 1.0) * np.clip(pctx, 1e-8, 1.0))

    hardness = np.where(y == 1, 1.0 - score, score)
    hardness = np.clip(hardness, 0.0, 1.0)

    return hardness, score


def mining_weights(rows, hardness):
    w = base_weights(rows)

    for i, row in enumerate(rows):
        h = float(hardness[i])

        if row["y"]:
            # False-negative-like training examples get much more attention.
            w[i] *= 1.0 + 7.0 * (h ** 2)
            if h >= 0.65:
                w[i] *= 1.8
        else:
            # Ham that looks spammy is exactly the class that causes costly FP.
            w[i] *= 1.0 + 10.0 * (h ** 2)
            if h >= 0.65:
                w[i] *= 2.2
            if "hard_ham" in row["source"]:
                w[i] *= 1.6

    return w


def fit_specialist_ensemble(x, rows, hardness, seed_offset):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    w = mining_weights(rows, hardness)

    models = []
    for j, alpha in enumerate((6e-5, 1.8e-4, 5e-4)):
        models.append(
            fit_one(
                x,
                y,
                w,
                alpha,
                SEED + seed_offset + j * 67,
            )
        )
    return models


def avg_prob(models, x):
    return np.mean([m.predict_proba(x)[:, 1] for m in models], axis=0)


def hard_subset_indices(rows, hardness):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)

    spam_idx = np.where(y == 1)[0]
    ham_idx = np.where(y == 0)[0]

    # Keep difficult spam plus difficult ham. Easy examples stay represented
    # through the full specialist ensemble above.
    spam_cut = np.quantile(hardness[spam_idx], 0.45)
    ham_cut = np.quantile(hardness[ham_idx], 0.55)

    idx = np.where(
        ((y == 1) & (hardness >= spam_cut))
        | ((y == 0) & (hardness >= ham_cut))
    )[0]

    return idx


def fit_error_experts(xtext, xctx, rows, hardness):
    idx = hard_subset_indices(rows, hardness)
    hard_rows = [rows[i] for i in idx]
    hard_h = hardness[idx]
    y = np.asarray([r["y"] for r in hard_rows], dtype=np.int32)

    w = mining_weights(hard_rows, hard_h)

    text_models = []
    ctx_models = []

    for j, alpha in enumerate((8e-5, 2.5e-4)):
        text_models.append(
            fit_one(
                xtext[idx],
                y,
                w,
                alpha,
                SEED + 8100 + j * 101,
            )
        )
        ctx_models.append(
            fit_one(
                xctx[idx],
                y,
                w,
                alpha,
                SEED + 9100 + j * 101,
            )
        )

    return text_models, ctx_models, len(idx)


def features(
    pbase_text,
    pbase_ctx,
    pspec_text,
    pspec_ctx,
    perror_text,
    perror_ctx,
    pham,
):
    signals = np.vstack([
        np.clip(pbase_text, 1e-8, 1.0),
        np.clip(pbase_ctx, 1e-8, 1.0),
        np.clip(pspec_text, 1e-8, 1.0),
        np.clip(pspec_ctx, 1e-8, 1.0),
        np.clip(perror_text, 1e-8, 1.0),
        np.clip(perror_ctx, 1e-8, 1.0),
        np.clip(1.0 - pham, 1e-8, 1.0),
    ])

    sorted_signals = np.sort(signals, axis=0)

    return {
        "score": np.exp(np.mean(np.log(signals), axis=0)),
        "consensus": sorted_signals[1],
        "median": np.median(signals, axis=0),
        "minSignal": sorted_signals[0],
        "agreement": (signals >= 0.50).sum(axis=0),
        "specialist": np.sqrt(
            np.clip(pspec_text, 1e-8, 1.0)
            * np.clip(pspec_ctx, 1e-8, 1.0)
        ),
        "errorExpert": np.sqrt(
            np.clip(perror_text, 1e-8, 1.0)
            * np.clip(perror_ctx, 1e-8, 1.0)
        ),
        "ham": pham,
    }


def stats(rows, pred, stage1):
    y = np.asarray([r["y"] for r in rows], dtype=bool)

    spam = int(y.sum())
    ham = int((~y).sum())
    tp = int((y & pred).sum())
    fp = int(((~y) & pred).sum())
    stage_tp = int((y & stage1).sum())
    stage_fp = int(((~y) & stage1).sum())

    return {
        "spamTotal": spam,
        "hamTotal": ham,
        "spamDetected": tp,
        "falsePositives": fp,
        "recall": tp / max(1, spam),
        "fpr": fp / max(1, ham),
        "rescuedOverStage1": tp - stage_tp,
        "addedFalsePositivesOverStage1": fp - stage_fp,
    }


def threshold_grid(values, fixed, low_q=0.30, count=24):
    q = np.quantile(values, np.linspace(low_q, 1.0, count))
    return np.unique(np.concatenate([np.asarray(fixed), q]))


def predict(stage1, feat, gate):
    rescue = (
        (~stage1)
        & (feat["score"] >= gate["scoreThreshold"])
        & (feat["consensus"] >= gate["consensusThreshold"])
        & (feat["median"] >= gate["medianThreshold"])
        & (feat["specialist"] >= gate["specialistThreshold"])
        & (feat["errorExpert"] >= gate["errorExpertThreshold"])
        & (feat["ham"] <= gate["hamVetoThreshold"])
        & (feat["agreement"] >= gate["minAgreement"])
    )

    return stage1 | rescue


def validation_halves(rows):
    a = np.zeros(len(rows), dtype=bool)
    bmask = np.zeros(len(rows), dtype=bool)

    buckets = {}
    for i, row in enumerate(rows):
        buckets.setdefault(row["source"], []).append(i)

    for indices in buckets.values():
        for j, idx in enumerate(indices):
            (a if j % 2 == 0 else bmask)[idx] = True

    return a, bmask


def choose_gate(rows, stage1, feat, safe):
    y = np.asarray([r["y"] for r in rows], dtype=bool)
    ham = ~y
    hard_ham = np.asarray([
        (not r["y"]) and ("hard_ham" in r["source"])
        for r in rows
    ], dtype=bool)

    half_a, half_b = validation_halves(rows)

    base_fp = int((ham & stage1).sum())
    base_hard_fp = int((hard_ham & stage1).sum())
    base_a_fp = int((ham & stage1 & half_a).sum())
    base_b_fp = int((ham & stage1 & half_b).sum())

    score_grid = threshold_grid(
        feat["score"],
        [.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.93,.95,.97,.98,.99,.995,1.000001],
    )
    consensus_grid = threshold_grid(
        feat["consensus"],
        [.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90],
        0.20,
        18,
    )
    median_grid = np.asarray([.45,.50,.55,.60,.65,.70,.75,.80,.85,.90])
    specialist_grid = np.asarray([.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90])
    error_grid = np.asarray([.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85])
    ham_grid = np.asarray([.03,.05,.08,.10,.12,.15,.20,.25,.30,.35,.40])
    agree_grid = (4, 5, 6)

    best = None

    for st in score_grid:
        m1 = feat["score"] >= st

        for ct in consensus_grid:
            m2 = m1 & (feat["consensus"] >= ct)
            if not np.any(m2 & (~stage1)):
                continue

            for mt in median_grid:
                m3 = m2 & (feat["median"] >= mt)
                if not np.any(m3 & (~stage1)):
                    continue

                for sp in specialist_grid:
                    m4 = m3 & (feat["specialist"] >= sp)
                    if not np.any(m4 & (~stage1)):
                        continue

                    for er in error_grid:
                        m5 = m4 & (feat["errorExpert"] >= er)
                        if not np.any(m5 & (~stage1)):
                            continue

                        for hv in ham_grid:
                            m6 = m5 & (feat["ham"] <= hv)
                            if not np.any(m6 & (~stage1)):
                                continue

                            for agreement in agree_grid:
                                rescue = (
                                    (~stage1)
                                    & m6
                                    & (feat["agreement"] >= agreement)
                                )
                                pred = stage1 | rescue

                                fp = int((ham & pred).sum())
                                hard_fp = int((hard_ham & pred).sum())
                                a_fp = int((ham & pred & half_a).sum())
                                b_fp = int((ham & pred & half_b).sum())

                                if hard_fp > base_hard_fp:
                                    continue

                                if safe:
                                    # Stronger than older versions: no added FP
                                    # is permitted in either validation half.
                                    if a_fp > base_a_fp or b_fp > base_b_fp:
                                        continue
                                else:
                                    if fp - base_fp > 1:
                                        continue
                                    if a_fp - base_a_fp > 1:
                                        continue
                                    if b_fp - base_b_fp > 1:
                                        continue

                                cur = stats(rows, pred, stage1)
                                point = {
                                    **cur,
                                    "scoreThreshold": float(st),
                                    "consensusThreshold": float(ct),
                                    "medianThreshold": float(mt),
                                    "specialistThreshold": float(sp),
                                    "errorExpertThreshold": float(er),
                                    "hamVetoThreshold": float(hv),
                                    "minAgreement": int(agreement),
                                    "hardHamFalsePositives": hard_fp,
                                    "validationHalfAAddedFp": a_fp - base_a_fp,
                                    "validationHalfBAddedFp": b_fp - base_b_fp,
                                }

                                key = (
                                    point["rescuedOverStage1"],
                                    -point["addedFalsePositivesOverStage1"],
                                    point["scoreThreshold"],
                                    point["consensusThreshold"],
                                    point["specialistThreshold"],
                                    point["errorExpertThreshold"],
                                    point["minAgreement"],
                                    -point["hamVetoThreshold"],
                                )

                                if best is None or key > best[0]:
                                    best = (key, point)

    if best is not None:
        return best[1]

    base = stats(rows, stage1.copy(), stage1)
    return {
        **base,
        "scoreThreshold": 1.000001,
        "consensusThreshold": 1.000001,
        "medianThreshold": 1.000001,
        "specialistThreshold": 1.000001,
        "errorExpertThreshold": 1.000001,
        "hamVetoThreshold": 0.0,
        "minAgreement": 7,
        "hardHamFalsePositives": base_hard_fp,
        "validationHalfAAddedFp": 0,
        "validationHalfBAddedFp": 0,
    }


def review(rows, stage1, feat):
    residual = [i for i in range(len(rows)) if not stage1[i]]
    budget = min(max(1, math.ceil(len(rows) * .01)), len(residual))

    top = sorted(
        residual,
        key=lambda i: feat["score"][i],
        reverse=True,
    )[:budget]

    top_spam = sum(rows[i]["y"] for i in top)

    rnd = random.Random(SEED)
    random_hits = []
    for _ in range(2000):
        sample = rnd.sample(residual, budget)
        random_hits.append(sum(rows[i]["y"] for i in sample))

    mean = float(np.mean(random_hits))

    return {
        "budget": budget,
        "topRiskSpamFound": int(top_spam),
        "randomSpamFoundMean": mean,
        "lift": top_spam / mean if mean else None,
    }


def main():
    REPORTS.mkdir(exist_ok=True)
    b.wait_rspamd()

    groups = b.prepare()
    train, val, test = v3.build_splits(groups)

    print("TRAIN", len(train), "VAL", len(val), "TEST", len(test), flush=True)
    print("v15: cross-fitted hard-example mining specialists", flush=True)

    b.reset_bayes()
    b.learn(train)

    print("[1/9] scan train", flush=True)
    tr = b.scan_many(train, "v15-train")
    print("[2/9] scan validation", flush=True)
    va = b.scan_many(val, "v15-val")
    print("[3/9] scan test", flush=True)
    te = b.scan_many(test, "v15-test")

    base = b.base_metrics(te)

    xt, xv, xe, ctx_t, ctx_v, ctx_e = v3.matrices(tr, va, te)
    residual_idx = [i for i, row in enumerate(tr) if not row["rspam"]]
    residual_rows = [tr[i] for i in residual_idx]

    xr_text = xt[residual_idx]
    xr_ctx = ctx_t[residual_idx]

    hardness, oof_score = crossfit_hardness(
        xr_text,
        xr_ctx,
        residual_rows,
    )

    print(
        "[5/9] hard mining: "
        f"mean={hardness.mean():.4f} "
        f"p90={np.quantile(hardness, .90):.4f}",
        flush=True,
    )

    # Base classical experts provide a stable reference.
    base_text_models = v3.fit_ensemble(xr_text, residual_rows)
    base_ctx_models = v3.fit_ensemble(xr_ctx, residual_rows)
    ham_models = v5.fit_ham_ensemble(xr_ctx, residual_rows)

    # Specialists are trained with cross-fitted error severity as sample weight.
    spec_text_models = fit_specialist_ensemble(
        xr_text, residual_rows, hardness, 2100
    )
    spec_ctx_models = fit_specialist_ensemble(
        xr_ctx, residual_rows, hardness, 3100
    )

    print("[6/9] train hardest-example-only experts", flush=True)
    error_text_models, error_ctx_models, mined_count = fit_error_experts(
        xr_text,
        xr_ctx,
        residual_rows,
        hardness,
    )

    def all_probs(x_text, x_ctx):
        return (
            v3.ensemble_predict(base_text_models, x_text),
            v3.ensemble_predict(base_ctx_models, x_ctx),
            avg_prob(spec_text_models, x_text),
            avg_prob(spec_ctx_models, x_ctx),
            avg_prob(error_text_models, x_text),
            avg_prob(error_ctx_models, x_ctx),
            v5.avg_prob(ham_models, x_ctx),
        )

    print("[7/9] score validation and test", flush=True)
    val_probs = all_probs(xv, ctx_v)
    test_probs = all_probs(xe, ctx_e)

    pbase_text_val, pbase_ctx_val, _, _, _, _, pham_val = val_probs
    pbase_text_test, pbase_ctx_test, _, _, _, _, pham_test = test_probs

    stage1_gate, _ = v5.choose_gate(
        va,
        pbase_text_val,
        pbase_ctx_val,
        pham_val,
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

    feat_val = features(*val_probs)
    feat_test = features(*test_probs)

    print("[8/9] choose robust rescue gate", flush=True)
    safe_gate = choose_gate(va, stage1_val, feat_val, safe=True)
    balanced_gate = choose_gate(va, stage1_val, feat_val, safe=False)

    safe_pred = predict(stage1_test, feat_test, safe_gate)
    balanced_pred = predict(stage1_test, feat_test, balanced_gate)

    stage1_stats = v5.stats(te, stage1_test)
    safe_stats = stats(te, safe_pred, stage1_test)
    balanced_stats = stats(te, balanced_pred, stage1_test)

    print("[9/9] write report", flush=True)

    result = {
        "version": "v15-crossfit-hard-example-mining",
        "dataset": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "testSpam": base["spamTotal"],
            "testHam": base["hamTotal"],
            "residualTraining": len(residual_rows),
            "hardSubsetTraining": mined_count,
        },
        "architecture": {
            "stage1": "v5 precision gate",
            "hardness": "4-fold out-of-fold error score on residual training only",
            "specialistTextModels": 3,
            "specialistContextModels": 3,
            "hardSubsetTextModels": 2,
            "hardSubsetContextModels": 2,
            "hamVetoModels": 3,
            "safeValidation": "zero added FP independently on two validation halves",
            "hardHamAddedFpBudget": 0,
            "testLabelsUsedForTrainingOrThresholds": False,
        },
        "evaluationNote": (
            "The public benchmark family has been observed repeatedly during "
            "development. Treat test results as engineering evidence, not a "
            "final unbiased production estimate."
        ),
        "rspamdBayes": base,
        "stage1ValidationGate": stage1_gate,
        "stage1Test": stage1_stats,
        "safeValidationGate": safe_gate,
        "safeTest": safe_stats,
        "balancedValidationGate": balanced_gate,
        "balancedTest": balanced_stats,
        "review1pctAfterStage1": review(te, stage1_test, feat_test),
        "hardnessSummary": {
            "mean": float(hardness.mean()),
            "p90": float(np.quantile(hardness, .90)),
            "max": float(hardness.max()),
        },
    }

    (REPORTS / "v15-benchmark.json").write_text(json.dumps(result, indent=2))

    md = [
        "# MailGuard v15 hard-example mining benchmark",
        "",
        f"Final test: {base['spamTotal']} spam + {base['hamTotal']} ham.",
        "",
        "| Mode | Spam recall | FP total | Extra spam vs stage 1 | Extra FP vs stage 1 |",
        "|---|---:|---:|---:|---:|",
        f"| Rspamd + Bayes | {base['recall']:.2%} | "
        f"{base['falsePositives']}/{base['hamTotal']} | - | - |",
        f"| Stage 1 (v5 precision) | {stage1_stats['recall']:.2%} | "
        f"{stage1_stats['falsePositives']}/{stage1_stats['hamTotal']} | 0 | 0 |",
        f"| v15 safe | {safe_stats['recall']:.2%} | "
        f"{safe_stats['falsePositives']}/{safe_stats['hamTotal']} | "
        f"{safe_stats['rescuedOverStage1']} | "
        f"{safe_stats['addedFalsePositivesOverStage1']} |",
        f"| v15 balanced | {balanced_stats['recall']:.2%} | "
        f"{balanced_stats['falsePositives']}/{balanced_stats['hamTotal']} | "
        f"{balanced_stats['rescuedOverStage1']} | "
        f"{balanced_stats['addedFalsePositivesOverStage1']} |",
        "",
        f"Hard-example subset: {mined_count}/{len(residual_rows)} residual training messages.",
        "",
        "## Validation gates",
        "",
        f"Safe: {safe_gate}",
        f"Balanced: {balanced_gate}",
        "",
        "## 1% review after stage 1",
        "",
        f"Random {result['review1pctAfterStage1']['randomSpamFoundMean']:.2f}, "
        f"top-risk {result['review1pctAfterStage1']['topRiskSpamFound']}, "
        f"lift {result['review1pctAfterStage1']['lift']}.",
        "",
        "Important: use a fresh modern lockbox before final quality claims.",
    ]

    report = "\n".join(md) + "\n"
    (REPORTS / "v15-benchmark.md").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
