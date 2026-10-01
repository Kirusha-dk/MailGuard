#!/usr/bin/env python3
import json
import math
import random
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.utils.class_weight import compute_sample_weight

import benchmark_improved as b
import benchmark_v3 as v3
import benchmark_v5 as v5

REPORTS = Path("reports")
SEED = 20261001


def weights(rows):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    w = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)
    for i, r in enumerate(rows):
        if not r["y"] and "hard_ham" in r["source"]:
            w[i] *= 10.0
        if not r["y"] and r["rscore"] >= 4.0:
            w[i] *= 4.0
        if r["y"] and r["rscore"] <= 4.0:
            w[i] *= 2.0
    return w


def fit_sgd_ensemble(x, rows, alphas=(7e-5, 2e-4, 7e-4)):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    sw = weights(rows)
    out = []
    for i, alpha in enumerate(alphas):
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            max_iter=350,
            tol=1e-5,
            average=True,
            random_state=SEED + i * 17,
            fit_intercept=True,
        )
        model.fit(x, y, sample_weight=sw)
        out.append(model)
    return out


def avg_prob(models, x):
    return np.mean([m.predict_proba(x)[:, 1] for m in models], axis=0)


def fit_nb_ensemble(x, rows):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    sw = weights(rows)
    models = []
    for alpha in (0.08, 0.25, 0.75):
        model = ComplementNB(alpha=alpha)
        model.fit(x, y, sample_weight=sw)
        models.append(model)
    return models


def make_tfidf(train_rows, val_rows, test_rows):
    train_docs = [v3.pure_doc(r) for r in train_rows]
    val_docs = [v3.pure_doc(r) for r in val_rows]
    test_docs = [v3.pure_doc(r) for r in test_rows]

    word = TfidfVectorizer(
        lowercase=True,
        sublinear_tf=True,
        min_df=2,
        max_df=0.995,
        max_features=180000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[\w@.\-]{2,}\b",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        lowercase=True,
        sublinear_tf=True,
        min_df=2,
        max_features=220000,
        analyzer="char_wb",
        ngram_range=(3, 6),
        dtype=np.float32,
    )

    print("[4/8] fit word TF-IDF", flush=True)
    xw_tr = word.fit_transform(train_docs)
    xw_va = word.transform(val_docs)
    xw_te = word.transform(test_docs)

    print("[5/8] fit char TF-IDF", flush=True)
    xc_tr = char.fit_transform(train_docs)
    xc_va = char.transform(val_docs)
    xc_te = char.transform(test_docs)

    return (xw_tr, xw_va, xw_te), (xc_tr, xc_va, xc_te)


def features(pword, pchar, pnb, phash, pctx, pham):
    signals = np.vstack([
        np.clip(pword, 1e-7, 1.0),
        np.clip(pchar, 1e-7, 1.0),
        np.clip(pnb, 1e-7, 1.0),
        np.clip(phash, 1e-7, 1.0),
        np.clip(pctx, 1e-7, 1.0),
        np.clip(1.0 - pham, 1e-7, 1.0),
    ])

    score = np.exp(np.mean(np.log(signals), axis=0))
    sorted_signals = np.sort(signals, axis=0)
    second_lowest = sorted_signals[1]
    median = np.median(signals, axis=0)
    agree = (signals >= 0.50).sum(axis=0)

    return {
        "score": score,
        "consensus": second_lowest,
        "median": median,
        "agreement": agree,
        "word": pword,
        "char": pchar,
        "nb": pnb,
        "ctx": pctx,
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


def val_halves(rows):
    a = np.zeros(len(rows), dtype=bool)
    bmask = np.zeros(len(rows), dtype=bool)
    by_source = {}
    for i, row in enumerate(rows):
        by_source.setdefault(row["source"], []).append(i)
    for indices in by_source.values():
        for j, idx in enumerate(indices):
            (a if j % 2 == 0 else bmask)[idx] = True
    return a, bmask


def grid(values, fixed, start=0.35, count=28):
    q = np.quantile(values, np.linspace(start, 1.0, count))
    return np.unique(np.concatenate([np.asarray(fixed), q]))


def prediction(stage1, feat, gate):
    rescue = (
        (~stage1)
        & (feat["score"] >= gate["scoreThreshold"])
        & (feat["consensus"] >= gate["consensusThreshold"])
        & (feat["median"] >= gate["medianThreshold"])
        & (feat["ctx"] >= gate["contextThreshold"])
        & (feat["ham"] <= gate["hamVetoThreshold"])
        & (feat["agreement"] >= gate["minAgreement"])
    )
    return stage1 | rescue


def choose_gate(rows, stage1, feat, mode):
    y = np.asarray([r["y"] for r in rows], dtype=bool)
    hard = np.asarray([
        (not r["y"]) and ("hard_ham" in r["source"])
        for r in rows
    ], dtype=bool)

    half_a, half_b = val_halves(rows)
    ham = ~y

    base_fp = int((ham & stage1).sum())
    base_hard_fp = int((hard & stage1).sum())
    base_fp_a = int((ham & stage1 & half_a).sum())
    base_fp_b = int((ham & stage1 & half_b).sum())

    score_grid = grid(
        feat["score"],
        [.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.93,.95,.97,.98,.99,.995,.999,1.000001],
    )
    consensus_grid = grid(
        feat["consensus"],
        [.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.95],
        0.15,
        18,
    )
    median_grid = np.asarray([.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.95])
    context_grid = np.asarray([.40,.50,.60,.70,.80,.90,.95])
    ham_grid = np.asarray([.03,.05,.08,.10,.15,.20,.25,.30,.40])
    agree_grid = (4, 5, 6)

    best = None

    for st in score_grid:
        ms = feat["score"] >= st
        for ct in consensus_grid:
            mc = ms & (feat["consensus"] >= ct)
            if not np.any(mc & (~stage1)):
                continue
            for mt in median_grid:
                mm = mc & (feat["median"] >= mt)
                if not np.any(mm & (~stage1)):
                    continue
                for xt in context_grid:
                    mx = mm & (feat["ctx"] >= xt)
                    if not np.any(mx & (~stage1)):
                        continue
                    for hv in ham_grid:
                        mh = mx & (feat["ham"] <= hv)
                        if not np.any(mh & (~stage1)):
                            continue
                        for ag in agree_grid:
                            rescue = (~stage1) & mh & (feat["agreement"] >= ag)
                            pred = stage1 | rescue

                            fp = int((ham & pred).sum())
                            hard_fp = int((hard & pred).sum())
                            fp_a = int((ham & pred & half_a).sum())
                            fp_b = int((ham & pred & half_b).sum())

                            if hard_fp > base_hard_fp:
                                continue

                            if mode == "safe":
                                if fp_a > base_fp_a or fp_b > base_fp_b:
                                    continue
                            else:
                                if fp - base_fp > 1:
                                    continue
                                if fp_a - base_fp_a > 1 or fp_b - base_fp_b > 1:
                                    continue

                            cur = stats(rows, pred, stage1)
                            point = {
                                **cur,
                                "scoreThreshold": float(st),
                                "consensusThreshold": float(ct),
                                "medianThreshold": float(mt),
                                "contextThreshold": float(xt),
                                "hamVetoThreshold": float(hv),
                                "minAgreement": int(ag),
                                "hardHamFalsePositives": hard_fp,
                                "validationHalfAAddedFp": fp_a - base_fp_a,
                                "validationHalfBAddedFp": fp_b - base_fp_b,
                            }

                            key = (
                                point["rescuedOverStage1"],
                                -point["addedFalsePositivesOverStage1"],
                                point["scoreThreshold"],
                                point["consensusThreshold"],
                                point["medianThreshold"],
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
        "contextThreshold": 1.000001,
        "hamVetoThreshold": 0.0,
        "minAgreement": 6,
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
    counts = []
    for _ in range(2000):
        sample = rnd.sample(residual, budget)
        counts.append(sum(rows[i]["y"] for i in sample))

    mean = float(np.mean(counts))
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
    print("v14: multi-view sparse consensus for low-FP residual spam", flush=True)

    b.reset_bayes()
    b.learn(train)

    print("[1/8] scan train", flush=True)
    tr = b.scan_many(train, "v14-train")
    print("[2/8] scan validation", flush=True)
    va = b.scan_many(val, "v14-val")
    print("[3/8] scan test", flush=True)
    te = b.scan_many(test, "v14-test")

    base = b.base_metrics(te)

    xt, xv, xe, ctx_t, ctx_v, ctx_e = v3.matrices(tr, va, te)
    residual_idx = [i for i, r in enumerate(tr) if not r["rspam"]]
    residual_rows = [tr[i] for i in residual_idx]

    # Existing independent hashed/context experts.
    hash_models = v3.fit_ensemble(xt[residual_idx], residual_rows)
    ctx_models = v3.fit_ensemble(ctx_t[residual_idx], residual_rows)
    ham_models = v5.fit_ham_ensemble(ctx_t[residual_idx], residual_rows)

    phash_val = v3.ensemble_predict(hash_models, xv)
    pctx_val = v3.ensemble_predict(ctx_models, ctx_v)
    pham_val = v5.avg_prob(ham_models, ctx_v)

    phash_test = v3.ensemble_predict(hash_models, xe)
    pctx_test = v3.ensemble_predict(ctx_models, ctx_e)
    pham_test = v5.avg_prob(ham_models, ctx_e)

    stage1_gate, _ = v5.choose_gate(va, phash_val, pctx_val, pham_val)
    stage1_val = v5.predict_gate(
        va, phash_val, pctx_val, pham_val,
        stage1_gate["textThreshold"],
        stage1_gate["contextThreshold"],
        stage1_gate["hamVetoThreshold"],
    )
    stage1_test = v5.predict_gate(
        te, phash_test, pctx_test, pham_test,
        stage1_gate["textThreshold"],
        stage1_gate["contextThreshold"],
        stage1_gate["hamVetoThreshold"],
    )

    (word, char) = make_tfidf(residual_rows, va, te)
    xw_tr, xw_va, xw_te = word
    xc_tr, xc_va, xc_te = char

    print("[6/8] train diverse sparse experts", flush=True)
    word_models = fit_sgd_ensemble(xw_tr, residual_rows)
    char_models = fit_sgd_ensemble(xc_tr, residual_rows)

    xnb_tr = hstack([xw_tr, xc_tr], format="csr")
    xnb_va = hstack([xw_va, xc_va], format="csr")
    xnb_te = hstack([xw_te, xc_te], format="csr")
    nb_models = fit_nb_ensemble(xnb_tr, residual_rows)

    pword_val = avg_prob(word_models, xw_va)
    pchar_val = avg_prob(char_models, xc_va)
    pnb_val = avg_prob(nb_models, xnb_va)

    pword_test = avg_prob(word_models, xw_te)
    pchar_test = avg_prob(char_models, xc_te)
    pnb_test = avg_prob(nb_models, xnb_te)

    feat_val = features(
        pword_val, pchar_val, pnb_val,
        phash_val, pctx_val, pham_val,
    )
    feat_test = features(
        pword_test, pchar_test, pnb_test,
        phash_test, pctx_test, pham_test,
    )

    print("[7/8] select robust two-half validation gates", flush=True)
    safe_gate = choose_gate(va, stage1_val, feat_val, "safe")
    balanced_gate = choose_gate(va, stage1_val, feat_val, "balanced")

    safe_pred = prediction(stage1_test, feat_test, safe_gate)
    balanced_pred = prediction(stage1_test, feat_test, balanced_gate)

    stage1_stats = v5.stats(te, stage1_test)
    safe_test = stats(te, safe_pred, stage1_test)
    balanced_test = stats(te, balanced_pred, stage1_test)

    print("[8/8] write report", flush=True)
    result = {
        "version": "v14-multiview-sparse-consensus",
        "dataset": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "testSpam": base["spamTotal"],
            "testHam": base["hamTotal"],
        },
        "architecture": {
            "stage1": "v5 precision gate",
            "wordView": "train-fitted TF-IDF word 1-2 grams, 3-model SGD ensemble",
            "charView": "train-fitted TF-IDF char 3-6 grams, 3-model SGD ensemble",
            "frequencyView": "3-model ComplementNB ensemble",
            "legacyHashView": "v3 word+char hashing ensemble",
            "rspamdContextView": "v3 context ensemble",
            "hamVeto": "v5 independent ham ensemble",
            "safeGate": "zero added FP required independently on both validation halves",
            "hardHamAddedFpBudget": 0,
        },
        "evaluationNote": (
            "Repeated development has observed this public benchmark family. "
            "Use these numbers as engineering evidence only; final claims need "
            "a fresh modern holdout."
        ),
        "rspamdBayes": base,
        "stage1ValidationGate": stage1_gate,
        "stage1Test": stage1_stats,
        "safeValidationGate": safe_gate,
        "safeTest": safe_test,
        "balancedValidationGate": balanced_gate,
        "balancedTest": balanced_test,
        "review1pctAfterStage1": review(te, stage1_test, feat_test),
    }

    (REPORTS / "v14-benchmark.json").write_text(json.dumps(result, indent=2))

    md = [
        "# MailGuard v14 multi-view sparse consensus benchmark",
        "",
        f"Final test: {base['spamTotal']} spam + {base['hamTotal']} ham.",
        "",
        "| Mode | Spam recall | FP total | Extra spam vs stage 1 | Extra FP vs stage 1 |",
        "|---|---:|---:|---:|---:|",
        f"| Rspamd + Bayes | {base['recall']:.2%} | "
        f"{base['falsePositives']}/{base['hamTotal']} | - | - |",
        f"| Stage 1 (v5 precision) | {stage1_stats['recall']:.2%} | "
        f"{stage1_stats['falsePositives']}/{stage1_stats['hamTotal']} | 0 | 0 |",
        f"| v14 safe | {safe_test['recall']:.2%} | "
        f"{safe_test['falsePositives']}/{safe_test['hamTotal']} | "
        f"{safe_test['rescuedOverStage1']} | "
        f"{safe_test['addedFalsePositivesOverStage1']} |",
        f"| v14 balanced | {balanced_test['recall']:.2%} | "
        f"{balanced_test['falsePositives']}/{balanced_test['hamTotal']} | "
        f"{balanced_test['rescuedOverStage1']} | "
        f"{balanced_test['addedFalsePositivesOverStage1']} |",
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
        "Important: the same public corpus family has been inspected repeatedly. "
        "A fresh lockbox is required before treating this as a final quality estimate.",
    ]

    report = "\n".join(md) + "\n"
    (REPORTS / "v14-benchmark.md").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
