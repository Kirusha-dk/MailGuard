#!/usr/bin/env python3
import json, math, random
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_sample_weight

import benchmark_improved as b
import benchmark_v3 as v3
import benchmark_v5 as v5

REPORTS = Path("reports")
SEED = 20261001

def fit_spam_models(xtrain, rows, extra_weights=None):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)

    for i, r in enumerate(rows):
        if not r["y"] and "hard_ham" in r["source"]:
            weights[i] *= 8.0
        if not r["y"] and r["rscore"] >= 4.0:
            weights[i] *= 4.0
        if r["y"] and r["rscore"] <= 4.0:
            weights[i] *= 2.0

    if extra_weights is not None:
        weights *= extra_weights

    models = []
    for i, alpha in enumerate((1e-4, 3e-4, 1e-3)):
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            max_iter=300,
            tol=1e-5,
            random_state=SEED + i,
            average=True,
            fit_intercept=True,
        )
        clf.fit(xtrain, y, sample_weight=weights)
        models.append(clf)
    return models

def avg_prob(models, x):
    return np.mean([m.predict_proba(x)[:, 1] for m in models], axis=0)

def hard_negative_weights(rows, ptext, pctx):
    risk = np.sqrt(np.clip(ptext, 1e-9, 1) * np.clip(pctx, 1e-9, 1))
    ham_idx = np.asarray([not r["y"] for r in rows], dtype=bool)
    ham_risk = risk[ham_idx]
    if ham_risk.size == 0:
        return np.ones(len(rows), dtype=np.float64)

    q90 = float(np.quantile(ham_risk, .90))
    q97 = float(np.quantile(ham_risk, .97))

    extra = np.ones(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        if r["y"]:
            continue
        if risk[i] >= q97:
            extra[i] *= 10.0
        elif risk[i] >= q90:
            extra[i] *= 5.0
    return extra

def candidates(values, lower=.50):
    fixed = np.asarray([
        lower, .55, .60, .65, .70, .75, .80, .85, .90, .93,
        .95, .97, .98, .99, .995, .998, .999, 1.000001
    ])
    q = np.quantile(values, np.linspace(.45, 1.0, 36))
    return np.unique(np.concatenate([fixed, q[q >= lower]]))

def ham_candidates(values):
    fixed = np.asarray([.02,.05,.08,.10,.15,.20,.25,.30,.35,.40,.50,.60,.70,.80])
    q = np.quantile(values, np.linspace(0, .85, 24))
    return np.unique(np.concatenate([fixed, q]))

def make_pred(rows, ptext, pctx, pham, tt, tc, hv):
    base = np.asarray([r["rspam"] for r in rows], dtype=bool)
    added = (
        (ptext >= tt) &
        (pctx >= tc) &
        (pham <= hv)
    )
    return base | added

def metrics(rows, pred):
    y = np.asarray([r["y"] for r in rows], dtype=bool)
    base = np.asarray([r["rspam"] for r in rows], dtype=bool)

    spam_total = int(y.sum())
    ham_total = int((~y).sum())
    tp = int((y & pred).sum())
    fp = int(((~y) & pred).sum())

    base_tp = int((y & base).sum())
    base_fp = int(((~y) & base).sum())

    return {
        "spamTotal": spam_total,
        "hamTotal": ham_total,
        "spamDetected": tp,
        "falsePositives": fp,
        "recall": tp / max(1, spam_total),
        "fpr": fp / max(1, ham_total),
        "baseSpamDetected": base_tp,
        "baseFalsePositives": base_fp,
        "rescuedSpam": tp - base_tp,
        "addedFalsePositives": fp - base_fp,
    }

def choose_gate(rows, ptext, pctx, pham, max_added_fp):
    y = np.asarray([r["y"] for r in rows], dtype=bool)
    base = np.asarray([r["rspam"] for r in rows], dtype=bool)
    hard_ham = np.asarray([
        (not r["y"]) and ("hard_ham" in r["source"]) for r in rows
    ], dtype=bool)
    base_hard_fp = int((hard_ham & base).sum())

    best = None
    for tt in candidates(ptext):
        for tc in candidates(pctx):
            for hv in ham_candidates(pham):
                pred = make_pred(rows, ptext, pctx, pham, tt, tc, hv)
                m = metrics(rows, pred)

                if m["addedFalsePositives"] > max_added_fp:
                    continue

                hard_fp = int((hard_ham & pred).sum())
                if hard_fp > base_hard_fp:
                    continue

                point = {
                    **m,
                    "textThreshold": float(tt),
                    "contextThreshold": float(tc),
                    "hamVetoThreshold": float(hv),
                    "hardHamFalsePositives": hard_fp,
                    "baseHardHamFalsePositives": base_hard_fp,
                }

                if (
                    best is None
                    or point["recall"] > best["recall"]
                    or (
                        point["recall"] == best["recall"]
                        and point["addedFalsePositives"] < best["addedFalsePositives"]
                    )
                ):
                    best = point

    if best is None:
        pred = base.copy()
        m = metrics(rows, pred)
        best = {
            **m,
            "textThreshold": 1.000001,
            "contextThreshold": 1.000001,
            "hamVetoThreshold": 0.0,
            "hardHamFalsePositives": base_hard_fp,
            "baseHardHamFalsePositives": base_hard_fp,
        }
    return best

def evaluate(rows, ptext, pctx, pham, gate):
    pred = make_pred(
        rows, ptext, pctx, pham,
        gate["textThreshold"],
        gate["contextThreshold"],
        gate["hamVetoThreshold"],
    )
    out = metrics(rows, pred)

    hard_ham = np.asarray([
        (not r["y"]) and ("hard_ham" in r["source"]) for r in rows
    ], dtype=bool)
    out["hardHamFalsePositives"] = int((hard_ham & pred).sum())
    return out

def review(rows, ptext, pctx, pham):
    residual = [i for i, r in enumerate(rows) if not r["rspam"]]
    budget = min(max(1, math.ceil(len(rows) * .01)), len(residual))
    risk = np.cbrt(
        np.clip(ptext, 1e-9, 1)
        * np.clip(pctx, 1e-9, 1)
        * np.clip(1.0 - pham, 1e-9, 1)
    )
    top = sorted(residual, key=lambda i: risk[i], reverse=True)[:budget]
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
    print("v6 goal: rescue false negatives with zero added validation FP", flush=True)

    b.reset_bayes()
    b.learn(train)

    tr = b.scan_many(train, "v6-train")
    va = b.scan_many(val, "v6-val")
    te = b.scan_many(test, "v6-test")

    base_test = b.base_metrics(te)

    xt, xv, xe, ctx_t, ctx_v, ctx_e = v3.matrices(tr, va, te)

    residual_idx = [i for i, r in enumerate(tr) if not r["rspam"]]
    residual_rows = [tr[i] for i in residual_idx]
    xt_res = xt[residual_idx]
    ctx_res = ctx_t[residual_idx]

    # First pass.
    text0 = fit_spam_models(xt_res, residual_rows)
    ctx0 = fit_spam_models(ctx_res, residual_rows)
    ptext_train0 = avg_prob(text0, xt_res)
    pctx_train0 = avg_prob(ctx0, ctx_res)

    # Hard-negative mining: normal mail that still looks spammy receives much
    # more weight on the second pass.
    mined = hard_negative_weights(residual_rows, ptext_train0, pctx_train0)

    text_models = fit_spam_models(xt_res, residual_rows, mined)
    ctx_models = fit_spam_models(ctx_res, residual_rows, mined)
    ham_models = v5.fit_ham_ensemble(ctx_res, residual_rows)

    ptext_val = avg_prob(text_models, xv)
    pctx_val = avg_prob(ctx_models, ctx_v)
    pham_val = avg_prob(ham_models, ctx_v)

    ptext_test = avg_prob(text_models, xe)
    pctx_test = avg_prob(ctx_models, ctx_e)
    pham_test = avg_prob(ham_models, ctx_e)

    zero_gate = choose_gate(va, ptext_val, pctx_val, pham_val, max_added_fp=0)
    one_gate = choose_gate(va, ptext_val, pctx_val, pham_val, max_added_fp=1)

    zero_test = evaluate(te, ptext_test, pctx_test, pham_test, zero_gate)
    one_test = evaluate(te, ptext_test, pctx_test, pham_test, one_gate)

    result = {
        "version": "v6-residual-hard-negative",
        "dataset": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "testSpam": base_test["spamTotal"],
            "testHam": base_test["hamTotal"],
        },
        "rspamdBayes": base_test,
        "zeroAddedFpValidationGate": zero_gate,
        "zeroAddedFpTest": zero_test,
        "oneAddedFpValidationGate": one_gate,
        "oneAddedFpTest": one_test,
        "review1pctResidual": review(te, ptext_test, pctx_test, pham_test),
    }

    (REPORTS / "v6-benchmark.json").write_text(json.dumps(result, indent=2))

    md = [
        "# MailGuard v6 residual hard-negative benchmark",
        "",
        f"Final untouched test: {base_test['spamTotal']} spam + {base_test['hamTotal']} ham.",
        "",
        "| Mode | Spam recall | Spam rescued over base | FP total | Added FP |",
        "|---|---:|---:|---:|---:|",
        f"| Rspamd + Bayes | {base_test['recall']:.2%} | 0 | "
        f"{base_test['falsePositives']}/{base_test['hamTotal']} | 0 |",
        f"| v6 zero-added-FP gate | {zero_test['recall']:.2%} | "
        f"{zero_test['rescuedSpam']} | {zero_test['falsePositives']}/{zero_test['hamTotal']} | "
        f"{zero_test['addedFalsePositives']} |",
        f"| v6 one-added-FP gate | {one_test['recall']:.2%} | "
        f"{one_test['rescuedSpam']} | {one_test['falsePositives']}/{one_test['hamTotal']} | "
        f"{one_test['addedFalsePositives']} |",
        "",
        "## 1% residual review",
        "",
        f"Random mean {result['review1pctResidual']['randomSpamFoundMean']:.2f}, "
        f"top-risk {result['review1pctResidual']['topRiskSpamFound']}, "
        f"lift {result['review1pctResidual']['lift']}.",
        "",
        "## Validation gates",
        "",
        f"Zero-added-FP: {zero_gate}",
        f"One-added-FP: {one_gate}",
    ]
    text = "\n".join(md) + "\n"
    (REPORTS / "v6-benchmark.md").write_text(text)
    print(text, flush=True)

if __name__ == "__main__":
    main()
