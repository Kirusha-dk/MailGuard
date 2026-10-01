#!/usr/bin/env python3
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

import benchmark_improved as b
import benchmark_v3 as v3
import benchmark_v5 as v5
import benchmark_v7_nn as n
import benchmark_v8_nn_veto as v8
import benchmark_v9_stack as v9

REPORTS = Path("reports")
SEED = 20261001

def fit_logistic(rows, x, indices):
    keep = [i for i in indices if not rows[i]["rspam"]]
    y = np.asarray([rows[i]["y"] for i in keep], dtype=np.int32)
    w = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)

    for j, i in enumerate(keep):
        row = rows[i]
        if not row["y"] and "hard_ham" in row["source"]:
            w[j] *= 14.0
        if not row["y"] and row["rscore"] >= 4.0:
            w[j] *= 7.0
        if row["y"] and row["rscore"] <= 4.0:
            w[j] *= 1.5

    model = LogisticRegression(
        C=0.22,
        penalty="l2",
        solver="lbfgs",
        max_iter=4000,
        random_state=SEED,
    )
    model.fit(x[keep], y, sample_weight=w)
    return model

def fit_tree(rows, x, indices):
    keep = [i for i in indices if not rows[i]["rspam"]]
    y = np.asarray([rows[i]["y"] for i in keep], dtype=np.int32)
    w = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)

    for j, i in enumerate(keep):
        row = rows[i]
        if not row["y"] and "hard_ham" in row["source"]:
            w[j] *= 16.0
        if not row["y"] and row["rscore"] >= 4.0:
            w[j] *= 8.0
        if row["y"] and row["rscore"] <= 4.0:
            w[j] *= 1.4

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.055,
        max_iter=180,
        max_leaf_nodes=9,
        min_samples_leaf=10,
        l2_regularization=2.0,
        random_state=SEED + 77,
    )
    model.fit(x[keep], y, sample_weight=w)
    return model

def metrics(rows, prediction):
    y = np.asarray([r["y"] for r in rows], dtype=bool)
    base = np.asarray([r["rspam"] for r in rows], dtype=bool)

    spam_total = int(y.sum())
    ham_total = int((~y).sum())
    tp = int((y & prediction).sum())
    fp = int(((~y) & prediction).sum())
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

def consensus_features(plog, ptree, ptext, pctx, pham_lin, pspam_nn, pham_nn):
    plog = np.asarray(plog)
    ptree = np.asarray(ptree)
    ptext = np.asarray(ptext)
    pctx = np.asarray(pctx)
    pham_lin = np.asarray(pham_lin)
    pspam_nn = np.asarray(pspam_nn)
    pham_nn = np.asarray(pham_nn)

    meta_pair = np.sqrt(np.clip(plog, 1e-9, 1) * np.clip(ptree, 1e-9, 1))
    ham_conf = np.sqrt(
        np.clip(1.0 - pham_lin, 1e-9, 1) *
        np.clip(1.0 - pham_nn, 1e-9, 1)
    )
    spam_experts = np.vstack([ptext, pctx, pspam_nn]).T
    spam_median = np.median(spam_experts, axis=1)
    spam_min = spam_experts.min(axis=1)

    precision_score = np.power(
        np.clip(meta_pair, 1e-9, 1)
        * np.clip(ham_conf, 1e-9, 1)
        * np.clip(spam_median, 1e-9, 1),
        1.0 / 3.0,
    )

    return {
        "meta": meta_pair,
        "ham": ham_conf,
        "median": spam_median,
        "minimum": spam_min,
        "score": precision_score,
    }

def grid(values, fixed):
    q = np.quantile(values, np.linspace(0.0, 1.0, 28))
    return np.unique(np.concatenate([np.asarray(fixed, dtype=float), q]))

def choose_gate(rows, signals, indices, max_added_fp=None, target_recall=None):
    subrows = [rows[i] for i in indices]
    base = np.asarray([r["rspam"] for r in subrows], dtype=bool)
    hard = np.asarray([
        (not r["y"]) and ("hard_ham" in r["source"])
        for r in subrows
    ], dtype=bool)
    base_hard_fp = int((hard & base).sum())

    score = signals["score"][indices]
    ham = signals["ham"][indices]
    meta = signals["meta"][indices]
    median = signals["median"][indices]
    minimum = signals["minimum"][indices]

    score_grid = grid(score, [.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.93,.95,.97,.98,.99,.995,.999,1.000001])
    ham_grid = grid(ham, [.20,.30,.40,.50,.60,.70,.75,.80,.85,.90,.93,.95,.97,.98,.99])
    meta_grid = grid(meta, [.20,.30,.40,.50,.60,.70,.75,.80,.85,.90,.93,.95,.97,.98,.99])
    median_grid = np.asarray([0.30,0.40,0.50,0.60,0.70,0.80,0.90])

    best = None

    # Precision gate: final score plus explicit agreement/veto conditions.
    for ts in score_grid:
        m_score = score >= ts
        for th in ham_grid:
            m_ham = ham >= th
            if not np.any(m_score & m_ham):
                continue
            for tm in meta_grid:
                base_mask = m_score & m_ham & (meta >= tm)
                if not np.any(base_mask):
                    continue
                for tmed in median_grid:
                    add = base_mask & (median >= tmed)

                    # Ultra-confidence rescue: all three spam experts agree very strongly
                    # and both ham models are strongly against ham.
                    ultra = (
                        (minimum >= .97) &
                        (meta >= .97) &
                        (ham >= .97)
                    )

                    pred = base | add | ultra
                    current = metrics(subrows, pred)
                    hard_fp = int((hard & pred).sum())
                    added_hard_fp = hard_fp - base_hard_fp

                    if max_added_fp is not None:
                        if current["addedFalsePositives"] > max_added_fp:
                            continue
                        if added_hard_fp > 0:
                            continue

                    if target_recall is not None and current["recall"] < target_recall:
                        continue

                    point = {
                        **current,
                        "scoreThreshold": float(ts),
                        "hamConfidenceThreshold": float(th),
                        "metaThreshold": float(tm),
                        "medianSpamThreshold": float(tmed),
                        "hardHamFalsePositives": hard_fp,
                        "addedHardHamFalsePositives": added_hard_fp,
                    }

                    if best is None:
                        best = point
                        continue

                    if target_recall is not None:
                        key = (
                            -point["falsePositives"],
                            -point["addedHardHamFalsePositives"],
                            point["recall"],
                            point["scoreThreshold"],
                        )
                        best_key = (
                            -best["falsePositives"],
                            -best["addedHardHamFalsePositives"],
                            best["recall"],
                            best["scoreThreshold"],
                        )
                    else:
                        key = (
                            point["recall"],
                            -point["addedFalsePositives"],
                            -point["addedHardHamFalsePositives"],
                            point["scoreThreshold"],
                        )
                        best_key = (
                            best["recall"],
                            -best["addedFalsePositives"],
                            -best["addedHardHamFalsePositives"],
                            best["scoreThreshold"],
                        )

                    if key > best_key:
                        best = point

    if best is None:
        base_metrics = metrics(subrows, base.copy())
        best = {
            **base_metrics,
            "scoreThreshold": 1.000001,
            "hamConfidenceThreshold": 1.0,
            "metaThreshold": 1.0,
            "medianSpamThreshold": 1.0,
            "hardHamFalsePositives": base_hard_fp,
            "addedHardHamFalsePositives": 0,
        }
    return best

def evaluate(rows, signals, gate):
    base = np.asarray([r["rspam"] for r in rows], dtype=bool)
    add = (
        (signals["score"] >= gate["scoreThreshold"]) &
        (signals["ham"] >= gate["hamConfidenceThreshold"]) &
        (signals["meta"] >= gate["metaThreshold"]) &
        (signals["median"] >= gate["medianSpamThreshold"])
    )
    ultra = (
        (signals["minimum"] >= .97) &
        (signals["meta"] >= .97) &
        (signals["ham"] >= .97)
    )
    pred = base | add | ultra
    return metrics(rows, pred)

def review(rows, score):
    residual = [i for i, r in enumerate(rows) if not r["rspam"]]
    budget = min(max(1, math.ceil(len(rows) * .01)), len(residual))
    top = sorted(residual, key=lambda i: score[i], reverse=True)[:budget]
    top_spam = sum(rows[i]["y"] for i in top)

    rnd = random.Random(SEED)
    hits = []
    for _ in range(2000):
        sample = rnd.sample(residual, budget)
        hits.append(sum(rows[i]["y"] for i in sample))
    mean = float(np.mean(hits))
    return {
        "budget": budget,
        "topRiskSpamFound": int(top_spam),
        "randomSpamFoundMean": mean,
        "lift": top_spam / mean if mean else None,
    }

def main():
    REPORTS.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    b.wait_rspamd()

    groups = b.prepare()
    train, val, test = v3.build_splits(groups)

    print("TRAIN", len(train), "VAL", len(val), "TEST", len(test), flush=True)
    print("v10: precision consensus stack with explicit neural ham veto", flush=True)

    b.reset_bayes()
    b.learn(train)
    tr = b.scan_many(train, "v10-train")
    va = b.scan_many(val, "v10-val")
    te = b.scan_many(test, "v10-test")
    base = b.base_metrics(te)

    xt, xv, xe, ctx_t, ctx_v, ctx_e = v3.matrices(tr, va, te)
    residual_idx = [i for i, r in enumerate(tr) if not r["rspam"]]
    residual_rows = [tr[i] for i in residual_idx]

    # Classical experts.
    text_models = v3.fit_ensemble(xt[residual_idx], residual_rows)
    context_models = v3.fit_ensemble(ctx_t[residual_idx], residual_rows)
    ham_linear_models = v5.fit_ham_ensemble(ctx_t[residual_idx], residual_rows)

    ptext_val = v3.ensemble_predict(text_models, xv)
    pctx_val = v3.ensemble_predict(context_models, ctx_v)
    pham_lin_val = v5.avg_prob(ham_linear_models, ctx_v)

    ptext_test = v3.ensemble_predict(text_models, xe)
    pctx_test = v3.ensemble_predict(context_models, ctx_e)
    pham_lin_test = v5.avg_prob(ham_linear_models, ctx_e)

    # Neural experts.
    residual_val = [r for r in va if not r["rspam"]]
    spam_train_ds = n.MailDataset(residual_rows, n.sample_weights(residual_rows))
    spam_val_ds = n.MailDataset(residual_val, n.sample_weights(residual_val))

    ham_train_rows = v8.flipped_rows(residual_rows)
    ham_val_rows = v8.flipped_rows(residual_val)
    ham_train_ds = n.MailDataset(ham_train_rows, v8.ham_weights(residual_rows))
    ham_val_ds = n.MailDataset(ham_val_rows, v8.ham_weights(residual_val))

    full_val_ds = n.MailDataset(va)
    full_test_ds = n.MailDataset(te)

    spam_nn_models = [
        n.train_one(spam_train_ds, spam_val_ds, SEED + 200 + i * 97)
        for i in range(3)
    ]
    ham_nn_models = [
        n.train_one(ham_train_ds, ham_val_ds, SEED + 5200 + i * 97)
        for i in range(3)
    ]

    pspam_nn_val = v8.predict(spam_nn_models, full_val_ds)
    pham_nn_val = v8.predict(ham_nn_models, full_val_ds)
    pspam_nn_test = v8.predict(spam_nn_models, full_test_ds)
    pham_nn_test = v8.predict(ham_nn_models, full_test_ds)

    xmeta_val = v9.meta_features(
        va, ptext_val, pctx_val, pham_lin_val, pspam_nn_val, pham_nn_val
    )
    xmeta_test = v9.meta_features(
        te, ptext_test, pctx_test, pham_lin_test, pspam_nn_test, pham_nn_test
    )

    meta_idx, gate_idx = v9.split_meta(va)

    logistic = fit_logistic(va, xmeta_val, meta_idx)
    tree = fit_tree(va, xmeta_val, meta_idx)

    plog_val = logistic.predict_proba(xmeta_val)[:, 1]
    ptree_val = tree.predict_proba(xmeta_val)[:, 1]
    plog_test = logistic.predict_proba(xmeta_test)[:, 1]
    ptree_test = tree.predict_proba(xmeta_test)[:, 1]

    val_signals = consensus_features(
        plog_val, ptree_val, ptext_val, pctx_val,
        pham_lin_val, pspam_nn_val, pham_nn_val
    )
    test_signals = consensus_features(
        plog_test, ptree_test, ptext_test, pctx_test,
        pham_lin_test, pspam_nn_test, pham_nn_test
    )

    gates = {
        "zero": choose_gate(va, val_signals, gate_idx, max_added_fp=0),
        "one": choose_gate(va, val_signals, gate_idx, max_added_fp=1),
        "target95": choose_gate(va, val_signals, gate_idx, target_recall=.95),
        "target97": choose_gate(va, val_signals, gate_idx, target_recall=.97),
        "target98": choose_gate(va, val_signals, gate_idx, target_recall=.98),
    }

    tests = {
        name: evaluate(te, test_signals, gate)
        for name, gate in gates.items()
    }

    result = {
        "version": "v10-precision-consensus",
        "dataset": {
            "train": len(train),
            "validation": len(val),
            "metaValidation": len(meta_idx),
            "gateValidation": len(gate_idx),
            "test": len(test),
            "testSpam": base["spamTotal"],
            "testHam": base["hamTotal"],
        },
        "architecture": {
            "linearSpamExperts": 6,
            "linearHamExperts": 3,
            "neuralSpamExperts": 3,
            "neuralHamExperts": 3,
            "metaLogistic": True,
            "metaHistogramGradientBoosting": True,
            "explicitHamVeto": True,
            "testUntouched": True,
        },
        "rspamdBayes": base,
        "validationGates": gates,
        "test": tests,
        "review1pctResidual": review(te, test_signals["score"]),
    }

    (REPORTS / "v10-benchmark.json").write_text(json.dumps(result, indent=2))

    md = [
        "# MailGuard v10 precision consensus benchmark",
        "",
        f"Final untouched test: {base['spamTotal']} spam + {base['hamTotal']} ham.",
        "",
        "| Mode | Spam recall | Spam rescued over base | FP total | Added FP |",
        "|---|---:|---:|---:|---:|",
        f"| Rspamd + Bayes | {base['recall']:.2%} | 0 | "
        f"{base['falsePositives']}/{base['hamTotal']} | 0 |",
    ]

    labels = {
        "zero": "v10 zero-added-FP gate",
        "one": "v10 one-added-FP gate",
        "target95": "v10 validation target 95%",
        "target97": "v10 validation target 97%",
        "target98": "v10 validation target 98%",
    }
    for name in ("zero", "one", "target95", "target97", "target98"):
        item = tests[name]
        md.append(
            f"| {labels[name]} | {item['recall']:.2%} | "
            f"{item['rescuedSpam']} | "
            f"{item['falsePositives']}/{item['hamTotal']} | "
            f"{item['addedFalsePositives']} |"
        )

    md += [
        "",
        "## 1% residual review",
        "",
        f"Random mean {result['review1pctResidual']['randomSpamFoundMean']:.2f}, "
        f"top-risk {result['review1pctResidual']['topRiskSpamFound']}, "
        f"lift {result['review1pctResidual']['lift']}.",
        "",
        "## Validation gates",
        "",
    ]
    for name in ("zero", "one", "target95", "target97", "target98"):
        md.append(f"{name}: {gates[name]}")

    text = "\n".join(md) + "\n"
    (REPORTS / "v10-benchmark.md").write_text(text)
    print(text, flush=True)

if __name__ == "__main__":
    main()
