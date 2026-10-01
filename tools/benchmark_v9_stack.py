#!/usr/bin/env python3
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

import benchmark_improved as b
import benchmark_v3 as v3
import benchmark_v5 as v5
import benchmark_v7_nn as n
import benchmark_v8_nn_veto as v8

REPORTS = Path("reports")
SEED = 20261001

def split_meta(rows):
    """Split validation into meta-train and gate-selection without touching test."""
    by_key = {}
    for i, row in enumerate(rows):
        key = (int(row["y"]), row["source"])
        by_key.setdefault(key, []).append(i)

    rnd = random.Random(SEED + 9000)
    meta = []
    gate = []
    for key, idxs in by_key.items():
        idxs = list(idxs)
        rnd.shuffle(idxs)
        cut = max(1, len(idxs) // 2)
        meta.extend(idxs[:cut])
        gate.extend(idxs[cut:])

    meta.sort()
    gate.sort()
    return np.asarray(meta, dtype=np.int64), np.asarray(gate, dtype=np.int64)

def clip_logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))

def meta_features(rows, ptext, pctx, pham_lin, pspam_nn, pham_nn):
    ptext = np.asarray(ptext)
    pctx = np.asarray(pctx)
    pham_lin = np.asarray(pham_lin)
    pspam_nn = np.asarray(pspam_nn)
    pham_nn = np.asarray(pham_nn)

    spam_triplet = np.vstack([ptext, pctx, pspam_nn]).T
    ham_conf = np.sqrt(
        np.clip(1.0 - pham_lin, 1e-9, 1.0) *
        np.clip(1.0 - pham_nn, 1e-9, 1.0)
    )
    spam_geo = np.cbrt(
        np.clip(ptext, 1e-9, 1.0) *
        np.clip(pctx, 1e-9, 1.0) *
        np.clip(pspam_nn, 1e-9, 1.0)
    )

    rscore = []
    ratio = []
    for row in rows:
        req = row["required"] if row["required"] else 6.0
        rscore.append(max(-3.0, min(3.0, row["rscore"] / 15.0)))
        ratio.append(max(-3.0, min(3.0, row["rscore"] / req)))

    return np.column_stack([
        ptext,
        pctx,
        1.0 - pham_lin,
        pspam_nn,
        1.0 - pham_nn,
        spam_triplet.min(axis=1),
        spam_triplet.max(axis=1),
        spam_geo,
        ham_conf,
        spam_geo * ham_conf,
        pspam_nn * (1.0 - pham_nn),
        pctx * (1.0 - pham_lin),
        np.abs(ptext - pctx),
        np.abs(pspam_nn - pctx),
        clip_logit(ptext),
        clip_logit(pctx),
        clip_logit(pspam_nn),
        clip_logit(1.0 - pham_lin),
        clip_logit(1.0 - pham_nn),
        np.asarray(rscore),
        np.asarray(ratio),
    ]).astype(np.float64)

def meta_weights(rows):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    w = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)
    for i, r in enumerate(rows):
        if not r["y"] and "hard_ham" in r["source"]:
            w[i] *= 10.0
        if not r["y"] and r["rscore"] >= 4.0:
            w[i] *= 5.0
        if r["y"] and r["rscore"] <= 4.0:
            w[i] *= 1.5
    return w

def train_meta(rows, x, idx):
    keep = [i for i in idx if not rows[i]["rspam"]]
    xtrain = x[keep]
    ytrain = np.asarray([rows[i]["y"] for i in keep], dtype=np.int32)
    wtrain = meta_weights([rows[i] for i in keep])

    clf = LogisticRegression(
        C=0.35,
        penalty="l2",
        solver="lbfgs",
        max_iter=3000,
        random_state=SEED,
    )
    clf.fit(xtrain, ytrain, sample_weight=wtrain)
    return clf

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

def threshold_candidates(values):
    fixed = np.asarray([
        0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
        0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99,
        0.995, 0.999, 1.000001
    ])
    return np.unique(np.concatenate([fixed, np.asarray(values)]))[::-1]

def choose_gate(rows, probabilities, indices, max_added_fp=None, target_recall=None):
    subrows = [rows[i] for i in indices]
    probs = probabilities[indices]
    base = np.asarray([r["rspam"] for r in subrows], dtype=bool)
    hard = np.asarray([
        (not r["y"]) and ("hard_ham" in r["source"])
        for r in subrows
    ], dtype=bool)
    base_hard_fp = int((hard & base).sum())

    best = None
    for threshold in threshold_candidates(probs):
        pred = base | (probs >= threshold)
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
            "threshold": float(threshold),
            "hardHamFalsePositives": hard_fp,
            "addedHardHamFalsePositives": added_hard_fp,
        }

        if best is None:
            best = point
            continue

        if target_recall is not None:
            # Hit the requested recall with the fewest false positives.
            key = (
                -point["falsePositives"],
                point["recall"],
                point["threshold"],
            )
            best_key = (
                -best["falsePositives"],
                best["recall"],
                best["threshold"],
            )
        else:
            key = (
                point["recall"],
                -point["addedFalsePositives"],
                point["threshold"],
            )
            best_key = (
                best["recall"],
                -best["addedFalsePositives"],
                best["threshold"],
            )

        if key > best_key:
            best = point

    if best is None:
        pred = base.copy()
        best = {
            **metrics(subrows, pred),
            "threshold": 1.000001,
            "hardHamFalsePositives": base_hard_fp,
            "addedHardHamFalsePositives": 0,
        }
    return best

def evaluate(rows, probabilities, gate):
    base = np.asarray([r["rspam"] for r in rows], dtype=bool)
    pred = base | (probabilities >= gate["threshold"])
    return metrics(rows, pred)

def review(rows, probabilities):
    residual = [i for i, r in enumerate(rows) if not r["rspam"]]
    budget = min(max(1, math.ceil(len(rows) * .01)), len(residual))
    top = sorted(residual, key=lambda i: probabilities[i], reverse=True)[:budget]
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
    print("v9: stacked linear + neural residual ensemble", flush=True)

    b.reset_bayes()
    b.learn(train)
    tr = b.scan_many(train, "v9-train")
    va = b.scan_many(val, "v9-val")
    te = b.scan_many(test, "v9-test")

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
        n.train_one(spam_train_ds, spam_val_ds, SEED + 100 + i * 97)
        for i in range(3)
    ]
    ham_nn_models = [
        n.train_one(ham_train_ds, ham_val_ds, SEED + 5100 + i * 97)
        for i in range(3)
    ]

    pspam_nn_val = v8.predict(spam_nn_models, full_val_ds)
    pham_nn_val = v8.predict(ham_nn_models, full_val_ds)
    pspam_nn_test = v8.predict(spam_nn_models, full_test_ds)
    pham_nn_test = v8.predict(ham_nn_models, full_test_ds)

    xmeta_val = meta_features(
        va, ptext_val, pctx_val, pham_lin_val, pspam_nn_val, pham_nn_val
    )
    xmeta_test = meta_features(
        te, ptext_test, pctx_test, pham_lin_test, pspam_nn_test, pham_nn_test
    )

    meta_idx, gate_idx = split_meta(va)
    meta_model = train_meta(va, xmeta_val, meta_idx)
    pmeta_val = meta_model.predict_proba(xmeta_val)[:, 1]
    pmeta_test = meta_model.predict_proba(xmeta_test)[:, 1]

    gates = {
        "zero": choose_gate(va, pmeta_val, gate_idx, max_added_fp=0),
        "one": choose_gate(va, pmeta_val, gate_idx, max_added_fp=1),
        "target98": choose_gate(va, pmeta_val, gate_idx, target_recall=.98),
    }

    tests = {name: evaluate(te, pmeta_test, gate) for name, gate in gates.items()}

    result = {
        "version": "v9-stacked-residual",
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
            "linearTextModels": 3,
            "linearContextModels": 3,
            "linearHamModels": 3,
            "neuralSpamModels": 3,
            "neuralHamModels": 3,
            "stacker": "LogisticRegression",
            "metaAndGateValidationSeparated": True,
            "testUntouched": True,
        },
        "rspamdBayes": base,
        "validationGates": gates,
        "test": tests,
        "review1pctResidual": review(te, pmeta_test),
    }

    (REPORTS / "v9-benchmark.json").write_text(json.dumps(result, indent=2))

    md = [
        "# MailGuard v9 stacked residual benchmark",
        "",
        f"Final untouched test: {base['spamTotal']} spam + {base['hamTotal']} ham.",
        "",
        "| Mode | Spam recall | Spam rescued over base | FP total | Added FP |",
        "|---|---:|---:|---:|---:|",
        f"| Rspamd + Bayes | {base['recall']:.2%} | 0 | "
        f"{base['falsePositives']}/{base['hamTotal']} | 0 |",
    ]
    labels = {
        "zero": "v9 zero-added-FP gate",
        "one": "v9 one-added-FP gate",
        "target98": "v9 validation target 98%",
    }
    for name in ("zero", "one", "target98"):
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
        f"Zero: {gates['zero']}",
        f"One: {gates['one']}",
        f"Target98: {gates['target98']}",
    ]
    text = "\n".join(md) + "\n"
    (REPORTS / "v9-benchmark.md").write_text(text)
    print(text, flush=True)

if __name__ == "__main__":
    main()
