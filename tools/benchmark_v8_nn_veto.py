#!/usr/bin/env python3
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import benchmark_improved as b
import benchmark_v3 as v3
import benchmark_v7_nn as n

REPORTS = Path("reports")
SEED = 20261001

def ham_weights(rows):
    labels = np.asarray([0 if r["y"] else 1 for r in rows], dtype=np.int32)
    positives = max(1, int(labels.sum()))
    negatives = max(1, len(labels) - positives)
    total = max(1, len(labels))

    pos_weight = total / (2.0 * positives)
    neg_weight = total / (2.0 * negatives)
    weights = np.where(labels == 1, pos_weight, neg_weight).astype(np.float32)

    for i, r in enumerate(rows):
        if not r["y"] and "hard_ham" in r["source"]:
            weights[i] *= 14.0
        if not r["y"] and r["rscore"] >= 4.0:
            weights[i] *= 6.0
        if r["y"] and r["rscore"] <= 4.0:
            weights[i] *= 1.5
    return weights

def flipped_rows(rows):
    return [{**r, "y": 0 if r["y"] else 1} for r in rows]

def predict(models, dataset):
    loader = DataLoader(dataset, batch_size=192, shuffle=False, collate_fn=n.collate)
    outputs = []
    for model in models:
        current = []
        with torch.no_grad():
            for ids, offsets, numeric, labels, weights in loader:
                current.append(torch.sigmoid(model(ids, offsets, numeric)).cpu().numpy())
        outputs.append(np.concatenate(current))
    return np.mean(outputs, axis=0)

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

def threshold_grid(values):
    fixed = np.asarray([
        .25, .35, .45, .50, .55, .60, .65, .70, .75, .80,
        .85, .90, .93, .95, .97, .98, .99, .995, .999, 1.000001
    ])
    quantiles = np.quantile(values, np.linspace(.25, 1.0, 40))
    return np.unique(np.concatenate([fixed, quantiles]))

def veto_grid(values):
    fixed = np.asarray([
        .01, .02, .03, .05, .08, .10, .15, .20, .25, .30,
        .35, .40, .50, .60, .70, .80, .90, .99
    ])
    quantiles = np.quantile(values, np.linspace(0.0, .90, 30))
    return np.unique(np.concatenate([fixed, quantiles]))

def choose_gate(rows, pspam, pham, max_added_fp, max_added_hard_fp):
    y = np.asarray([r["y"] for r in rows], dtype=bool)
    base = np.asarray([r["rspam"] for r in rows], dtype=bool)
    hard = np.asarray([
        (not r["y"]) and ("hard_ham" in r["source"])
        for r in rows
    ], dtype=bool)

    base_hard_fp = int((hard & base).sum())
    best = None

    # Two independent neural opinions are required:
    # the spam network must be confident and the ham-veto network must not be.
    for spam_threshold in threshold_grid(pspam):
        spam_mask = pspam >= spam_threshold
        for ham_threshold in veto_grid(pham):
            pred = base | (spam_mask & (pham <= ham_threshold))
            current = metrics(rows, pred)
            hard_fp = int((hard & pred).sum())
            added_hard_fp = hard_fp - base_hard_fp

            if current["addedFalsePositives"] > max_added_fp:
                continue
            if added_hard_fp > max_added_hard_fp:
                continue

            point = {
                **current,
                "spamThreshold": float(spam_threshold),
                "hamVetoThreshold": float(ham_threshold),
                "hardHamFalsePositives": hard_fp,
                "addedHardHamFalsePositives": added_hard_fp,
            }

            key = (
                point["recall"],
                -point["addedFalsePositives"],
                -point["addedHardHamFalsePositives"],
                point["spamThreshold"],
                -point["hamVetoThreshold"],
            )
            if best is None:
                best = point
            else:
                best_key = (
                    best["recall"],
                    -best["addedFalsePositives"],
                    -best["addedHardHamFalsePositives"],
                    best["spamThreshold"],
                    -best["hamVetoThreshold"],
                )
                if key > best_key:
                    best = point

    if best is None:
        base_metrics = metrics(rows, base.copy())
        best = {
            **base_metrics,
            "spamThreshold": 1.000001,
            "hamVetoThreshold": 0.0,
            "hardHamFalsePositives": base_hard_fp,
            "addedHardHamFalsePositives": 0,
        }
    return best

def evaluate(rows, pspam, pham, gate):
    base = np.asarray([r["rspam"] for r in rows], dtype=bool)
    pred = base | (
        (pspam >= gate["spamThreshold"]) &
        (pham <= gate["hamVetoThreshold"])
    )
    return metrics(rows, pred), pred

def review(rows, pspam, pham):
    residual = [i for i, r in enumerate(rows) if not r["rspam"]]
    budget = min(max(1, math.ceil(len(rows) * .01)), len(residual))
    risk = np.sqrt(
        np.clip(pspam, 1e-9, 1.0) *
        np.clip(1.0 - pham, 1e-9, 1.0)
    )
    top = sorted(residual, key=lambda i: risk[i], reverse=True)[:budget]
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
    print("v8: neural spam ensemble + independent neural ham veto", flush=True)

    b.reset_bayes()
    b.learn(train)

    tr = b.scan_many(train, "v8-train")
    va = b.scan_many(val, "v8-val")
    te = b.scan_many(test, "v8-test")
    base_test = b.base_metrics(te)

    residual_train = [r for r in tr if not r["rspam"]]
    residual_val = [r for r in va if not r["rspam"]]

    spam_train_ds = n.MailDataset(residual_train, n.sample_weights(residual_train))
    spam_val_ds = n.MailDataset(residual_val, n.sample_weights(residual_val))

    ham_train_rows = flipped_rows(residual_train)
    ham_val_rows = flipped_rows(residual_val)
    ham_train_ds = n.MailDataset(ham_train_rows, ham_weights(residual_train))
    ham_val_ds = n.MailDataset(ham_val_rows, ham_weights(residual_val))

    full_val_ds = n.MailDataset(va)
    full_test_ds = n.MailDataset(te)

    spam_models = [
        n.train_one(spam_train_ds, spam_val_ds, SEED + i * 97)
        for i in range(3)
    ]
    ham_models = [
        n.train_one(ham_train_ds, ham_val_ds, SEED + 5000 + i * 97)
        for i in range(3)
    ]

    pspam_val = predict(spam_models, full_val_ds)
    pham_val = predict(ham_models, full_val_ds)
    pspam_test = predict(spam_models, full_test_ds)
    pham_test = predict(ham_models, full_test_ds)

    gates = {
        "zero": choose_gate(va, pspam_val, pham_val, 0, 0),
        "one": choose_gate(va, pspam_val, pham_val, 1, 0),
        "three": choose_gate(va, pspam_val, pham_val, 3, 1),
    }

    tests = {}
    preds = {}
    for name, gate in gates.items():
        tests[name], preds[name] = evaluate(te, pspam_test, pham_test, gate)

    result = {
        "version": "v8-neural-ham-veto",
        "dataset": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "testSpam": base_test["spamTotal"],
            "testHam": base_test["hamTotal"],
        },
        "architecture": {
            "spamEnsemble": 3,
            "hamVetoEnsemble": 3,
            "residualOnlyTraining": True,
            "twoIndependentNeuralSignals": True,
        },
        "rspamdBayes": base_test,
        "validationGates": gates,
        "test": tests,
        "review1pctResidual": review(te, pspam_test, pham_test),
    }

    (REPORTS / "v8-benchmark.json").write_text(json.dumps(result, indent=2))

    md = [
        "# MailGuard v8 neural ham-veto benchmark",
        "",
        f"Final untouched test: {base_test['spamTotal']} spam + {base_test['hamTotal']} ham.",
        "",
        "| Mode | Spam recall | Spam rescued over base | FP total | Added FP |",
        "|---|---:|---:|---:|---:|",
        f"| Rspamd + Bayes | {base_test['recall']:.2%} | 0 | "
        f"{base_test['falsePositives']}/{base_test['hamTotal']} | 0 |",
    ]

    labels = {
        "zero": "v8 zero-added-FP validation gate",
        "one": "v8 one-added-FP validation gate",
        "three": "v8 three-added-FP validation gate",
    }
    for key in ("zero", "one", "three"):
        item = tests[key]
        md.append(
            f"| {labels[key]} | {item['recall']:.2%} | "
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
        f"Three: {gates['three']}",
    ]

    text = "\n".join(md) + "\n"
    (REPORTS / "v8-benchmark.md").write_text(text)
    print(text, flush=True)

if __name__ == "__main__":
    main()
