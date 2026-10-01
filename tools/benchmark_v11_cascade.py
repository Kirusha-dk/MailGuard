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
import benchmark_v5 as v5
import benchmark_v7_nn as n
import benchmark_v8_nn_veto as v8

REPORTS = Path("reports")
SEED = 20261001


def model_matrix(models, dataset):
    loader = DataLoader(dataset, batch_size=192, shuffle=False, collate_fn=n.collate)
    outputs = []
    for model in models:
        current = []
        with torch.no_grad():
            for ids, offsets, numeric, labels, weights in loader:
                current.append(torch.sigmoid(model(ids, offsets, numeric)).cpu().numpy())
        outputs.append(np.concatenate(current))
    return np.vstack(outputs)


def stats(rows, pred, base_pred=None):
    y = np.asarray([r["y"] for r in rows], dtype=bool)
    if base_pred is None:
        base_pred = np.asarray([r["rspam"] for r in rows], dtype=bool)

    spam_total = int(y.sum())
    ham_total = int((~y).sum())
    tp = int((y & pred).sum())
    fp = int(((~y) & pred).sum())
    base_tp = int((y & base_pred).sum())
    base_fp = int(((~y) & base_pred).sum())

    return {
        "spamTotal": spam_total,
        "hamTotal": ham_total,
        "spamDetected": tp,
        "falsePositives": fp,
        "recall": tp / max(1, spam_total),
        "fpr": fp / max(1, ham_total),
        "rescuedOverStage1": tp - base_tp,
        "addedFalsePositivesOverStage1": fp - base_fp,
    }


def confidence_features(ptext, pctx, pham_lin, spam_matrix, ham_matrix):
    pspam = spam_matrix.mean(axis=0)
    pham = ham_matrix.mean(axis=0)

    signals = np.vstack([
        np.clip(ptext, 1e-9, 1.0),
        np.clip(pctx, 1e-9, 1.0),
        np.clip(pspam, 1e-9, 1.0),
        np.clip(1.0 - pham_lin, 1e-9, 1.0),
        np.clip(1.0 - pham, 1e-9, 1.0),
    ])

    score = np.exp(np.mean(np.log(signals), axis=0))
    consensus = signals.min(axis=0)
    spam_std = spam_matrix.std(axis=0)
    ham_std = ham_matrix.std(axis=0)

    return {
        "score": score,
        "consensus": consensus,
        "pspam": pspam,
        "pham": pham,
        "spamStd": spam_std,
        "hamStd": ham_std,
    }


def grid(values, fixed, qcount=35):
    q = np.quantile(values, np.linspace(0.35, 1.0, qcount))
    return np.unique(np.concatenate([np.asarray(fixed), q]))


def rescue_prediction(stage1, feat, gate):
    rescue = (
        (~stage1)
        & (feat["score"] >= gate["scoreThreshold"])
        & (feat["consensus"] >= gate["consensusThreshold"])
        & (feat["pspam"] >= gate["neuralSpamThreshold"])
        & (feat["pham"] <= gate["neuralHamThreshold"])
        & (feat["spamStd"] <= gate["maxSpamStd"])
        & (feat["hamStd"] <= gate["maxHamStd"])
    )
    return stage1 | rescue


def choose_rescue_gate(rows, stage1, feat, max_added_fp):
    y = np.asarray([r["y"] for r in rows], dtype=bool)
    hard = np.asarray([
        (not r["y"]) and ("hard_ham" in r["source"])
        for r in rows
    ], dtype=bool)

    stage1_hard_fp = int((hard & stage1).sum())
    stage1_fp = int(((~y) & stage1).sum())

    score_grid = grid(feat["score"], [.55,.60,.65,.70,.75,.80,.85,.90,.93,.95,.97,.98,.99,.995,.999,1.000001])
    consensus_grid = grid(feat["consensus"], [.30,.40,.50,.60,.70,.75,.80,.85,.90,.93,.95,.97,.99,1.000001], 24)
    spam_grid = np.asarray([.50,.60,.70,.80,.85,.90,.93,.95,.97,.98,.99,.995,.999])
    ham_grid = np.asarray([.01,.02,.03,.05,.08,.10,.15,.20,.25,.30,.40])
    std_grid = np.asarray([.015,.025,.04,.06,.08,.10,.15,.25])

    best = None

    # Precompute reusable masks; this keeps the 6-D search cheap on a small validation set.
    for st in score_grid:
        m_score = feat["score"] >= st
        for ct in consensus_grid:
            m_cons = m_score & (feat["consensus"] >= ct)
            if not np.any(m_cons & (~stage1)):
                continue
            for nt in spam_grid:
                m_spam = m_cons & (feat["pspam"] >= nt)
                if not np.any(m_spam & (~stage1)):
                    continue
                for ht in ham_grid:
                    m_ham = m_spam & (feat["pham"] <= ht)
                    if not np.any(m_ham & (~stage1)):
                        continue
                    for ss in std_grid:
                        m_ss = m_ham & (feat["spamStd"] <= ss)
                        if not np.any(m_ss & (~stage1)):
                            continue
                        for hs in std_grid:
                            rescue = (
                                (~stage1)
                                & m_ss
                                & (feat["hamStd"] <= hs)
                            )
                            pred = stage1 | rescue
                            fp = int(((~y) & pred).sum())
                            added_fp = fp - stage1_fp
                            if added_fp > max_added_fp:
                                continue

                            hard_fp = int((hard & pred).sum())
                            added_hard = hard_fp - stage1_hard_fp
                            if added_hard > 0:
                                continue

                            cur = stats(rows, pred, stage1)
                            point = {
                                **cur,
                                "scoreThreshold": float(st),
                                "consensusThreshold": float(ct),
                                "neuralSpamThreshold": float(nt),
                                "neuralHamThreshold": float(ht),
                                "maxSpamStd": float(ss),
                                "maxHamStd": float(hs),
                                "hardHamFalsePositives": hard_fp,
                                "addedHardHamFalsePositives": added_hard,
                            }

                            # First maximize extra spam caught. Then prefer fewer FP and
                            # stricter/more stable consensus.
                            key = (
                                point["rescuedOverStage1"],
                                -point["addedFalsePositivesOverStage1"],
                                point["scoreThreshold"],
                                point["consensusThreshold"],
                                point["neuralSpamThreshold"],
                                -point["neuralHamThreshold"],
                                -point["maxSpamStd"],
                                -point["maxHamStd"],
                            )
                            if best is None or key > best[0]:
                                best = (key, point)

    if best is None:
        cur = stats(rows, stage1.copy(), stage1)
        return {
            **cur,
            "scoreThreshold": 1.000001,
            "consensusThreshold": 1.000001,
            "neuralSpamThreshold": 1.000001,
            "neuralHamThreshold": 0.0,
            "maxSpamStd": 0.0,
            "maxHamStd": 0.0,
            "hardHamFalsePositives": stage1_hard_fp,
            "addedHardHamFalsePositives": 0,
        }

    return best[1]


def review(rows, stage1, feat):
    residual = [i for i in range(len(rows)) if not stage1[i]]
    budget = min(max(1, math.ceil(len(rows) * .01)), len(residual))
    ordered = sorted(residual, key=lambda i: feat["score"][i], reverse=True)[:budget]
    top_spam = sum(rows[i]["y"] for i in ordered)

    rnd = random.Random(SEED)
    hits = []
    for _ in range(2000):
        sample = rnd.sample(residual, budget)
        hits.append(sum(rows[i]["y"] for i in sample))

    mean = float(np.mean(hits))
    return {
        "budget": budget,
        "residualCandidates": len(residual),
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
    print("v11: precision linear gate + uncertainty-aware neural rescue", flush=True)

    b.reset_bayes()
    b.learn(train)

    tr = b.scan_many(train, "v11-train")
    va = b.scan_many(val, "v11-val")
    te = b.scan_many(test, "v11-test")
    base = b.base_metrics(te)

    xt, xv, xe, ctx_t, ctx_v, ctx_e = v3.matrices(tr, va, te)
    residual_idx = [i for i, r in enumerate(tr) if not r["rspam"]]
    residual_rows = [tr[i] for i in residual_idx]

    # Stage 1: the best conservative classical gate from v5.
    text_models = v3.fit_ensemble(xt[residual_idx], residual_rows)
    context_models = v3.fit_ensemble(ctx_t[residual_idx], residual_rows)
    ham_linear_models = v5.fit_ham_ensemble(ctx_t[residual_idx], residual_rows)

    ptext_val = v3.ensemble_predict(text_models, xv)
    pctx_val = v3.ensemble_predict(context_models, ctx_v)
    pham_lin_val = v5.avg_prob(ham_linear_models, ctx_v)

    ptext_test = v3.ensemble_predict(text_models, xe)
    pctx_test = v3.ensemble_predict(context_models, ctx_e)
    pham_lin_test = v5.avg_prob(ham_linear_models, ctx_e)

    stage1_gate, _ = v5.choose_gate(va, ptext_val, pctx_val, pham_lin_val)
    stage1_val = v5.predict_gate(
        va, ptext_val, pctx_val, pham_lin_val,
        stage1_gate["textThreshold"],
        stage1_gate["contextThreshold"],
        stage1_gate["hamVetoThreshold"],
    )
    stage1_test = v5.predict_gate(
        te, ptext_test, pctx_test, pham_lin_test,
        stage1_gate["textThreshold"],
        stage1_gate["contextThreshold"],
        stage1_gate["hamVetoThreshold"],
    )

    # Stage 2: independent neural spam and ham ensembles. A rescue is allowed
    # only when all experts agree and both neural ensembles are stable.
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
        n.train_one(spam_train_ds, spam_val_ds, SEED + 1100 + i * 97)
        for i in range(5)
    ]
    ham_nn_models = [
        n.train_one(ham_train_ds, ham_val_ds, SEED + 6100 + i * 97)
        for i in range(5)
    ]

    spam_matrix_val = model_matrix(spam_nn_models, full_val_ds)
    ham_matrix_val = model_matrix(ham_nn_models, full_val_ds)
    spam_matrix_test = model_matrix(spam_nn_models, full_test_ds)
    ham_matrix_test = model_matrix(ham_nn_models, full_test_ds)

    feat_val = confidence_features(
        ptext_val, pctx_val, pham_lin_val, spam_matrix_val, ham_matrix_val
    )
    feat_test = confidence_features(
        ptext_test, pctx_test, pham_lin_test, spam_matrix_test, ham_matrix_test
    )

    safe_gate = choose_rescue_gate(va, stage1_val, feat_val, max_added_fp=0)
    balanced_gate = choose_rescue_gate(va, stage1_val, feat_val, max_added_fp=1)

    safe_test_pred = rescue_prediction(stage1_test, feat_test, safe_gate)
    balanced_test_pred = rescue_prediction(stage1_test, feat_test, balanced_gate)

    stage1_test_stats = v5.stats(te, stage1_test)
    safe_test = stats(te, safe_test_pred, stage1_test)
    balanced_test = stats(te, balanced_test_pred, stage1_test)

    result = {
        "version": "v11-precision-neural-cascade",
        "dataset": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "testSpam": base["spamTotal"],
            "testHam": base["hamTotal"],
        },
        "architecture": {
            "stage1": "v5 precision linear three-key gate",
            "stage2": "5x neural spam + 5x neural ham uncertainty-aware rescue",
            "allExpertsMustAgree": True,
            "hardHamRescueFpBudget": 0,
            "testUntouchedWithinRun": True,
        },
        "rspamdBayes": base,
        "stage1ValidationGate": stage1_gate,
        "stage1Test": stage1_test_stats,
        "safeRescueValidationGate": safe_gate,
        "safeCascadeTest": safe_test,
        "balancedRescueValidationGate": balanced_gate,
        "balancedCascadeTest": balanced_test,
        "review1pctAfterStage1": review(te, stage1_test, feat_test),
    }

    (REPORTS / "v11-benchmark.json").write_text(json.dumps(result, indent=2))

    md = [
        "# MailGuard v11 precision + neural rescue benchmark",
        "",
        f"Final test: {base['spamTotal']} spam + {base['hamTotal']} ham.",
        "",
        "| Mode | Spam recall | FP total | Extra spam vs stage 1 | Extra FP vs stage 1 |",
        "|---|---:|---:|---:|---:|",
        f"| Rspamd + Bayes | {base['recall']:.2%} | {base['falsePositives']}/{base['hamTotal']} | - | - |",
        f"| Stage 1 (v5 precision) | {stage1_test_stats['recall']:.2%} | "
        f"{stage1_test_stats['falsePositives']}/{stage1_test_stats['hamTotal']} | 0 | 0 |",
        f"| v11 safe cascade | {safe_test['recall']:.2%} | "
        f"{safe_test['falsePositives']}/{safe_test['hamTotal']} | "
        f"{safe_test['rescuedOverStage1']} | {safe_test['addedFalsePositivesOverStage1']} |",
        f"| v11 balanced cascade | {balanced_test['recall']:.2%} | "
        f"{balanced_test['falsePositives']}/{balanced_test['hamTotal']} | "
        f"{balanced_test['rescuedOverStage1']} | {balanced_test['addedFalsePositivesOverStage1']} |",
        "",
        "## Validation rescue gates",
        "",
        f"Safe: {safe_gate}",
        f"Balanced: {balanced_gate}",
        "",
        "## 1% review after stage 1",
        "",
        f"Random {result['review1pctAfterStage1']['randomSpamFoundMean']:.2f}, "
        f"top-risk {result['review1pctAfterStage1']['topRiskSpamFound']}, "
        f"lift {result['review1pctAfterStage1']['lift']}.",
    ]
    text = "\n".join(md) + "\n"
    (REPORTS / "v11-benchmark.md").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
