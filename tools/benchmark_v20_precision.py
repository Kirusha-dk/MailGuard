#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight

import benchmark_improved as b
import benchmark_v3 as v3
import benchmark_v5 as v5
import benchmark_v16_reputation as v16
import benchmark_v18_overnight as v18
import benchmark_v19_combined as v19

REPORTS = Path("reports")
MODELS = Path("models/v20")
SEED = 20261002


def ordered_halves(rows):
    idx = sorted(
        range(len(rows)),
        key=lambda i: (v18.original_date(rows[i]["path"]), str(rows[i]["path"])),
    )
    cut = len(idx) // 2
    a = np.zeros(len(rows), dtype=bool)
    bmask = np.zeros(len(rows), dtype=bool)
    a[idx[:cut]] = True
    bmask[idx[cut:]] = True
    return a, bmask


def feature_matrix(
    rows,
    ptext,
    pctx,
    pham,
    prep,
    prep_dis,
    pchar,
    pg,
    pd,
    stage1,
    v16pred,
    nearpred,
):
    out = []
    for i, row in enumerate(rows):
        required = float(row["required"] or 6.0)
        rscore = float(row["rscore"] or 0.0)
        ratio = rscore / required if required else 0.0
        symbols = {name.upper() for name, _ in row["symbols"]}
        pos = sum(1 for _, s in row["symbols"] if s > 0)
        neg = sum(1 for _, s in row["symbols"] if s < 0)

        base = [
            ptext[i],
            pctx[i],
            1.0 - pham[i],
            prep[i],
            1.0 - min(1.0, prep_dis[i]),
            pchar[i],
            pg[i],
            pd[i],
            math.sqrt(max(1e-8, pg[i] * pd[i])),
            float(stage1[i]),
            float(v16pred[i]),
            float(nearpred[i]),
            max(-4.0, min(4.0, rscore / 10.0)),
            max(-4.0, min(4.0, ratio)),
            math.log1p(len(row["raw"])) / 14.0,
            min(pos, 50) / 50.0,
            min(neg, 50) / 50.0,
            float("BAYES_HAM" in symbols),
            float("MIME_GOOD" in symbols),
            float("LOCAL_OUTBOUND" in symbols),
        ]

        # A few consensus statistics make the guard robust to one overconfident expert.
        probs = np.asarray(
            [ptext[i], pctx[i], 1.0 - pham[i], prep[i], pchar[i], pg[i], pd[i]],
            dtype=np.float64,
        )
        base.extend([
            float(np.min(probs)),
            float(np.median(probs)),
            float(np.mean(probs)),
            float(np.std(probs)),
            float(np.sum(probs >= 0.80) / len(probs)),
            float(np.sum(probs >= 0.95) / len(probs)),
        ])
        out.append(base)
    return np.asarray(out, dtype=np.float64)


def fit_precision_guard(x, rows):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)

    # Customer use-case strongly penalises ham classified as spam.
    for i, row in enumerate(rows):
        if not row["y"]:
            weights[i] *= 14.0
            if float(row["rscore"]) >= 2.0:
                weights[i] *= 1.8
            symbols = {name.upper() for name, _ in row["symbols"]}
            if "BAYES_HAM" in symbols:
                weights[i] *= 1.6
        elif float(row["rscore"]) <= 3.5:
            weights[i] *= 1.8

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.30,
            max_iter=1200,
            solver="lbfgs",
            random_state=SEED + 20,
        ),
    )
    model.fit(x, y, logisticregression__sample_weight=weights)
    return model


def select_guard(rows, raw_score, max_fp, margin):
    y = np.asarray([bool(r["y"]) for r in rows], dtype=bool)
    ham_scores = raw_score[~y]
    spam_scores = raw_score[y]

    # Candidate threshold search, but never below the observed ham ceiling + margin
    # for the safe gate. This is intentionally conservative.
    ham_ceiling = float(np.max(ham_scores)) if len(ham_scores) else 1.0
    candidates = np.unique(np.concatenate([
        np.asarray([
            .50,.60,.70,.75,.80,.85,.88,.90,.92,.94,.95,.96,.97,.975,
            .98,.985,.99,.992,.994,.996,.998,.999,1.000001
        ]),
        np.quantile(raw_score, np.linspace(.50, 1.0, 160)),
    ]))

    best = None
    for threshold in candidates:
        if max_fp == 0 and threshold < min(1.000001, ham_ceiling + margin):
            continue
        pred = raw_score >= threshold
        tp = int((y & pred).sum())
        fp = int(((~y) & pred).sum())
        if fp > max_fp:
            continue
        point = {
            "threshold": float(threshold),
            "spamDetected": tp,
            "falsePositives": fp,
            "recall": tp / max(1, int(y.sum())),
            "hamCeiling": ham_ceiling,
            "safetyMargin": margin,
        }
        key = (tp, -fp, threshold)
        if best is None or key > best[0]:
            best = (key, point)

    if best is None:
        return {
            "threshold": 1.000001,
            "spamDetected": 0,
            "falsePositives": 0,
            "recall": 0.0,
            "hamCeiling": ham_ceiling,
            "safetyMargin": margin,
        }
    return best[1]


def apply_guard(base_rspamd, score, gate):
    return base_rspamd | (score >= gate["threshold"])


def main():
    torch.set_num_threads(min(6, os.cpu_count() or 2))
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    b.wait_rspamd()

    print("[1/16] prepare chronological 10k benchmark", flush=True)
    all_enron = v18.enron.prepare_enron()
    enron_train, enron_val, enron_test = v18.split_enron(all_enron)
    groups = b.prepare()
    sa_train, _, _ = v3.build_splits(groups)
    train_paths = list(enron_train) + list(sa_train)

    print(
        "TRAIN", len(train_paths),
        "VAL", len(enron_val),
        "LOCKED TEST", len(enron_test),
        flush=True,
    )

    print("[2/16] train Rspamd Bayes", flush=True)
    b.reset_bayes()
    b.learn(train_paths)

    print("[3/16] scan train", flush=True)
    tr = b.scan_many(train_paths, "v20-train", workers=16)
    print("[4/16] scan validation", flush=True)
    va = b.scan_many(enron_val, "v20-val", workers=16)
    print("[5/16] scan 10k test", flush=True)
    te = b.scan_many(enron_test, "v20-test", workers=16)

    base_val = np.asarray([v18.is_protected(r) for r in va], dtype=bool)
    base_test = np.asarray([v18.is_protected(r) for r in te], dtype=bool)

    print("[6/16] train text/context baseline", flush=True)
    xt, xv, xe, ctx_t, ctx_v, ctx_e = v3.matrices(tr, va, te)
    residual_idx = [i for i, row in enumerate(tr) if not v18.is_protected(row)]
    residual_rows = [tr[i] for i in residual_idx]

    text_models = v3.fit_ensemble(xt[residual_idx], residual_rows)
    ctx_models = v3.fit_ensemble(ctx_t[residual_idx], residual_rows)
    ham_models = v5.fit_ham_ensemble(ctx_t[residual_idx], residual_rows)

    ptext_val = v3.ensemble_predict(text_models, xv)
    pctx_val = v3.ensemble_predict(ctx_models, ctx_v)
    pham_val = v5.avg_prob(ham_models, ctx_v)
    ptext_test = v3.ensemble_predict(text_models, xe)
    pctx_test = v3.ensemble_predict(ctx_models, ctx_e)
    pham_test = v5.avg_prob(ham_models, ctx_e)

    stage1_gate, _ = v5.choose_gate(va, ptext_val, pctx_val, pham_val)
    stage1_val = v5.predict_gate(
        va, ptext_val, pctx_val, pham_val,
        stage1_gate["textThreshold"],
        stage1_gate["contextThreshold"],
        stage1_gate["hamVetoThreshold"],
    )
    stage1_test = v5.predict_gate(
        te, ptext_test, pctx_test, pham_test,
        stage1_gate["textThreshold"],
        stage1_gate["contextThreshold"],
        stage1_gate["hamVetoThreshold"],
    )

    print("[7/16] build v16 reputation/campaign branch", flush=True)
    residual_meta = [v19.rep_meta(r) for r in residual_rows]
    val_meta = [v19.rep_meta(r) for r in va]
    test_meta = [v19.rep_meta(r) for r in te]
    xrep_train = v16.reputation_features_oof(residual_rows, residual_meta)
    y_residual = np.asarray([r["y"] for r in residual_rows], dtype=np.int32)
    rep_db = v16.ReputationDB().fit(residual_meta, y_residual)
    xrep_val = v16.reputation_features(rep_db, va, val_meta)
    xrep_test = v16.reputation_features(rep_db, te, test_meta)

    rep_models = v16.fit_reputation_models(xrep_train, residual_rows)
    prep_val, prep_dis_val = v16.rep_prob(rep_models, xrep_val)
    prep_test, prep_dis_test = v16.rep_prob(rep_models, xrep_test)

    feat_val = v16.combined_features(
        ptext_val, pctx_val, pham_val, prep_val, prep_dis_val
    )
    feat_test = v16.combined_features(
        ptext_test, pctx_test, pham_test, prep_test, prep_dis_test
    )

    v16_gate = v16.choose_gate(va, stage1_val, feat_val, safe=True)
    v16_val = v16.predict(stage1_val, feat_val, v16_gate)
    v16_test = v16.predict(stage1_test, feat_test, v16_gate)

    print("[8/16] train char near-miss expert", flush=True)
    char_train = b.CHAR.transform([b.extract_document(r["raw"]) for r in residual_rows])
    char_val = b.CHAR.transform([b.extract_document(r["raw"]) for r in va])
    char_test = b.CHAR.transform([b.extract_document(r["raw"]) for r in te])
    char_models = v19.fit_char_models(char_train, residual_rows)
    pchar_val = v19.avg_prob(char_models, char_val)
    pchar_test = v19.avg_prob(char_models, char_test)

    near_val = np.exp(np.mean(np.log(np.vstack([
        np.clip(pchar_val, 1e-8, 1.0),
        np.clip(ptext_val, 1e-8, 1.0),
        np.clip(pctx_val, 1e-8, 1.0),
        np.clip(prep_val, 1e-8, 1.0),
        np.clip(1.0 - pham_val, 1e-8, 1.0),
    ])), axis=0))
    near_test = np.exp(np.mean(np.log(np.vstack([
        np.clip(pchar_test, 1e-8, 1.0),
        np.clip(ptext_test, 1e-8, 1.0),
        np.clip(pctx_test, 1e-8, 1.0),
        np.clip(prep_test, 1e-8, 1.0),
        np.clip(1.0 - pham_test, 1e-8, 1.0),
    ])), axis=0))

    near_mask_val = (
        (feat_val["median"] >= 0.12)
        & (feat_val["consensus"] >= 0.015)
        & (pham_val <= 0.97)
    )
    near_mask_test = (
        (feat_test["median"] >= 0.12)
        & (feat_test["consensus"] >= 0.015)
        & (pham_test <= 0.97)
    )
    near_gate = v19.choose_simple_rescue(
        va, v16_val, near_val, near_mask_val, 0, "near-safe"
    )
    near_val_pred = v19.apply_simple_rescue(
        v16_val, near_val, near_mask_val, near_gate
    )
    near_test_pred = v19.apply_simple_rescue(
        v16_test, near_test, near_mask_test, near_gate
    )

    print("[9/16] vectorize neural residuals", flush=True)
    general_ds = v18.MailDataset(
        residual_rows, v18.base_weights(residual_rows, deep=False)
    )
    deep_rows = [r for r in residual_rows if float(r["rscore"]) <= 3.5]
    if len({r["y"] for r in deep_rows}) < 2:
        deep_rows = residual_rows
    deep_ds = v18.MailDataset(
        deep_rows, v18.base_weights(deep_rows, deep=True)
    )
    val_ds = v18.MailDataset(va, np.ones(len(va), dtype=np.float32))
    test_ds = v18.MailDataset(te, np.ones(len(te), dtype=np.float32))

    print("[10/16] train neural experts", flush=True)
    general_nn = []
    deep_nn = []
    for i in range(2):
        m, _ = v18.train_model(
            f"v20-general-{i+1}", general_ds, val_ds, va,
            SEED + i * 101, max_epochs=22, patience=5, min_epochs=8,
            max_added_fp=0,
        )
        general_nn.append(m)
    for i in range(2):
        m, _ = v18.train_model(
            f"v20-deep-{i+1}", deep_ds, val_ds, va,
            SEED + 5000 + i * 131, max_epochs=26, patience=6, min_epochs=9,
            max_added_fp=0,
        )
        deep_nn.append(m)

    print("[11/16] score neural branches", flush=True)
    pg_val, _ = v18.ensemble_predict(general_nn, val_ds)
    pd_val, _ = v18.ensemble_predict(deep_nn, val_ds)
    pg_test, _ = v18.ensemble_predict(general_nn, test_ds)
    pd_test, _ = v18.ensemble_predict(deep_nn, test_ds)

    print("[12/16] fit chronological precision guard", flush=True)
    xval = feature_matrix(
        va, ptext_val, pctx_val, pham_val, prep_val, prep_dis_val,
        pchar_val, pg_val, pd_val, stage1_val, v16_val, near_val_pred
    )
    xtest = feature_matrix(
        te, ptext_test, pctx_test, pham_test, prep_test, prep_dis_test,
        pchar_test, pg_test, pd_test, stage1_test, v16_test, near_test_pred
    )

    fit_mask, gate_mask = ordered_halves(va)
    guard = fit_precision_guard(
        xval[fit_mask],
        [va[i] for i in np.where(fit_mask)[0]],
    )
    score_val = guard.predict_proba(xval)[:, 1]
    score_test = guard.predict_proba(xtest)[:, 1]

    gate_rows = [va[i] for i in np.where(gate_mask)[0]]
    gate_scores = score_val[gate_mask]

    safe_gate = select_guard(
        gate_rows, gate_scores, max_fp=0, margin=0.015
    )
    balanced_gate = select_guard(
        gate_rows, gate_scores, max_fp=1, margin=0.0
    )

    print("[13/16] apply precision guard to 10k test", flush=True)
    safe_pred = apply_guard(base_test, score_test, safe_gate)
    balanced_pred = apply_guard(base_test, score_test, balanced_gate)

    rspamd_stats = v19.metrics(te, base_test)
    stage1_stats = v19.metrics(te, stage1_test)
    v16_stats = v19.metrics(te, v16_test)
    near_stats = v19.metrics(te, near_test_pred)
    safe_stats = v19.metrics(te, safe_pred)
    balanced_stats = v19.metrics(te, balanced_pred)

    print("[14/16] save guard", flush=True)
    import pickle
    with open(MODELS / "precision-guard.pkl", "wb") as f:
        pickle.dump({
            "model": guard,
            "safeGate": safe_gate,
            "balancedGate": balanced_gate,
        }, f)

    result = {
        "version": "v20-precision-guard",
        "dataset": {
            "lockedTest": len(te),
            "testSpam": safe_stats["spamTotal"],
            "testHam": safe_stats["hamTotal"],
            "validation": len(va),
            "guardFit": int(fit_mask.sum()),
            "guardGateSelection": int(gate_mask.sum()),
            "split": "chronological within validation; 10k test untouched by v20 fitting",
        },
        "method": {
            "base": "v16 reputation/campaign + v17 char/near-miss + v18-style neural signals",
            "precisionGuard": "weighted logistic meta-classifier",
            "hamWeightMultiplier": 14.0,
            "safeGate": "0 FP on later validation half plus score safety margin",
            "testLabelsUsedForTrainingOrThresholds": False,
            "individualV19TestErrorsInspected": False,
        },
        "rspamdBayes": rspamd_stats,
        "stage1": stage1_stats,
        "v16": v16_stats,
        "nearMiss": near_stats,
        "v20SafeGate": safe_gate,
        "v20Safe": safe_stats,
        "v20BalancedGate": balanced_gate,
        "v20Balanced": balanced_stats,
        "warning": (
            "This is still the same Enron engineering benchmark family used by v19. "
            "Use fresh modern customer/local holdout data before production claims."
        ),
    }
    (REPORTS / "v20-precision.json").write_text(json.dumps(result, indent=2))

    print("[15/16] write report", flush=True)
    rows = [
        ("Rspamd + Bayes", rspamd_stats),
        ("Stage 1", stage1_stats),
        ("v16", v16_stats),
        ("+ near-miss", near_stats),
        ("v20 safe", safe_stats),
        ("v20 balanced", balanced_stats),
    ]
    md = [
        "# MailGuard v20 precision-guard benchmark",
        "",
        f"Locked test: **{len(te)} unique emails** "
        f"({safe_stats['spamTotal']} spam + {safe_stats['hamTotal']} ham).",
        "",
        "| Mode | Recall | FN | FP | FP rate | Precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, s in rows:
        md.append(
            f"| {label} | {s['recall']:.2%} | {s['falseNegatives']} | "
            f"{s['falsePositives']} | {s['fpr']:.3%} | {s['precision']:.3%} |"
        )
    md.extend([
        "",
        f"Safe gate: {safe_gate}",
        f"Balanced gate: {balanced_gate}",
        "",
        "v20 intentionally trades some recall for a much stronger FP guard.",
        "The guard is fit on the earlier validation half and thresholded on the later half.",
    ])

    report = "\n".join(md) + "\n"
    (REPORTS / "v20-precision.md").write_text(report)
    print(report, flush=True)

    print("[16/16] done", flush=True)


if __name__ == "__main__":
    main()
