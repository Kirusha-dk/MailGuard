#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_sample_weight

import benchmark_improved as b
import benchmark_v3 as v3
import benchmark_v5 as v5
import benchmark_v15_hard_mining as v15
import benchmark_v16_reputation as v16
import benchmark_v18_overnight as v18

REPORTS = Path("reports")
MODELS = Path("models/v19")
SEED = 20261002


def rep_meta(row):
    meta = v16.parse_mail(row)
    # Enron benchmark transport headers are synthetic. Do not let the
    # reputation branch learn fake per-message sender/routing identifiers.
    if b"X-MailGuard-Dataset: enron-spam" in row["raw"][:5000]:
        for key in (
            "sender",
            "senderDomain",
            "returnDomain",
            "messageIdDomain",
            "ip24",
        ):
            meta[key] = ""
    return meta


def metrics(rows, pred):
    y = np.asarray([bool(r["y"]) for r in rows], dtype=bool)
    spam = int(y.sum())
    ham = int((~y).sum())
    tp = int((y & pred).sum())
    fp = int(((~y) & pred).sum())
    return {
        "spamTotal": spam,
        "hamTotal": ham,
        "spamDetected": tp,
        "falseNegatives": spam - tp,
        "falsePositives": fp,
        "recall": tp / max(1, spam),
        "precision": tp / max(1, tp + fp),
        "fpr": fp / max(1, ham),
    }


def choose_simple_rescue(
    rows,
    base_pred,
    score,
    extra_mask,
    max_added_fp,
    name,
):
    y = np.asarray([bool(r["y"]) for r in rows], dtype=bool)
    ham = ~y
    base_fp = int((ham & base_pred).sum())
    base_tp = int((y & base_pred).sum())

    residual = (~base_pred) & extra_mask
    vals = score[residual]
    if len(vals) == 0:
        return {
            "name": name,
            "threshold": 1.000001,
            "rescuedSpam": 0,
            "addedFalsePositives": 0,
        }

    candidates = np.unique(np.concatenate([
        np.asarray([
            .25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,
            .88,.90,.92,.94,.95,.96,.97,.98,.985,.99,.995,.999,
            1.000001,
        ]),
        np.quantile(vals, np.linspace(.05, 1.0, 180)),
    ]))

    best = None
    for threshold in candidates:
        rescue = residual & (score >= threshold)
        pred = base_pred | rescue
        tp = int((y & pred).sum())
        fp = int((ham & pred).sum())
        added_fp = fp - base_fp
        if added_fp > max_added_fp:
            continue

        point = {
            "name": name,
            "threshold": float(threshold),
            "rescuedSpam": tp - base_tp,
            "addedFalsePositives": added_fp,
        }
        key = (
            point["rescuedSpam"],
            -point["addedFalsePositives"],
            point["threshold"],
        )
        if best is None or key > best[0]:
            best = (key, point)

    if best is None:
        return {
            "name": name,
            "threshold": 1.000001,
            "rescuedSpam": 0,
            "addedFalsePositives": 0,
        }
    return best[1]


def apply_simple_rescue(base_pred, score, extra_mask, gate):
    return base_pred | (
        (~base_pred)
        & extra_mask
        & (score >= gate["threshold"])
    )


def fit_char_models(x, rows):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)

    for i, row in enumerate(rows):
        if row["y"] and float(row["rscore"]) <= 3.5:
            weights[i] *= 3.0
        if (not row["y"]) and float(row["rscore"]) >= 3.0:
            weights[i] *= 5.0
        if (not row["y"]) and any(
            name.upper() == "BAYES_HAM" for name, _ in row["symbols"]
        ):
            weights[i] *= 1.5

    models = []
    for j, alpha in enumerate((7e-5, 2e-4, 6e-4)):
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            max_iter=30,
            tol=1e-4,
            random_state=SEED + 1700 + j * 97,
        )
        model.fit(x, y, sample_weight=weights)
        models.append(model)
    return models


def avg_prob(models, x):
    return np.mean([m.predict_proba(x)[:, 1] for m in models], axis=0)


def main():
    torch.set_num_threads(min(6, os.cpu_count() or 2))
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    b.wait_rspamd()

    print("[1/15] prepare chronological Enron lockbox", flush=True)
    all_enron = v18.enron.prepare_enron()
    enron_train, enron_val, enron_test = v18.split_enron(all_enron)
    if len(enron_test) != 10000:
        raise RuntimeError(f"Expected 10000 test emails, got {len(enron_test)}")

    # Add older SpamAssassin data only to training for extra lexical diversity.
    groups = b.prepare()
    sa_train, _sa_val, _sa_test = v3.build_splits(groups)
    train_paths = list(enron_train) + list(sa_train)

    print(
        "TRAIN", len(train_paths),
        "VAL", len(enron_val),
        "LOCKED TEST", len(enron_test),
        flush=True,
    )

    print("[2/15] train Rspamd Bayes on training only", flush=True)
    b.reset_bayes()
    b.learn(train_paths)

    print("[3/15] scan training", flush=True)
    tr = b.scan_many(train_paths, "v19-train", workers=16)
    print("[4/15] scan validation", flush=True)
    va = b.scan_many(enron_val, "v19-val", workers=16)
    print("[5/15] scan locked 10k test", flush=True)
    te = b.scan_many(enron_test, "v19-test", workers=16)

    base_rspamd = np.asarray([v18.is_protected(r) for r in te], dtype=bool)
    rspamd_stats = metrics(te, base_rspamd)

    print("[6/15] train text/context/ham baseline", flush=True)
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

    print("[7/15] build v16 reputation/campaign branch", flush=True)
    residual_meta = [rep_meta(r) for r in residual_rows]
    val_meta = [rep_meta(r) for r in va]
    test_meta = [rep_meta(r) for r in te]

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

    v16_safe_gate = v16.choose_gate(va, stage1_val, feat_val, safe=True)
    v16_bal_gate = v16.choose_gate(va, stage1_val, feat_val, safe=False)
    v16_safe_val = v16.predict(stage1_val, feat_val, v16_safe_gate)
    v16_safe_test = v16.predict(stage1_test, feat_test, v16_safe_gate)
    v16_bal_test = v16.predict(stage1_test, feat_test, v16_bal_gate)

    print("[8/15] train v17-style char near-miss specialist", flush=True)
    char_train = b.CHAR.transform([
        b.extract_document(r["raw"]) for r in residual_rows
    ])
    char_val = b.CHAR.transform([b.extract_document(r["raw"]) for r in va])
    char_test = b.CHAR.transform([b.extract_document(r["raw"]) for r in te])

    char_models = fit_char_models(char_train, residual_rows)
    pchar_val = avg_prob(char_models, char_val)
    pchar_test = avg_prob(char_models, char_test)

    # Near-miss score: independent lexical evidence + v16 evidence + ham shield.
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

    # Only rescue messages that are close enough to the main evidence boundary.
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

    near_safe_gate = choose_simple_rescue(
        va, v16_safe_val, near_val, near_mask_val, 0, "near-safe"
    )
    near_bal_gate = choose_simple_rescue(
        va, v16_safe_val, near_val, near_mask_val, 1, "near-balanced"
    )

    v17ish_safe_val = apply_simple_rescue(
        v16_safe_val, near_val, near_mask_val, near_safe_gate
    )
    v17ish_safe_test = apply_simple_rescue(
        v16_safe_test, near_test, near_mask_test, near_safe_gate
    )
    v17ish_bal_test = apply_simple_rescue(
        v16_safe_test, near_test, near_mask_test, near_bal_gate
    )

    print("[9/15] vectorize neural residual data", flush=True)
    general_weights = v18.base_weights(residual_rows, deep=False)
    deep_rows = [
        row for row in residual_rows
        if float(row["rscore"]) <= 3.5
    ]
    if len({r["y"] for r in deep_rows}) < 2:
        deep_rows = residual_rows
    deep_weights = v18.base_weights(deep_rows, deep=True)

    general_ds = v18.MailDataset(residual_rows, general_weights)
    deep_ds = v18.MailDataset(deep_rows, deep_weights)
    val_ds = v18.MailDataset(
        va, np.ones(len(va), dtype=np.float32)
    )
    test_ds = v18.MailDataset(
        te, np.ones(len(te), dtype=np.float32)
    )

    print("[10/15] train 2 general neural experts", flush=True)
    general_nn = []
    general_meta = []
    for i in range(2):
        model, meta = v18.train_model(
            f"v19-general-{i+1}",
            general_ds,
            val_ds,
            va,
            SEED + i * 101,
            max_epochs=28,
            patience=6,
            min_epochs=9,
            max_added_fp=0,
        )
        general_nn.append(model)
        general_meta.append(meta)

    print("[11/15] train 2 deep-error neural experts", flush=True)
    deep_nn = []
    deep_meta = []
    for i in range(2):
        model, meta = v18.train_model(
            f"v19-deep-{i+1}",
            deep_ds,
            val_ds,
            va,
            SEED + 5000 + i * 131,
            max_epochs=32,
            patience=7,
            min_epochs=10,
            max_added_fp=0,
        )
        deep_nn.append(model)
        deep_meta.append(meta)

    print("[12/15] score neural deep branch", flush=True)
    pg_val, _ = v18.ensemble_predict(general_nn, val_ds)
    pd_val, _ = v18.ensemble_predict(deep_nn, val_ds)
    pg_test, _ = v18.ensemble_predict(general_nn, test_ds)
    pd_test, _ = v18.ensemble_predict(deep_nn, test_ds)

    low_val = np.asarray([float(r["rscore"]) <= 3.5 for r in va], dtype=bool)
    low_test = np.asarray([float(r["rscore"]) <= 3.5 for r in te], dtype=bool)

    neural_val = np.exp(
        0.55 * np.log(np.clip(pg_val, 1e-8, 1.0))
        + 0.45 * np.log(np.clip(pd_val, 1e-8, 1.0))
    )
    neural_test = np.exp(
        0.55 * np.log(np.clip(pg_test, 1e-8, 1.0))
        + 0.45 * np.log(np.clip(pd_test, 1e-8, 1.0))
    )

    deep_safe_gate = choose_simple_rescue(
        va,
        v17ish_safe_val,
        neural_val,
        low_val,
        0,
        "deep-safe",
    )
    deep_bal_gate = choose_simple_rescue(
        va,
        v17ish_safe_val,
        neural_val,
        low_val,
        1,
        "deep-balanced",
    )

    final_safe = apply_simple_rescue(
        v17ish_safe_test,
        neural_test,
        low_test,
        deep_safe_gate,
    )
    final_bal = apply_simple_rescue(
        v17ish_bal_test,
        neural_test,
        low_test,
        deep_bal_gate,
    )

    print("[13/15] calculate locked-test metrics", flush=True)
    stage1_stats = metrics(te, stage1_test)
    v16_safe_stats = metrics(te, v16_safe_test)
    v16_bal_stats = metrics(te, v16_bal_test)
    near_safe_stats = metrics(te, v17ish_safe_test)
    near_bal_stats = metrics(te, v17ish_bal_test)
    final_safe_stats = metrics(te, final_safe)
    final_bal_stats = metrics(te, final_bal)

    print("[14/15] save final neural package", flush=True)
    torch.save({
        "version": "v19-combined",
        "generalStates": [m.state_dict() for m in general_nn],
        "deepStates": [m.state_dict() for m in deep_nn],
        "nearSafeGate": near_safe_gate,
        "nearBalancedGate": near_bal_gate,
        "deepSafeGate": deep_safe_gate,
        "deepBalancedGate": deep_bal_gate,
    }, MODELS / "mailguard-v19-neural.pt")

    result = {
        "version": "v19-combined",
        "dataset": {
            "enronUnique": len(all_enron),
            "train": len(train_paths),
            "validation": len(va),
            "lockedTest": len(te),
            "testSpam": final_safe_stats["spamTotal"],
            "testHam": final_safe_stats["hamTotal"],
            "split": "class-stratified chronological Enron lockbox",
        },
        "architecture": {
            "stage1": "text/context/ham precision baseline",
            "v16": "local reputation + campaign + URL-domain rescue",
            "v17UsefulParts": "char specialist + near-miss rescue + ham shield",
            "v18UsefulParts": "general neural experts + low-Rspamd deep specialists",
            "testLabelsUsedForTrainingOrThresholds": False,
            "syntheticTransportIdentityExcludedFromReputation": True,
        },
        "rspamdBayes": rspamd_stats,
        "stage1": stage1_stats,
        "v16Safe": v16_safe_stats,
        "v16Balanced": v16_bal_stats,
        "afterNearMissSafe": near_safe_stats,
        "afterNearMissBalanced": near_bal_stats,
        "v19Safe": final_safe_stats,
        "v19Balanced": final_bal_stats,
        "gates": {
            "stage1": stage1_gate,
            "v16Safe": v16_safe_gate,
            "v16Balanced": v16_bal_gate,
            "nearSafe": near_safe_gate,
            "nearBalanced": near_bal_gate,
            "deepSafe": deep_safe_gate,
            "deepBalanced": deep_bal_gate,
        },
        "training": {
            "generalNeural": general_meta,
            "deepNeural": deep_meta,
            "residualTraining": len(residual_rows),
            "deepTraining": len(deep_rows),
        },
        "warning": (
            "The locked test is old Enron-Spam with synthetic transport headers. "
            "It is useful for large chronological holdout testing but is not a "
            "direct estimate of current production mail."
        ),
    }

    (REPORTS / "v19-combined.json").write_text(json.dumps(result, indent=2))

    print("[15/15] write report", flush=True)
    rows = [
        ("Rspamd + Bayes", rspamd_stats),
        ("Stage 1", stage1_stats),
        ("v16 reputation/campaign", v16_safe_stats),
        ("+ v17 near-miss", near_safe_stats),
        ("v19 final safe", final_safe_stats),
        ("v19 final balanced", final_bal_stats),
    ]

    md = [
        "# MailGuard v19 combined benchmark",
        "",
        f"Locked test: **{len(te)} unique emails** "
        f"({final_safe_stats['spamTotal']} spam + "
        f"{final_safe_stats['hamTotal']} ham).",
        "",
        "| Mode | Recall | FN | FP | FP rate | Precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, s in rows:
        md.append(
            f"| {label} | {s['recall']:.2%} | "
            f"{s['falseNegatives']} | {s['falsePositives']} | "
            f"{s['fpr']:.3%} | {s['precision']:.3%} |"
        )

    md.extend([
        "",
        f"Near-miss safe gate: {near_safe_gate}",
        f"Deep-neural safe gate: {deep_safe_gate}",
        "",
        "v19 combines v16 reputation/campaign evidence, the useful v17 "
        "near-miss/char logic, and v18 deep neural specialists.",
        "",
        "The 10k test labels are not used for training or threshold selection.",
    ])

    report = "\n".join(md) + "\n"
    (REPORTS / "v19-combined.md").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
