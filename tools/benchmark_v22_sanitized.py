#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import hstack
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
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
MODELS = Path("models/v22")
SEED = 20261004



def clean_doc(row):
    # The Enron benchmark wraps every record in a synthetic RFC822 envelope.
    # v18 already knows how to mask those fake sender/recipient identifiers.
    return v18.clean_document(row)


def clean_context_doc(row):
    text = clean_doc(row)
    pseudo = ["__action_" + str(row["action"]).lower().replace(" ", "_")]
    for name, score in row["symbols"][:120]:
        safe = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
        pseudo.append("__sym_" + safe)
        if score >= 2:
            pseudo.append("__sympos_" + safe)
        elif score <= -2:
            pseudo.append("__symneg_" + safe)
    return text + "\n" + " ".join(pseudo)


def sanitized_matrices(train, val, test):
    pure_tr = [clean_doc(r) for r in train]
    pure_va = [clean_doc(r) for r in val]
    pure_te = [clean_doc(r) for r in test]
    ctx_tr_docs = [clean_context_doc(r) for r in train]
    ctx_va_docs = [clean_context_doc(r) for r in val]
    ctx_te_docs = [clean_context_doc(r) for r in test]

    text_tr = hstack(
        [b.WORD.transform(pure_tr), b.CHAR.transform(pure_tr)],
        format="csr",
    )
    text_va = hstack(
        [b.WORD.transform(pure_va), b.CHAR.transform(pure_va)],
        format="csr",
    )
    text_te = hstack(
        [b.WORD.transform(pure_te), b.CHAR.transform(pure_te)],
        format="csr",
    )
    ctx_tr = hstack(
        [
            b.WORD.transform(ctx_tr_docs),
            b.CHAR.transform(ctx_tr_docs),
            v3.numeric(train),
        ],
        format="csr",
    )
    ctx_va = hstack(
        [
            b.WORD.transform(ctx_va_docs),
            b.CHAR.transform(ctx_va_docs),
            v3.numeric(val),
        ],
        format="csr",
    )
    ctx_te = hstack(
        [
            b.WORD.transform(ctx_te_docs),
            b.CHAR.transform(ctx_te_docs),
            v3.numeric(test),
        ],
        format="csr",
    )
    return text_tr, text_va, text_te, ctx_tr, ctx_va, ctx_te


def train_tfidf_svm_experts(train_rows, val_rows, test_rows):
    print("[8b/16] fit leakage-clean TF-IDF/SVM experts", flush=True)
    tr_docs = [clean_doc(r) for r in train_rows]
    va_docs = [clean_doc(r) for r in val_rows]
    te_docs = [clean_doc(r) for r in test_rows]

    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.997,
        max_features=140000,
        sublinear_tf=True,
        token_pattern=r"(?u)\\b[\\w@.\\-]{2,}\\b",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=140000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    xw_tr = word.fit_transform(tr_docs)
    xw_va = word.transform(va_docs)
    xw_te = word.transform(te_docs)
    xc_tr = char.fit_transform(tr_docs)
    xc_va = char.transform(va_docs)
    xc_te = char.transform(te_docs)

    y = np.asarray([int(r["y"]) for r in train_rows], dtype=np.int32)
    sw = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)

    # Two ordinary specialists.
    word_model = LinearSVC(C=0.55, random_state=SEED + 2201)
    char_model = LinearSVC(C=0.45, random_state=SEED + 2202)
    word_model.fit(xw_tr, y, sample_weight=sw)
    char_model.fit(xc_tr, y, sample_weight=sw)

    # A deliberately conservative specialist trained to reject ham-like
    # messages even when the other experts are overconfident.
    sw_prec = sw.copy()
    for i, row in enumerate(train_rows):
        if not row["y"]:
            sw_prec[i] *= 6.0
            if float(row["rscore"]) >= 2.0:
                sw_prec[i] *= 1.8
        elif float(row["rscore"]) <= 3.5:
            sw_prec[i] *= 1.8

    precision_model = LinearSVC(C=0.32, random_state=SEED + 2203)
    precision_model.fit(
        hstack([xw_tr, xc_tr], format="csr"),
        y,
        sample_weight=sw_prec,
    )

    def score(model, x):
        return expit(np.clip(model.decision_function(x), -12.0, 12.0))

    pword_val = score(word_model, xw_va)
    pword_test = score(word_model, xw_te)
    pchar_val = score(char_model, xc_va)
    pchar_test = score(char_model, xc_te)
    pprec_val = score(
        precision_model,
        hstack([xw_va, xc_va], format="csr"),
    )
    pprec_test = score(
        precision_model,
        hstack([xw_te, xc_te], format="csr"),
    )

    return (
        pword_val,
        pword_test,
        pchar_val,
        pchar_test,
        pprec_val,
        pprec_test,
    )

def chronological_fit_and_slices(rows, n_slices=3):
    fit = np.zeros(len(rows), dtype=bool)
    slice_id = np.full(len(rows), -1, dtype=np.int32)

    # Stratify by class, but preserve chronology within each class.
    # Earliest 50% trains the precision guard. The later 50% is split into
    # several chronological calibration eras so one lucky interval cannot set
    # the whole threshold.
    for label in (0, 1):
        idx = sorted(
            [i for i, row in enumerate(rows) if int(bool(row["y"])) == label],
            key=lambda i: (
                v18.original_date(rows[i]["path"]),
                str(rows[i]["path"]),
            ),
        )
        cut = len(idx) // 2
        fit[idx[:cut]] = True
        tail = idx[cut:]
        chunks = np.array_split(np.asarray(tail, dtype=np.int64), n_slices)
        for sid, chunk in enumerate(chunks):
            slice_id[chunk] = sid

    return fit, slice_id


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
    ptfidf_word,
    ptfidf_char,
    pprecision,
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
            ptfidf_word[i],
            ptfidf_char[i],
            pprecision[i],
            math.sqrt(max(1e-8, pg[i] * pd[i])),
            math.sqrt(max(1e-8, ptfidf_word[i] * ptfidf_char[i])),
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
            [
                ptext[i], pctx[i], 1.0 - pham[i], prep[i], pchar[i],
                pg[i], pd[i], ptfidf_word[i], ptfidf_char[i], pprecision[i],
            ],
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


def agreement_count(
    ptext, pctx, pham, prep, pchar, pg, pd,
    ptfidf_word, ptfidf_char, pprecision,
):
    stack = np.vstack([
        ptext,
        pctx,
        1.0 - pham,
        prep,
        pchar,
        pg,
        pd,
        ptfidf_word,
        ptfidf_char,
        pprecision,
    ])
    return np.sum(stack >= 0.80, axis=0).astype(np.int32)


def robust_select_guard(
    rows,
    raw_score,
    agreement,
    slice_id,
    max_total_fp,
    max_slice_fp,
    name,
):
    y = np.asarray([bool(r["y"]) for r in rows], dtype=bool)
    calib = slice_id >= 0
    yc = y[calib]
    sc = raw_score[calib]
    ac = agreement[calib]
    sic = slice_id[calib]

    fixed = np.asarray([
        .40,.50,.60,.65,.70,.75,.80,.84,.86,.88,.90,.92,.94,.95,.96,
        .97,.975,.98,.985,.99,.992,.994,.996,.998,.999,1.000001
    ])
    # Exact observed boundaries make FP budgets deterministic. nextafter lets
    # us place a gate just above a ham outlier without adding an arbitrary
    # probability margin that can collapse recall.
    observed = np.unique(sc)
    candidates = np.unique(np.concatenate([
        fixed,
        observed,
        np.nextafter(observed, 1.0),
        np.quantile(sc, np.linspace(.30, 1.0, 180)),
    ]))

    best = None
    for min_agreement in (2, 3, 4, 5, 6, 7):
        for threshold in candidates:
            pred = (sc >= threshold) & (ac >= min_agreement)
            fp_total = int(((~yc) & pred).sum())
            if fp_total > max_total_fp:
                continue

            slice_stats = []
            stable = True
            for sid in sorted(set(sic.tolist())):
                sm = sic == sid
                sy = yc[sm]
                sp = pred[sm]
                fp = int(((~sy) & sp).sum())
                tp = int((sy & sp).sum())
                spam = int(sy.sum())
                recall = tp / max(1, spam)
                if fp > max_slice_fp:
                    stable = False
                    break
                slice_stats.append({
                    "slice": int(sid),
                    "spam": spam,
                    "tp": tp,
                    "fp": fp,
                    "recall": recall,
                })
            if not stable:
                continue

            tp_total = int((yc & pred).sum())
            spam_total = int(yc.sum())
            recalls = [s["recall"] for s in slice_stats]
            worst_recall = min(recalls) if recalls else 0.0
            point = {
                "name": name,
                "threshold": float(threshold),
                "minAgreement": int(min_agreement),
                "spamDetected": tp_total,
                "falsePositives": fp_total,
                "recall": tp_total / max(1, spam_total),
                "worstSliceRecall": worst_recall,
                "sliceStats": slice_stats,
            }
            # Stability first, then total catch, then fewer FP.
            key = (
                round(worst_recall, 8),
                tp_total,
                -fp_total,
                min_agreement,
                threshold,
            )
            if best is None or key > best[0]:
                best = (key, point)

    if best is None:
        return {
            "name": name,
            "threshold": 1.000001,
            "minAgreement": 7,
            "spamDetected": 0,
            "falsePositives": 0,
            "recall": 0.0,
            "worstSliceRecall": 0.0,
            "sliceStats": [],
        }
    return best[1]


def apply_guard(base_rspamd, score, agreement, gate):
    return base_rspamd | (
        (score >= gate["threshold"])
        & (agreement >= gate["minAgreement"])
    )


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
    tr = b.scan_many(train_paths, "v22-train", workers=16)
    print("[4/16] scan validation", flush=True)
    va = b.scan_many(enron_val, "v22-val", workers=16)
    print("[5/16] scan 10k test", flush=True)
    te = b.scan_many(enron_test, "v22-test", workers=16)

    base_val = np.asarray([v18.is_protected(r) for r in va], dtype=bool)
    base_test = np.asarray([v18.is_protected(r) for r in te], dtype=bool)

    print("[6/16] train text/context baseline", flush=True)
    xt, xv, xe, ctx_t, ctx_v, ctx_e = sanitized_matrices(tr, va, te)
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
    char_train = b.CHAR.transform([clean_doc(r) for r in residual_rows])
    char_val = b.CHAR.transform([clean_doc(r) for r in va])
    char_test = b.CHAR.transform([clean_doc(r) for r in te])
    char_models = v19.fit_char_models(char_train, residual_rows)
    pchar_val = v19.avg_prob(char_models, char_val)
    pchar_test = v19.avg_prob(char_models, char_test)

    (
        ptfidf_word_val,
        ptfidf_word_test,
        ptfidf_char_val,
        ptfidf_char_test,
        pprecision_val,
        pprecision_test,
    ) = train_tfidf_svm_experts(residual_rows, va, te)

    near_val = np.exp(np.mean(np.log(np.vstack([
        np.clip(pchar_val, 1e-8, 1.0),
        np.clip(ptext_val, 1e-8, 1.0),
        np.clip(pctx_val, 1e-8, 1.0),
        np.clip(prep_val, 1e-8, 1.0),
        np.clip(1.0 - pham_val, 1e-8, 1.0),
        np.clip(ptfidf_word_val, 1e-8, 1.0),
        np.clip(ptfidf_char_val, 1e-8, 1.0),
    ])), axis=0))
    near_test = np.exp(np.mean(np.log(np.vstack([
        np.clip(pchar_test, 1e-8, 1.0),
        np.clip(ptext_test, 1e-8, 1.0),
        np.clip(pctx_test, 1e-8, 1.0),
        np.clip(prep_test, 1e-8, 1.0),
        np.clip(1.0 - pham_test, 1e-8, 1.0),
        np.clip(ptfidf_word_test, 1e-8, 1.0),
        np.clip(ptfidf_char_test, 1e-8, 1.0),
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
            f"v22-general-{i+1}", general_ds, val_ds, va,
            SEED + i * 101, max_epochs=22, patience=5, min_epochs=8,
            max_added_fp=0,
        )
        general_nn.append(m)
    for i in range(2):
        m, _ = v18.train_model(
            f"v22-deep-{i+1}", deep_ds, val_ds, va,
            SEED + 5000 + i * 131, max_epochs=26, patience=6, min_epochs=9,
            max_added_fp=0,
        )
        deep_nn.append(m)

    print("[11/16] score neural branches", flush=True)
    pg_val, _ = v18.ensemble_predict(general_nn, val_ds)
    pd_val, _ = v18.ensemble_predict(deep_nn, val_ds)
    pg_test, _ = v18.ensemble_predict(general_nn, test_ds)
    pd_test, _ = v18.ensemble_predict(deep_nn, test_ds)

    print("[12/16] fit multi-era precision guard", flush=True)
    xval = feature_matrix(
        va, ptext_val, pctx_val, pham_val, prep_val, prep_dis_val,
        pchar_val, pg_val, pd_val,
        ptfidf_word_val, ptfidf_char_val, pprecision_val,
        stage1_val, v16_val, near_val_pred
    )
    xtest = feature_matrix(
        te, ptext_test, pctx_test, pham_test, prep_test, prep_dis_test,
        pchar_test, pg_test, pd_test,
        ptfidf_word_test, ptfidf_char_test, pprecision_test,
        stage1_test, v16_test, near_test_pred
    )

    fit_mask, slice_id = chronological_fit_and_slices(va, n_slices=3)
    fit_rows = [va[i] for i in np.where(fit_mask)[0]]
    guard = fit_precision_guard(xval[fit_mask], fit_rows)
    score_val = guard.predict_proba(xval)[:, 1]
    score_test = guard.predict_proba(xtest)[:, 1]

    agree_val = agreement_count(
        ptext_val, pctx_val, pham_val, prep_val, pchar_val, pg_val, pd_val,
        ptfidf_word_val, ptfidf_char_val, pprecision_val,
    )
    agree_test = agreement_count(
        ptext_test, pctx_test, pham_test, prep_test, pchar_test, pg_test, pd_test,
        ptfidf_word_test, ptfidf_char_test, pprecision_test,
    )

    ultra_gate = robust_select_guard(
        va, score_val, agree_val, slice_id,
        max_total_fp=0, max_slice_fp=0, name="ultra-safe"
    )
    safe_gate = robust_select_guard(
        va, score_val, agree_val, slice_id,
        max_total_fp=1, max_slice_fp=1, name="safe"
    )
    balanced_gate = robust_select_guard(
        va, score_val, agree_val, slice_id,
        max_total_fp=3, max_slice_fp=2, name="balanced"
    )

    print("[13/16] apply calibrated guards to 10k test", flush=True)
    ultra_pred = apply_guard(base_test, score_test, agree_test, ultra_gate)
    safe_pred = apply_guard(base_test, score_test, agree_test, safe_gate)
    balanced_pred = apply_guard(base_test, score_test, agree_test, balanced_gate)

    rspamd_stats = v19.metrics(te, base_test)
    stage1_stats = v19.metrics(te, stage1_test)
    v16_stats = v19.metrics(te, v16_test)
    near_stats = v19.metrics(te, near_test_pred)
    ultra_stats = v19.metrics(te, ultra_pred)
    safe_stats = v19.metrics(te, safe_pred)
    balanced_stats = v19.metrics(te, balanced_pred)

    print("[14/16] save calibrated guard", flush=True)
    import pickle
    with open(MODELS / "precision-guard.pkl", "wb") as f:
        pickle.dump({
            "model": guard,
            "ultraGate": ultra_gate,
            "safeGate": safe_gate,
            "balancedGate": balanced_gate,
        }, f)

    calibration_count = int((slice_id >= 0).sum())
    result = {
        "version": "v22-multiera-calibration",
        "dataset": {
            "lockedTest": len(te),
            "testSpam": safe_stats["spamTotal"],
            "testHam": safe_stats["hamTotal"],
            "validation": len(va),
            "guardFit": int(fit_mask.sum()),
            "guardCalibration": calibration_count,
            "calibrationSlices": 3,
            "split": "class-stratified chronological fit + 3 later calibration eras",
        },
        "method": {
            "base": "v16 reputation/campaign + sanitized text/char + v18 neural + TF-IDF/SVM specialists",
            "precisionGuard": "weighted logistic meta-classifier",
            "hamWeightMultiplier": 14.0,
            "agreementRule": "requires agreement across sanitized hash, neural, reputation, TF-IDF word/char and conservative SVM experts",
            "syntheticEnvelopeLeakageRemoved": True,
            "safeBudget": "at most 1 FP across calibration and at most 1 in any era",
            "balancedBudget": "at most 3 FP across calibration and at most 2 in any era",
            "testLabelsUsedForTrainingOrThresholds": False,
            "individualV19OrV20TestErrorsInspected": False,
        },
        "rspamdBayes": rspamd_stats,
        "stage1": stage1_stats,
        "v16": v16_stats,
        "nearMiss": near_stats,
        "v22UltraGate": ultra_gate,
        "v22Ultra": ultra_stats,
        "v22SafeGate": safe_gate,
        "v22Safe": safe_stats,
        "v22BalancedGate": balanced_gate,
        "v22Balanced": balanced_stats,
        "warning": (
            "This remains the Enron engineering benchmark family already used "
            "for earlier iterations. It is not a production estimate; the next "
            "confidence step should be a fresh modern lockbox."
        ),
    }
    (REPORTS / "v22-calibration.json").write_text(json.dumps(result, indent=2))

    print("[15/16] write report", flush=True)
    rows = [
        ("Rspamd + Bayes", rspamd_stats),
        ("Stage 1", stage1_stats),
        ("v16", v16_stats),
        ("+ near-miss", near_stats),
        ("v22 ultra-safe", ultra_stats),
        ("v22 safe", safe_stats),
        ("v22 balanced", balanced_stats),
    ]
    md = [
        "# MailGuard v22 sanitized hard-negative ensemble benchmark",
        "",
        f"Test: **{len(te)} unique emails** "
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
        f"Ultra-safe gate: {ultra_gate}",
        f"Safe gate: {safe_gate}",
        f"Balanced gate: {balanced_gate}",
        "",
        "Thresholds are selected on three later chronological calibration eras.",
        "No arbitrary +0.015 probability margin is used.",
        "The 10k labels are not used by the v22 fitter or gate selector.",
    ])
    report = "\n".join(md) + "\n"
    (REPORTS / "v22-calibration.md").write_text(report)
    print(report, flush=True)

    print("[16/16] done", flush=True)


if __name__ == "__main__":
    main()
