#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from scipy.special import expit
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
MODELS = Path("models/v24")
SEED = 20261006


def tfidf_documents(rows):
    docs = []
    subjects = []
    for row in rows:
        doc = b.extract_document(row["raw"])
        if len(doc) > 70000:
            doc = doc[:70000]
        docs.append(doc)
        first = doc.split("\n", 1)[0]
        subjects.append(first.replace("subject ", "", 1))
    return docs, subjects


def fit_forensic_text_experts(train_rows, val_rows, test_rows):
    tr_docs, tr_subj = tfidf_documents(train_rows)
    va_docs, va_subj = tfidf_documents(val_rows)
    te_docs, te_subj = tfidf_documents(test_rows)
    y = np.asarray([int(bool(r["y"])) for r in train_rows], dtype=np.int32)
    w = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)

    # A modest extra ham penalty makes the branch useful for a low-FP ensemble.
    for i, row in enumerate(train_rows):
        if not row["y"]:
            w[i] *= 1.8
        elif float(row["rscore"] or 0.0) <= 3.5:
            w[i] *= 1.35

    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        min_df=2,
        max_df=0.997,
        max_features=180000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[\w@.\-]{2,}\b",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        lowercase=True,
        sublinear_tf=True,
        min_df=2,
        max_features=180000,
        analyzer="char_wb",
        ngram_range=(3, 5),
        dtype=np.float32,
    )
    subject = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        min_df=2,
        max_features=50000,
        ngram_range=(1, 3),
        token_pattern=r"(?u)\b[\w@.\-]{2,}\b",
        dtype=np.float32,
    )

    xw = word.fit_transform(tr_docs)
    xc = char.fit_transform(tr_docs)
    xs = subject.fit_transform(tr_subj)

    models = []
    scores = []
    for name, x, xv, xe, cval in [
        ("word", xw, word.transform(va_docs), word.transform(te_docs), 1.25),
        ("char", xc, char.transform(va_docs), char.transform(te_docs), 1.05),
        ("subject", xs, subject.transform(va_subj), subject.transform(te_subj), 0.85),
    ]:
        clf = LinearSVC(C=cval, class_weight=None, random_state=SEED + len(models) * 17)
        clf.fit(x, y, sample_weight=w)
        mv = np.asarray(clf.decision_function(xv), dtype=np.float64)
        me = np.asarray(clf.decision_function(xe), dtype=np.float64)

        # Keep monotonic margins while compressing extreme SVM values.
        pv = expit(mv)
        pe = expit(me)
        print(
            f"forensic-{name} val mean={float(pv.mean()):.4f} "
            f"test mean={float(pe.mean()):.4f}",
            flush=True,
        )
        models.append((name, clf))
        scores.append((pv, pe))

    return {
        "models": models,
        "vectorizers": {"word": word, "char": char, "subject": subject},
        "val": [x[0] for x in scores],
        "test": [x[1] for x in scores],
    }


def chronological_folds(rows, n_folds=5):
    fold_id = np.full(len(rows), -1, dtype=np.int32)
    for label in (0, 1):
        idx = sorted(
            [i for i, row in enumerate(rows) if int(bool(row["y"])) == label],
            key=lambda i: (
                v18.original_date(rows[i]["path"]),
                str(rows[i]["path"]),
            ),
        )
        chunks = np.array_split(np.asarray(idx, dtype=np.int64), n_folds)
        for fid, chunk in enumerate(chunks):
            fold_id[chunk] = fid
    return fold_id


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
    pword,
    pchar_tfidf,
    psubject,
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
            max(-4.0, min(4.0, rscore / 10.0)),
            max(-4.0, min(4.0, ratio)),
            math.log1p(len(row["raw"])) / 14.0,
            min(pos, 50) / 50.0,
            min(neg, 50) / 50.0,
            float("BAYES_HAM" in symbols),
            float("MIME_GOOD" in symbols),
            float("LOCAL_OUTBOUND" in symbols),
            pword[i],
            pchar_tfidf[i],
            psubject[i],
            math.sqrt(max(1e-8, pword[i] * pchar_tfidf[i])),
        ]

        # A few consensus statistics make the guard robust to one overconfident expert.
        probs = np.asarray(
            [ptext[i], pctx[i], 1.0 - pham[i], prep[i], pchar[i], pg[i], pd[i],
             pword[i], pchar_tfidf[i], psubject[i]],
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


def sample_weights(rows):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    w = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)
    for i, row in enumerate(rows):
        symbols = {name.upper() for name, _ in row["symbols"]}
        rscore = float(row["rscore"] or 0.0)
        if not row["y"]:
            # Strong ham penalty, but not so extreme that recall collapses.
            w[i] *= 9.0
            if rscore >= 2.0:
                w[i] *= 1.7
            if "BAYES_HAM" in symbols:
                w[i] *= 1.5
            if "MIME_GOOD" in symbols:
                w[i] *= 1.15
        elif rscore <= 3.5:
            w[i] *= 2.1
    return w


def fit_meta_pair(x, rows, seed):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    w = sample_weights(rows)

    linear = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.22,
            max_iter=1400,
            solver="lbfgs",
            random_state=seed,
        ),
    )
    linear.fit(x, y, logisticregression__sample_weight=w)

    tree = HistGradientBoostingClassifier(
        learning_rate=0.055,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=28,
        l2_regularization=1.8,
        random_state=seed + 1,
    )
    tree.fit(x, y, sample_weight=w)
    return linear, tree


def meta_score(models, x, raw_probs):
    linear, tree = models
    pl = np.clip(linear.predict_proba(x)[:, 1], 1e-8, 1.0)
    pt = np.clip(tree.predict_proba(x)[:, 1], 1e-8, 1.0)

    # Both models must agree. Independent branch consensus then gently
    # suppresses isolated overconfident predictions.
    pair = np.sqrt(pl * pt)
    median = np.median(raw_probs, axis=0)
    agree = np.sum(raw_probs >= 0.80, axis=0)
    score = pair * (0.72 + 0.28 * median) * (0.84 + 0.16 * agree / raw_probs.shape[0])
    return np.clip(score, 0.0, 1.0), pl, pt, agree.astype(np.int32)


def raw_probability_stack(
    ptext, pctx, pham, prep, pchar, pg, pd, pword, pchar_tfidf, psubject
):
    return np.vstack([
        np.clip(ptext, 1e-8, 1.0),
        np.clip(pctx, 1e-8, 1.0),
        np.clip(1.0 - pham, 1e-8, 1.0),
        np.clip(prep, 1e-8, 1.0),
        np.clip(pchar, 1e-8, 1.0),
        np.clip(pg, 1e-8, 1.0),
        np.clip(pd, 1e-8, 1.0),
        np.clip(pword, 1e-8, 1.0),
        np.clip(pchar_tfidf, 1e-8, 1.0),
        np.clip(psubject, 1e-8, 1.0),
    ])


def fit_ham_veto(x, rows, seed):
    # Separate model whose only job is to recognise legitimate mail that the
    # spam stack may find suspicious.  It never sees the final test labels.
    y_ham = np.asarray([0 if r["y"] else 1 for r in rows], dtype=np.int32)
    w = compute_sample_weight(class_weight="balanced", y=y_ham).astype(np.float64)
    for i, row in enumerate(rows):
        symbols = {name.upper() for name, _ in row["symbols"]}
        rscore = float(row["rscore"] or 0.0)
        if y_ham[i]:
            w[i] *= 3.2
            if rscore >= 1.5:
                w[i] *= 1.8
            if "BAYES_HAM" in symbols:
                w[i] *= 1.35
            if "MIME_GOOD" in symbols:
                w[i] *= 1.20
        elif rscore <= 3.5:
            # Do not let the veto learn to reject deep spam too aggressively.
            w[i] *= 1.35

    model = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=220,
        max_leaf_nodes=13,
        min_samples_leaf=22,
        l2_regularization=2.5,
        random_state=seed,
    )
    model.fit(x, y_ham, sample_weight=w)
    return model


def select_dual_guard(
    rows,
    spam_score,
    ham_risk,
    agreement,
    fold_id,
    target_recall,
    max_total_fp,
    max_fold_fp,
    name,
):
    y = np.asarray([bool(r["y"]) for r in rows], dtype=bool)

    score_candidates = np.unique(np.concatenate([
        np.asarray([
            .08,.10,.12,.14,.16,.18,.20,.25,.30,.35,.40,.45,.50,.55,.60,
            .65,.70,.74,.78,.82,.86,.90,.94,.97,.99
        ]),
        np.quantile(spam_score, np.linspace(.10, .995, 180)),
    ]))
    ham_candidates = np.unique(np.concatenate([
        np.asarray([
            .01,.02,.03,.04,.05,.06,.08,.10,.12,.15,.18,.20,.25,.30,.35,
            .40,.50,.60,.70,.80,.90,.98,1.0
        ]),
        np.quantile(ham_risk, np.linspace(.03, 1.0, 55)),
    ]))

    feasible = []
    fallback = []
    for min_agreement in (1, 2, 3, 4, 5):
        agree_mask = agreement >= min_agreement
        for max_ham in ham_candidates:
            eligible = agree_mask & (ham_risk <= max_ham)
            if not np.any(eligible):
                continue
            for threshold in score_candidates:
                pred = eligible & (spam_score >= threshold)
                fp_total = int(((~y) & pred).sum())
                if fp_total > max_total_fp:
                    continue
                tp_total = int((y & pred).sum())
                recall = tp_total / max(1, int(y.sum()))

                fold_stats = []
                stable = True
                recalls = []
                for fid in sorted(set(fold_id.tolist())):
                    m = fold_id == fid
                    fy = y[m]
                    fpred = pred[m]
                    fp = int(((~fy) & fpred).sum())
                    tp = int((fy & fpred).sum())
                    spam = int(fy.sum())
                    fr = tp / max(1, spam)
                    if fp > max_fold_fp:
                        stable = False
                        break
                    recalls.append(fr)
                    fold_stats.append({
                        "fold": int(fid),
                        "spam": spam,
                        "tp": tp,
                        "fp": fp,
                        "recall": fr,
                    })
                if not stable:
                    continue

                point = {
                    "name": name,
                    "threshold": float(threshold),
                    "maxHamRisk": float(max_ham),
                    "minAgreement": int(min_agreement),
                    "spamDetected": tp_total,
                    "falsePositives": fp_total,
                    "recall": recall,
                    "worstFoldRecall": min(recalls) if recalls else 0.0,
                    "meanFoldRecall": float(np.mean(recalls)) if recalls else 0.0,
                    "foldStats": fold_stats,
                }

                # If target recall is attainable, minimize false positives first.
                if recall >= target_recall:
                    feasible.append(point)
                fallback.append(point)

    if feasible:
        return min(
            feasible,
            key=lambda p: (
                p["falsePositives"],
                -p["worstFoldRecall"],
                -p["recall"],
                p["maxHamRisk"],
                -p["threshold"],
            ),
        )

    if fallback:
        return max(
            fallback,
            key=lambda p: (
                p["recall"],
                p["worstFoldRecall"],
                -p["falsePositives"],
            ),
        )

    return {
        "name": name,
        "threshold": 1.000001,
        "maxHamRisk": 0.0,
        "minAgreement": 10,
        "spamDetected": 0,
        "falsePositives": 0,
        "recall": 0.0,
        "worstFoldRecall": 0.0,
        "meanFoldRecall": 0.0,
        "foldStats": [],
    }


def apply_dual_guard(base_rspamd, spam_score, ham_risk, agreement, gate):
    return base_rspamd | (
        (spam_score >= gate["threshold"])
        & (ham_risk <= gate["maxHamRisk"])
        & (agreement >= gate["minAgreement"])
    )


def robust_select_guard(
    rows,
    raw_score,
    agreement,
    fold_id,
    max_total_fp,
    max_fold_fp,
    min_agreement_values,
    name,
):
    y = np.asarray([bool(r["y"]) for r in rows], dtype=bool)
    fixed = np.asarray([
        .30,.35,.40,.45,.50,.55,.60,.65,.70,.74,.76,.78,.80,.82,.84,.86,
        .88,.90,.92,.94,.95,.96,.97,.98,.985,.99,.995,.999,1.000001
    ])
    observed = np.unique(raw_score)
    candidates = np.unique(np.concatenate([
        fixed,
        observed,
        np.nextafter(observed, 1.0),
        np.quantile(raw_score, np.linspace(.15, 1.0, 220)),
    ]))

    best = None
    for min_agreement in min_agreement_values:
        for threshold in candidates:
            pred = (raw_score >= threshold) & (agreement >= min_agreement)
            fp_total = int(((~y) & pred).sum())
            if fp_total > max_total_fp:
                continue

            stats = []
            stable = True
            for fid in sorted(set(fold_id.tolist())):
                m = fold_id == fid
                fy = y[m]
                fpred = pred[m]
                fp = int(((~fy) & fpred).sum())
                tp = int((fy & fpred).sum())
                spam = int(fy.sum())
                if fp > max_fold_fp:
                    stable = False
                    break
                stats.append({
                    "fold": int(fid),
                    "spam": spam,
                    "tp": tp,
                    "fp": fp,
                    "recall": tp / max(1, spam),
                })
            if not stable:
                continue

            tp_total = int((y & pred).sum())
            recalls = [s["recall"] for s in stats]
            worst = min(recalls) if recalls else 0.0
            mean = float(np.mean(recalls)) if recalls else 0.0
            point = {
                "name": name,
                "threshold": float(threshold),
                "minAgreement": int(min_agreement),
                "spamDetected": tp_total,
                "falsePositives": fp_total,
                "recall": tp_total / max(1, int(y.sum())),
                "worstFoldRecall": worst,
                "meanFoldRecall": mean,
                "foldStats": stats,
            }
            # Primary objective is caught spam under the explicit FP budget;
            # stability breaks close ties rather than forcing a low-recall gate.
            key = (tp_total, round(worst, 8), -fp_total, min_agreement, threshold)
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
            "worstFoldRecall": 0.0,
            "meanFoldRecall": 0.0,
            "foldStats": [],
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
    tr = b.scan_many(train_paths, "v24-train", workers=16)
    print("[4/16] scan validation", flush=True)
    va = b.scan_many(enron_val, "v24-val", workers=16)
    print("[5/16] scan 10k test", flush=True)
    te = b.scan_many(enron_test, "v24-test", workers=16)

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
    for i in range(3):
        m, _ = v18.train_model(
            f"v24-general-{i+1}", general_ds, val_ds, va,
            SEED + i * 101, max_epochs=30, patience=7, min_epochs=10,
            max_added_fp=0,
        )
        general_nn.append(m)
    for i in range(3):
        m, _ = v18.train_model(
            f"v24-deep-{i+1}", deep_ds, val_ds, va,
            SEED + 5000 + i * 131, max_epochs=26, patience=6, min_epochs=9,
            max_added_fp=0,
        )
        deep_nn.append(m)

    print("[11/16] score neural branches", flush=True)
    pg_val, _ = v18.ensemble_predict(general_nn, val_ds)
    pd_val, _ = v18.ensemble_predict(deep_nn, val_ds)
    pg_test, _ = v18.ensemble_predict(general_nn, test_ds)
    pd_test, _ = v18.ensemble_predict(deep_nn, test_ds)

    print("[12/17] train forensic TF-IDF/SVM specialists", flush=True)
    forensic = fit_forensic_text_experts(tr, va, te)
    pword_val, pchar_tfidf_val, psubject_val = forensic["val"]
    pword_test, pchar_tfidf_test, psubject_test = forensic["test"]

    print("[13/17] cross-fit stacked precision model", flush=True)
    xval = feature_matrix(
        va, ptext_val, pctx_val, pham_val, prep_val, prep_dis_val,
        pchar_val, pg_val, pd_val, stage1_val, v16_val, near_val_pred,
        pword_val, pchar_tfidf_val, psubject_val
    )
    xtest = feature_matrix(
        te, ptext_test, pctx_test, pham_test, prep_test, prep_dis_test,
        pchar_test, pg_test, pd_test, stage1_test, v16_test, near_test_pred,
        pword_test, pchar_tfidf_test, psubject_test
    )

    raw_val = raw_probability_stack(
        ptext_val, pctx_val, pham_val, prep_val, pchar_val, pg_val, pd_val,
        pword_val, pchar_tfidf_val, psubject_val
    )
    raw_test = raw_probability_stack(
        ptext_test, pctx_test, pham_test, prep_test, pchar_test, pg_test, pd_test,
        pword_test, pchar_tfidf_test, psubject_test
    )

    fold_id = chronological_folds(va, n_folds=5)
    oof_score = np.zeros(len(va), dtype=np.float64)
    oof_linear = np.zeros(len(va), dtype=np.float64)
    oof_tree = np.zeros(len(va), dtype=np.float64)
    oof_agree = np.zeros(len(va), dtype=np.int32)
    oof_ham_risk = np.zeros(len(va), dtype=np.float64)

    for fid in range(5):
        trmask = fold_id != fid
        homask = fold_id == fid
        fit_rows = [va[i] for i in np.where(trmask)[0]]
        pair = fit_meta_pair(xval[trmask], fit_rows, SEED + 2200 + fid * 31)
        veto = fit_ham_veto(xval[trmask], fit_rows, SEED + 3300 + fid * 37)
        score, pl, pt, ag = meta_score(pair, xval[homask], raw_val[:, homask])
        ham_risk = veto.predict_proba(xval[homask])[:, 1]
        oof_score[homask] = score
        oof_linear[homask] = pl
        oof_tree[homask] = pt
        oof_agree[homask] = ag
        oof_ham_risk[homask] = ham_risk
        print(
            f"meta fold {fid+1}/5 rows={int(homask.sum())} "
            f"score_mean={float(score.mean()):.4f}",
            flush=True,
        )

    ultra_gate = select_dual_guard(
        va, oof_score, oof_ham_risk, oof_agree, fold_id,
        target_recall=0.80, max_total_fp=2, max_fold_fp=1,
        name="ultra-safe"
    )
    safe_gate = select_dual_guard(
        va, oof_score, oof_ham_risk, oof_agree, fold_id,
        target_recall=0.90, max_total_fp=6, max_fold_fp=2,
        name="safe"
    )
    target93_gate = select_dual_guard(
        va, oof_score, oof_ham_risk, oof_agree, fold_id,
        target_recall=0.93, max_total_fp=10, max_fold_fp=3,
        name="target-93"
    )
    high_recall_gate = select_dual_guard(
        va, oof_score, oof_ham_risk, oof_agree, fold_id,
        target_recall=0.95, max_total_fp=16, max_fold_fp=5,
        name="target-95"
    )

    print("[14/17] fit full stack and evaluate 10k", flush=True)
    full_pair = fit_meta_pair(xval, va, SEED + 2299)
    full_veto = fit_ham_veto(xval, va, SEED + 3399)
    score_test, pl_test, pt_test, agree_test = meta_score(
        full_pair, xtest, raw_test
    )
    ham_risk_test = full_veto.predict_proba(xtest)[:, 1]

    ultra_pred = apply_dual_guard(
        base_test, score_test, ham_risk_test, agree_test, ultra_gate
    )
    safe_pred = apply_dual_guard(
        base_test, score_test, ham_risk_test, agree_test, safe_gate
    )
    target93_pred = apply_dual_guard(
        base_test, score_test, ham_risk_test, agree_test, target93_gate
    )
    high_recall_pred = apply_dual_guard(
        base_test, score_test, ham_risk_test, agree_test, high_recall_gate
    )

    rspamd_stats = v19.metrics(te, base_test)
    stage1_stats = v19.metrics(te, stage1_test)
    v16_stats = v19.metrics(te, v16_test)
    near_stats = v19.metrics(te, near_test_pred)
    ultra_stats = v19.metrics(te, ultra_pred)
    safe_stats = v19.metrics(te, safe_pred)
    target93_stats = v19.metrics(te, target93_pred)
    high_recall_stats = v19.metrics(te, high_recall_pred)

    print("[15/17] save v24 stack", flush=True)
    import pickle
    with open(MODELS / "stacked-precision.pkl", "wb") as f:
        pickle.dump({
            "models": full_pair,
            "hamVeto": full_veto,
            "ultraGate": ultra_gate,
            "safeGate": safe_gate,
            "target93Gate": target93_gate,
            "highRecallGate": high_recall_gate,
            "forensicTextModels": forensic["models"],
            "forensicVectorizers": forensic["vectorizers"],
        }, f)

    result = {
        "version": "v24-hard-ham-veto",
        "dataset": {
            "lockedTest": len(te),
            "testSpam": safe_stats["spamTotal"],
            "testHam": safe_stats["hamTotal"],
            "validation": len(va),
            "metaFolds": 5,
            "split": "class-stratified chronological 5-fold OOF meta calibration",
        },
        "method": {
            "base": "v16 reputation/campaign + v17 char/near-miss + v18-style neural signals",
            "neuralEnsemble": "3 general + 3 deep specialists",
            "forensicText": "word TF-IDF SVM + char TF-IDF SVM + subject TF-IDF SVM",
            "meta": "weighted logistic + HistGradientBoosting geometric stack",
            "hardHamVeto": "cross-fitted independent HistGradientBoosting ham detector",
            "binaryValidationSelectedDecisionsUsedAsMetaFeatures": False,
            "hardHamWeightMultiplier": 9.0,
            "testLabelsUsedForTrainingOrThresholds": False,
            "individualTestErrorsInspected": False,
        },
        "rspamdBayes": rspamd_stats,
        "stage1": stage1_stats,
        "v16": v16_stats,
        "nearMiss": near_stats,
        "v24UltraGate": ultra_gate,
        "v24Ultra": ultra_stats,
        "v24SafeGate": safe_gate,
        "v24Safe": safe_stats,
        "v24Target93Gate": target93_gate,
        "v24Target93": target93_stats,
        "v24Target95Gate": high_recall_gate,
        "v24Target95": high_recall_stats,
        "warning": (
            "This is an engineering iteration on the existing Enron benchmark "
            "family. A fresh modern lockbox is still required for a final claim."
        ),
    }
    (REPORTS / "v24-crossfit.json").write_text(json.dumps(result, indent=2))

    print("[16/17] write report", flush=True)
    rows = [
        ("Rspamd + Bayes", rspamd_stats),
        ("Stage 1", stage1_stats),
        ("v16", v16_stats),
        ("+ near-miss", near_stats),
        ("v24 ultra-safe", ultra_stats),
        ("v24 safe", safe_stats),
        ("v24 target-93", target93_stats),
        ("v24 target-95", high_recall_stats),
    ]
    md = [
        "# MailGuard v24 hard-ham-veto benchmark",
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
        f"Target-93 gate: {target93_gate}",
        f"Target-95 gate: {high_recall_gate}",
        "",
        "Gate selection uses only 5-fold out-of-fold validation predictions.",
        "The 10k test labels are not used by the v24 fitter or gate selector.",
    ])
    report = "\n".join(md) + "\n"
    (REPORTS / "v24-crossfit.md").write_text(report)
    print(report, flush=True)

    print("[17/17] done", flush=True)


if __name__ == "__main__":
    main()
