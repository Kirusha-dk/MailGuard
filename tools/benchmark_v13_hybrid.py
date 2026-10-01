#!/usr/bin/env python3
import hashlib
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import benchmark_improved as b
import benchmark_v3 as v3
import benchmark_v5 as v5
import benchmark_v7_nn as n
import benchmark_v8_nn_veto as v8

REPORTS = Path("reports")
SEED = 20261001

TF_VOCAB = 1 << 15
TF_LEN = 192
TF_DIM = 48
TOKEN_RE = re.compile(r"[\w@.\-]{2,}", re.UNICODE)


def stable_hash(token):
    digest = hashlib.blake2b(token.encode("utf-8", "ignore"), digest_size=8).digest()
    return 2 + (int.from_bytes(digest, "little") % (TF_VOCAB - 2))


def seq_ids(row):
    text = b.extract_document(row["raw"]).lower()
    words = TOKEN_RE.findall(text)

    ids = [1]
    room = TF_LEN - 1

    if len(words) <= room - 20:
        chosen = words
    else:
        head = words[: room - 62]
        tail = words[-44:]
        chosen = head + ["__tail__"] + tail

    ids.extend(stable_hash("w:" + word) for word in chosen)

    context = ["a:" + re.sub(r"\W+", "_", row["action"].lower())]
    for name, score in row["symbols"][:14]:
        safe = re.sub(r"\W+", "_", name.lower())
        if score >= 2:
            context.append("sp:" + safe)
        elif score <= -2:
            context.append("sn:" + safe)
        else:
            context.append("s:" + safe)

    context_ids = [stable_hash(x) for x in context]
    if len(ids) + len(context_ids) > TF_LEN:
        ids = ids[: TF_LEN - len(context_ids)]
    ids.extend(context_ids)
    return np.asarray(ids[:TF_LEN], dtype=np.int64)


class TransformerDataset(Dataset):
    def __init__(self, rows, weights=None):
        self.tokens = [seq_ids(row) for row in rows]
        self.numeric = [n.numeric_features(row) for row in rows]
        self.labels = np.asarray([row["y"] for row in rows], dtype=np.float32)
        self.weights = (
            np.ones(len(rows), dtype=np.float32)
            if weights is None
            else np.asarray(weights, dtype=np.float32)
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.tokens[index],
            self.numeric[index],
            self.labels[index],
            self.weights[index],
        )


def tf_collate(batch):
    token_arrays, numeric, labels, weights = zip(*batch)
    length = max(len(x) for x in token_arrays)
    padded = np.zeros((len(token_arrays), length), dtype=np.int64)

    for i, arr in enumerate(token_arrays):
        padded[i, : len(arr)] = arr

    return (
        torch.from_numpy(padded),
        torch.from_numpy(np.stack(numeric)),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TF_VOCAB, TF_DIM, padding_idx=0)
        self.position = nn.Embedding(TF_LEN, TF_DIM)

        layer = nn.TransformerEncoderLayer(
            d_model=TF_DIM,
            nhead=4,
            dim_feedforward=128,
            dropout=0.18,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(TF_DIM)
        self.head = nn.Sequential(
            nn.Linear(TF_DIM + n.NUMERIC_DIM, 64),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(64, 24),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(24, 1),
        )

    def forward(self, tokens, numeric):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        x = self.embedding(tokens) + self.position(positions)
        padding = tokens.eq(0)
        x = self.encoder(x, src_key_padding_mask=padding)
        cls = self.norm(x[:, 0, :])
        return self.head(torch.cat([cls, numeric], dim=1)).squeeze(1)


def weighted_loss(logits, labels, weights):
    raw = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    return (raw * weights).sum() / torch.clamp(weights.sum(), min=1.0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_transformer(train_ds, val_ds, seed, label):
    print(f"[{label}] start", flush=True)
    set_seed(seed)

    model = TinyTransformer()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=7e-4,
        weight_decay=1.5e-4,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=72,
        shuffle=True,
        collate_fn=tf_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=144,
        shuffle=False,
        collate_fn=tf_collate,
    )

    best_state = None
    best_val = float("inf")
    stale = 0

    for epoch in range(1, 9):
        model.train()
        train_losses = []

        for tokens, numeric, labels, weights in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, numeric)
            loss = weighted_loss(logits, labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for tokens, numeric, labels, weights in val_loader:
                logits = model(tokens, numeric)
                val_losses.append(float(weighted_loss(logits, labels, weights)))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        print(
            f"[{label}] epoch={epoch:02d}/08 "
            f"train={train_loss:.5f} val={val_loss:.5f}",
            flush=True,
        )

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                print(f"[{label}] early stop", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    print(f"[{label}] done best_val={best_val:.5f}", flush=True)
    return model


def tf_predict(models, dataset):
    loader = DataLoader(
        dataset,
        batch_size=144,
        shuffle=False,
        collate_fn=tf_collate,
    )
    outputs = []

    for model in models:
        current = []
        with torch.no_grad():
            for tokens, numeric, labels, weights in loader:
                current.append(
                    torch.sigmoid(model(tokens, numeric)).cpu().numpy()
                )
        outputs.append(np.concatenate(current))

    matrix = np.vstack(outputs)
    return matrix.mean(axis=0), matrix.std(axis=0)


def mlp_matrix(models, dataset):
    loader = DataLoader(
        dataset,
        batch_size=192,
        shuffle=False,
        collate_fn=n.collate,
    )
    outputs = []

    for model in models:
        current = []
        with torch.no_grad():
            for ids, offsets, numeric, labels, weights in loader:
                current.append(
                    torch.sigmoid(model(ids, offsets, numeric)).cpu().numpy()
                )
        outputs.append(np.concatenate(current))

    return np.vstack(outputs)


def hybrid_features(
    ptext,
    pctx,
    pham_linear,
    spam_mlp_matrix,
    ham_mlp_matrix,
    transformer_spam,
    transformer_std,
):
    pmlp_spam = spam_mlp_matrix.mean(axis=0)
    pmlp_ham = ham_mlp_matrix.mean(axis=0)
    mlp_spam_std = spam_mlp_matrix.std(axis=0)
    mlp_ham_std = ham_mlp_matrix.std(axis=0)

    core = np.vstack([
        np.clip(ptext, 1e-9, 1.0),
        np.clip(pctx, 1e-9, 1.0),
        np.clip(pmlp_spam, 1e-9, 1.0),
        np.clip(1.0 - pham_linear, 1e-9, 1.0),
        np.clip(1.0 - pmlp_ham, 1e-9, 1.0),
    ])

    core_score = np.exp(np.mean(np.log(core), axis=0))
    transformer_support = np.clip(transformer_spam, 1e-6, 1.0)

    # Transformer is deliberately a minority vote. It can strengthen or weaken
    # an already strong v11-style decision but cannot dominate it.
    hybrid_score = np.power(core_score, 0.82) * np.power(
        0.45 + 0.55 * transformer_support,
        0.18,
    )

    return {
        "coreScore": core_score,
        "hybridScore": hybrid_score,
        "coreMin": core.min(axis=0),
        "mlpSpam": pmlp_spam,
        "mlpHam": pmlp_ham,
        "mlpSpamStd": mlp_spam_std,
        "mlpHamStd": mlp_ham_std,
        "transformerSpam": transformer_spam,
        "transformerStd": transformer_std,
    }


def stats(rows, pred, stage1):
    y = np.asarray([r["y"] for r in rows], dtype=bool)

    spam_total = int(y.sum())
    ham_total = int((~y).sum())
    tp = int((y & pred).sum())
    fp = int(((~y) & pred).sum())
    stage_tp = int((y & stage1).sum())
    stage_fp = int(((~y) & stage1).sum())

    return {
        "spamTotal": spam_total,
        "hamTotal": ham_total,
        "spamDetected": tp,
        "falsePositives": fp,
        "recall": tp / max(1, spam_total),
        "fpr": fp / max(1, ham_total),
        "rescuedOverStage1": tp - stage_tp,
        "addedFalsePositivesOverStage1": fp - stage_fp,
    }


def values_grid(values, fixed, start=0.45, count=28):
    quantiles = np.quantile(values, np.linspace(start, 1.0, count))
    return np.unique(np.concatenate([np.asarray(fixed), quantiles]))


def predict_gate(stage1, feat, gate):
    rescue = (
        (~stage1)
        & (feat["hybridScore"] >= gate["hybridThreshold"])
        & (feat["coreMin"] >= gate["coreMinThreshold"])
        & (feat["mlpSpam"] >= gate["mlpSpamThreshold"])
        & (feat["mlpHam"] <= gate["mlpHamThreshold"])
        & (feat["transformerSpam"] >= gate["transformerThreshold"])
        & (feat["mlpSpamStd"] <= gate["maxMlpStd"])
        & (feat["mlpHamStd"] <= gate["maxMlpStd"])
        & (feat["transformerStd"] <= gate["maxTransformerStd"])
    )
    return stage1 | rescue


def choose_gate(rows, stage1, feat, max_added_fp):
    y = np.asarray([r["y"] for r in rows], dtype=bool)
    hard = np.asarray([
        (not r["y"]) and ("hard_ham" in r["source"])
        for r in rows
    ], dtype=bool)

    base_fp = int(((~y) & stage1).sum())
    base_hard_fp = int((hard & stage1).sum())

    hybrid_grid = values_grid(
        feat["hybridScore"],
        [.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.93,.95,.97,.98,.99,.995,.999,1.000001],
    )
    core_min_grid = values_grid(
        feat["coreMin"],
        [.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.95],
        0.25,
        18,
    )
    mlp_spam_grid = np.asarray([.50,.60,.70,.75,.80,.85,.90,.93,.95,.97,.98,.99])
    mlp_ham_grid = np.asarray([.03,.05,.08,.10,.15,.20,.25,.30,.35,.40])
    tf_grid = np.asarray([.20,.30,.40,.50,.60,.70,.80,.90])
    mlp_std_grid = np.asarray([.04,.06,.08,.10,.15,.20,.30])
    tf_std_grid = np.asarray([.02,.04,.06,.08,.12,.20,.35])

    best = None

    for ht in hybrid_grid:
        mask_hybrid = feat["hybridScore"] >= ht

        for cm in core_min_grid:
            mask_core = mask_hybrid & (feat["coreMin"] >= cm)
            if not np.any(mask_core & (~stage1)):
                continue

            for ms in mlp_spam_grid:
                mask_spam = mask_core & (feat["mlpSpam"] >= ms)
                if not np.any(mask_spam & (~stage1)):
                    continue

                for mh in mlp_ham_grid:
                    mask_ham = mask_spam & (feat["mlpHam"] <= mh)
                    if not np.any(mask_ham & (~stage1)):
                        continue

                    for tt in tf_grid:
                        mask_tf = mask_ham & (feat["transformerSpam"] >= tt)
                        if not np.any(mask_tf & (~stage1)):
                            continue

                        for mstd in mlp_std_grid:
                            mask_mstd = (
                                mask_tf
                                & (feat["mlpSpamStd"] <= mstd)
                                & (feat["mlpHamStd"] <= mstd)
                            )
                            if not np.any(mask_mstd & (~stage1)):
                                continue

                            for tstd in tf_std_grid:
                                rescue = (
                                    (~stage1)
                                    & mask_mstd
                                    & (feat["transformerStd"] <= tstd)
                                )
                                pred = stage1 | rescue

                                fp = int(((~y) & pred).sum())
                                if fp - base_fp > max_added_fp:
                                    continue

                                hard_fp = int((hard & pred).sum())
                                if hard_fp > base_hard_fp:
                                    continue

                                cur = stats(rows, pred, stage1)
                                point = {
                                    **cur,
                                    "hybridThreshold": float(ht),
                                    "coreMinThreshold": float(cm),
                                    "mlpSpamThreshold": float(ms),
                                    "mlpHamThreshold": float(mh),
                                    "transformerThreshold": float(tt),
                                    "maxMlpStd": float(mstd),
                                    "maxTransformerStd": float(tstd),
                                    "hardHamFalsePositives": hard_fp,
                                }

                                key = (
                                    point["rescuedOverStage1"],
                                    -point["addedFalsePositivesOverStage1"],
                                    point["hybridThreshold"],
                                    point["coreMinThreshold"],
                                    point["mlpSpamThreshold"],
                                    -point["mlpHamThreshold"],
                                    point["transformerThreshold"],
                                    -point["maxMlpStd"],
                                    -point["maxTransformerStd"],
                                )

                                if best is None or key > best[0]:
                                    best = (key, point)

    if best is None:
        cur = stats(rows, stage1.copy(), stage1)
        return {
            **cur,
            "hybridThreshold": 1.000001,
            "coreMinThreshold": 1.000001,
            "mlpSpamThreshold": 1.000001,
            "mlpHamThreshold": 0.0,
            "transformerThreshold": 1.000001,
            "maxMlpStd": 0.0,
            "maxTransformerStd": 0.0,
            "hardHamFalsePositives": base_hard_fp,
        }

    return best[1]


def review(rows, stage1, feat):
    residual = [i for i in range(len(rows)) if not stage1[i]]
    budget = min(max(1, math.ceil(len(rows) * .01)), len(residual))

    top = sorted(
        residual,
        key=lambda i: feat["hybridScore"][i],
        reverse=True,
    )[:budget]
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
    print("v13: v11 feature ensemble + lightweight Transformer minority vote", flush=True)

    b.reset_bayes()
    b.learn(train)

    print("[1/7] scan train", flush=True)
    tr = b.scan_many(train, "v13-train")
    print("[2/7] scan validation", flush=True)
    va = b.scan_many(val, "v13-val")
    print("[3/7] scan test", flush=True)
    te = b.scan_many(test, "v13-test")

    base = b.base_metrics(te)

    xt, xv, xe, ctx_t, ctx_v, ctx_e = v3.matrices(tr, va, te)
    residual_idx = [i for i, r in enumerate(tr) if not r["rspam"]]
    residual_rows = [tr[i] for i in residual_idx]
    residual_val = [r for r in va if not r["rspam"]]

    print("[4/7] train classical experts", flush=True)
    text_models = v3.fit_ensemble(xt[residual_idx], residual_rows)
    context_models = v3.fit_ensemble(ctx_t[residual_idx], residual_rows)
    ham_linear_models = v5.fit_ham_ensemble(ctx_t[residual_idx], residual_rows)

    ptext_val = v3.ensemble_predict(text_models, xv)
    pctx_val = v3.ensemble_predict(context_models, ctx_v)
    pham_linear_val = v5.avg_prob(ham_linear_models, ctx_v)

    ptext_test = v3.ensemble_predict(text_models, xe)
    pctx_test = v3.ensemble_predict(context_models, ctx_e)
    pham_linear_test = v5.avg_prob(ham_linear_models, ctx_e)

    stage1_gate, _ = v5.choose_gate(
        va,
        ptext_val,
        pctx_val,
        pham_linear_val,
    )
    stage1_val = v5.predict_gate(
        va,
        ptext_val,
        pctx_val,
        pham_linear_val,
        stage1_gate["textThreshold"],
        stage1_gate["contextThreshold"],
        stage1_gate["hamVetoThreshold"],
    )
    stage1_test = v5.predict_gate(
        te,
        ptext_test,
        pctx_test,
        pham_linear_test,
        stage1_gate["textThreshold"],
        stage1_gate["contextThreshold"],
        stage1_gate["hamVetoThreshold"],
    )

    print("[5/7] train v11-style MLP experts", flush=True)
    spam_train_ds = n.MailDataset(
        residual_rows,
        n.sample_weights(residual_rows),
    )
    spam_val_ds = n.MailDataset(
        residual_val,
        n.sample_weights(residual_val),
    )

    ham_train_rows = v8.flipped_rows(residual_rows)
    ham_val_rows = v8.flipped_rows(residual_val)
    ham_train_ds = n.MailDataset(
        ham_train_rows,
        v8.ham_weights(residual_rows),
    )
    ham_val_ds = n.MailDataset(
        ham_val_rows,
        v8.ham_weights(residual_val),
    )

    full_val_mlp = n.MailDataset(va)
    full_test_mlp = n.MailDataset(te)

    spam_mlp_models = [
        n.train_one(spam_train_ds, spam_val_ds, SEED + 1300 + i * 97)
        for i in range(3)
    ]
    ham_mlp_models = [
        n.train_one(ham_train_ds, ham_val_ds, SEED + 6300 + i * 97)
        for i in range(3)
    ]

    spam_mlp_val = mlp_matrix(spam_mlp_models, full_val_mlp)
    ham_mlp_val = mlp_matrix(ham_mlp_models, full_val_mlp)
    spam_mlp_test = mlp_matrix(spam_mlp_models, full_test_mlp)
    ham_mlp_test = mlp_matrix(ham_mlp_models, full_test_mlp)

    print("[6/7] train lightweight Transformer vote", flush=True)
    tf_train = TransformerDataset(
        residual_rows,
        n.sample_weights(residual_rows),
    )
    tf_val = TransformerDataset(
        residual_val,
        n.sample_weights(residual_val),
    )
    full_val_tf = TransformerDataset(va)
    full_test_tf = TransformerDataset(te)

    tf_models = [
        train_transformer(
            tf_train,
            tf_val,
            SEED + 9300 + i * 211,
            f"transformer {i + 1}/2",
        )
        for i in range(2)
    ]

    ptf_val, ptf_std_val = tf_predict(tf_models, full_val_tf)
    ptf_test, ptf_std_test = tf_predict(tf_models, full_test_tf)

    feat_val = hybrid_features(
        ptext_val,
        pctx_val,
        pham_linear_val,
        spam_mlp_val,
        ham_mlp_val,
        ptf_val,
        ptf_std_val,
    )
    feat_test = hybrid_features(
        ptext_test,
        pctx_test,
        pham_linear_test,
        spam_mlp_test,
        ham_mlp_test,
        ptf_test,
        ptf_std_test,
    )

    print("[7/7] select gates and evaluate", flush=True)
    safe_gate = choose_gate(va, stage1_val, feat_val, max_added_fp=0)
    balanced_gate = choose_gate(va, stage1_val, feat_val, max_added_fp=1)

    safe_pred = predict_gate(stage1_test, feat_test, safe_gate)
    balanced_pred = predict_gate(stage1_test, feat_test, balanced_gate)

    stage1_stats = v5.stats(te, stage1_test)
    safe_test = stats(te, safe_pred, stage1_test)
    balanced_test = stats(te, balanced_pred, stage1_test)

    result = {
        "version": "v13-hybrid-transformer-vote",
        "dataset": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "testSpam": base["spamTotal"],
            "testHam": base["hamTotal"],
        },
        "architecture": {
            "stage1": "v5 precision linear gate",
            "mlpSpamModels": 3,
            "mlpHamModels": 3,
            "transformerSpamModels": 2,
            "transformerRole": "minority vote",
            "transformerLayers": 1,
            "transformerDim": TF_DIM,
            "maxSequenceLength": TF_LEN,
            "hardHamAddedFpBudget": 0,
        },
        "evaluationNote": (
            "This repository has iterated repeatedly against the same public "
            "benchmark family. Treat these numbers as engineering guidance, "
            "not a final unbiased production estimate."
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

    (REPORTS / "v13-benchmark.json").write_text(json.dumps(result, indent=2))

    md = [
        "# MailGuard v13 hybrid benchmark",
        "",
        f"Final test: {base['spamTotal']} spam + {base['hamTotal']} ham.",
        "",
        "| Mode | Spam recall | FP total | Extra spam vs stage 1 | Extra FP vs stage 1 |",
        "|---|---:|---:|---:|---:|",
        f"| Rspamd + Bayes | {base['recall']:.2%} | "
        f"{base['falsePositives']}/{base['hamTotal']} | - | - |",
        f"| Stage 1 (v5 precision) | {stage1_stats['recall']:.2%} | "
        f"{stage1_stats['falsePositives']}/{stage1_stats['hamTotal']} | 0 | 0 |",
        f"| v13 safe | {safe_test['recall']:.2%} | "
        f"{safe_test['falsePositives']}/{safe_test['hamTotal']} | "
        f"{safe_test['rescuedOverStage1']} | "
        f"{safe_test['addedFalsePositivesOverStage1']} |",
        f"| v13 balanced | {balanced_test['recall']:.2%} | "
        f"{balanced_test['falsePositives']}/{balanced_test['hamTotal']} | "
        f"{balanced_test['rescuedOverStage1']} | "
        f"{balanced_test['addedFalsePositivesOverStage1']} |",
        "",
        "Transformer is an additional vote; classical + MLP evidence remains primary.",
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
        "Note: repeated development has observed this public benchmark family; "
        "a fresh modern holdout is required for final claims.",
    ]

    report = "\n".join(md) + "\n"
    (REPORTS / "v13-benchmark.md").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
