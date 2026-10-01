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
VOCAB_SIZE = 1 << 15
MAX_LEN = 224
D_MODEL = 64
NUMERIC_DIM = 7
TOKEN_RE = re.compile(r"[\w@.\-]{2,}", re.UNICODE)


def stable_hash(token):
    digest = hashlib.blake2b(token.encode("utf-8", "ignore"), digest_size=8).digest()
    return 2 + (int.from_bytes(digest, "little") % (VOCAB_SIZE - 2))


def sequence_ids(row):
    text = b.extract_document(row["raw"]).lower()
    words = TOKEN_RE.findall(text)

    # 1 is a reserved CLS token, 0 is padding.
    ids = [1]
    room = MAX_LEN - 1

    # Keep the first part of the message because subject/header information is
    # usually near the beginning. Also keep a tail slice for unsubscribe/footer
    # and payload clues that often appear near the end.
    if len(words) <= room - 24:
        chosen = words
    else:
        head = words[: room - 88]
        tail = words[-64:]
        chosen = head + ["__tail__"] + tail

    ids.extend(stable_hash("w:" + w) for w in chosen)

    # Reserve the final positions for Rspamd context. This is a separate token
    # channel in addition to the numeric metadata branch.
    context = ["action:" + re.sub(r"\W+", "_", row["action"].lower())]
    for name, score in row["symbols"][:20]:
        safe = re.sub(r"\W+", "_", name.lower())
        if score >= 2:
            context.append("sympos:" + safe)
        elif score <= -2:
            context.append("symneg:" + safe)
        else:
            context.append("sym:" + safe)

    context_ids = [stable_hash(x) for x in context]
    if len(ids) + len(context_ids) > MAX_LEN:
        ids = ids[: MAX_LEN - len(context_ids)]
    ids.extend(context_ids)
    return np.asarray(ids[:MAX_LEN], dtype=np.int64)


class TransformerDataset(Dataset):
    def __init__(self, rows, weights=None):
        self.rows = rows
        self.tokens = [sequence_ids(r) for r in rows]
        self.numeric = [n.numeric_features(r) for r in rows]
        self.labels = np.asarray([r["y"] for r in rows], dtype=np.float32)
        self.weights = (
            np.ones(len(rows), dtype=np.float32)
            if weights is None
            else np.asarray(weights, dtype=np.float32)
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return (
            self.tokens[index],
            self.numeric[index],
            self.labels[index],
            self.weights[index],
        )


def collate(batch):
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


class MailTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=0)
        self.position = nn.Embedding(MAX_LEN, D_MODEL)

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=4,
            dim_feedforward=192,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Sequential(
            nn.Linear(D_MODEL + NUMERIC_DIM, 96),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, tokens, numeric):
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        x = self.embedding(tokens) + self.position(positions)
        padding = tokens.eq(0)
        x = self.encoder(x, src_key_padding_mask=padding)
        cls = self.norm(x[:, 0, :])
        return self.head(torch.cat([cls, numeric], dim=1)).squeeze(1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def weighted_loss(logits, labels, weights):
    raw = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    return (raw * weights).sum() / torch.clamp(weights.sum(), min=1.0)


def train_one(train_ds, val_ds, seed, label="model"):\n    print(f"[{label}] start", flush=True)
    set_seed(seed)
    model = MailTransformer()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        weight_decay=8e-5,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        collate_fn=collate,
    )

    best_state = None
    best_val = float("inf")
    stale = 0

    for epoch in range(1, 9):
        model.train()
        losses = []

        for tokens, numeric, labels, weights in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, numeric)
            loss = weighted_loss(logits, labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5)
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for tokens, numeric, labels, weights in val_loader:
                logits = model(tokens, numeric)
                val_losses.append(float(weighted_loss(logits, labels, weights)))

        train_loss = float(np.mean(losses))
        val_loss = float(np.mean(val_losses))
        print(
            f"[{label}] seed={seed} epoch={epoch:02d} "
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
            if stale >= 3:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def model_matrix(models, dataset):
    loader = DataLoader(
        dataset,
        batch_size=96,
        shuffle=False,
        collate_fn=collate,
    )
    outputs = []

    for model in models:
        current = []
        with torch.no_grad():
            for tokens, numeric, labels, weights in loader:
                current.append(torch.sigmoid(model(tokens, numeric)).cpu().numpy())
        outputs.append(np.concatenate(current))

    return np.vstack(outputs)


def combined_features(ptext, pctx, pham_lin, spam_matrix, ham_matrix):
    pspam = spam_matrix.mean(axis=0)
    pham = ham_matrix.mean(axis=0)
    spam_std = spam_matrix.std(axis=0)
    ham_std = ham_matrix.std(axis=0)

    evidence = np.vstack([
        np.clip(ptext, 1e-9, 1.0),
        np.clip(pctx, 1e-9, 1.0),
        np.clip(pspam, 1e-9, 1.0),
        np.clip(1.0 - pham_lin, 1e-9, 1.0),
        np.clip(1.0 - pham, 1e-9, 1.0),
    ])
    score = np.exp(np.mean(np.log(evidence), axis=0))

    return {
        "score": score,
        "minEvidence": evidence.min(axis=0),
        "transformerSpam": pspam,
        "transformerHam": pham,
        "spamStd": spam_std,
        "hamStd": ham_std,
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


def candidate(values, fixed, qcount=28):
    quantiles = np.quantile(values, np.linspace(0.45, 1.0, qcount))
    return np.unique(np.concatenate([np.asarray(fixed), quantiles]))


def prediction(stage1, feat, gate):
    rescue = (
        (~stage1)
        & (feat["score"] >= gate["scoreThreshold"])
        & (feat["minEvidence"] >= gate["minEvidenceThreshold"])
        & (feat["transformerSpam"] >= gate["transformerSpamThreshold"])
        & (feat["transformerHam"] <= gate["transformerHamThreshold"])
        & (feat["spamStd"] <= gate["maxTransformerSpamStd"])
        & (feat["hamStd"] <= gate["maxTransformerHamStd"])
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

    score_grid = candidate(
        feat["score"],
        [.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.93,.95,.97,.98,.99,.995,.999,1.000001],
    )
    evidence_grid = candidate(
        feat["minEvidence"],
        [.20,.30,.40,.50,.60,.70,.75,.80,.85,.90,.93,.95,.97,.99,1.000001],
        20,
    )
    tspam_grid = np.asarray([.55,.65,.75,.80,.85,.90,.93,.95,.97,.98,.99,.995])
    tham_grid = np.asarray([.01,.02,.03,.05,.08,.10,.15,.20,.25,.30,.40])
    std_grid = np.asarray([.01,.02,.035,.05,.075,.10,.15,.25])

    best = None

    for score_threshold in score_grid:
        m_score = feat["score"] >= score_threshold
        for evidence_threshold in evidence_grid:
            m_evidence = m_score & (feat["minEvidence"] >= evidence_threshold)
            if not np.any(m_evidence & (~stage1)):
                continue

            for spam_threshold in tspam_grid:
                m_spam = m_evidence & (feat["transformerSpam"] >= spam_threshold)
                if not np.any(m_spam & (~stage1)):
                    continue

                for ham_threshold in tham_grid:
                    m_ham = m_spam & (feat["transformerHam"] <= ham_threshold)
                    if not np.any(m_ham & (~stage1)):
                        continue

                    for max_std in std_grid:
                        rescue = (
                            (~stage1)
                            & m_ham
                            & (feat["spamStd"] <= max_std)
                            & (feat["hamStd"] <= max_std)
                        )
                        pred = stage1 | rescue
                        fp = int(((~y) & pred).sum())
                        added_fp = fp - base_fp
                        if added_fp > max_added_fp:
                            continue

                        hard_fp = int((hard & pred).sum())
                        if hard_fp > base_hard_fp:
                            continue

                        cur = stats(rows, pred, stage1)
                        point = {
                            **cur,
                            "scoreThreshold": float(score_threshold),
                            "minEvidenceThreshold": float(evidence_threshold),
                            "transformerSpamThreshold": float(spam_threshold),
                            "transformerHamThreshold": float(ham_threshold),
                            "maxTransformerSpamStd": float(max_std),
                            "maxTransformerHamStd": float(max_std),
                            "hardHamFalsePositives": hard_fp,
                        }

                        key = (
                            point["rescuedOverStage1"],
                            -point["addedFalsePositivesOverStage1"],
                            point["scoreThreshold"],
                            point["minEvidenceThreshold"],
                            point["transformerSpamThreshold"],
                            -point["transformerHamThreshold"],
                            -point["maxTransformerSpamStd"],
                        )
                        if best is None or key > best[0]:
                            best = (key, point)

    if best is None:
        cur = stats(rows, stage1.copy(), stage1)
        return {
            **cur,
            "scoreThreshold": 1.000001,
            "minEvidenceThreshold": 1.000001,
            "transformerSpamThreshold": 1.000001,
            "transformerHamThreshold": 0.0,
            "maxTransformerSpamStd": 0.0,
            "maxTransformerHamStd": 0.0,
            "hardHamFalsePositives": base_hard_fp,
        }

    return best[1]


def review(rows, stage1, feat):
    residual = [i for i in range(len(rows)) if not stage1[i]]
    budget = min(max(1, math.ceil(len(rows) * .01)), len(residual))

    ordered = sorted(
        residual,
        key=lambda i: feat["score"][i],
        reverse=True,
    )[:budget]
    top_spam = sum(rows[i]["y"] for i in ordered)

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
    print("v12.1: fast Transformer encoder + metadata + ham veto", flush=True)

    b.reset_bayes()
    b.learn(train)

    print("[1/6] scanning train", flush=True)\n    tr = b.scan_many(train, "v12.1-train")
    print("[2/6] scanning validation", flush=True)\n    va = b.scan_many(val, "v12.1-val")
    print("[3/6] scanning test", flush=True)\n    te = b.scan_many(test, "v12.1-test")
    base = b.base_metrics(te)

    xt, xv, xe, ctx_t, ctx_v, ctx_e = v3.matrices(tr, va, te)
    residual_idx = [i for i, r in enumerate(tr) if not r["rspam"]]
    residual_rows = [tr[i] for i in residual_idx]

    # Conservative first stage from v5.
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
        va,
        ptext_val,
        pctx_val,
        pham_lin_val,
        stage1_gate["textThreshold"],
        stage1_gate["contextThreshold"],
        stage1_gate["hamVetoThreshold"],
    )
    stage1_test = v5.predict_gate(
        te,
        ptext_test,
        pctx_test,
        pham_lin_test,
        stage1_gate["textThreshold"],
        stage1_gate["contextThreshold"],
        stage1_gate["hamVetoThreshold"],
    )

    residual_val = [r for r in va if not r["rspam"]]

    spam_train = TransformerDataset(
        residual_rows,
        n.sample_weights(residual_rows),
    )
    spam_val = TransformerDataset(
        residual_val,
        n.sample_weights(residual_val),
    )

    ham_train_rows = v8.flipped_rows(residual_rows)
    ham_val_rows = v8.flipped_rows(residual_val)
    ham_train = TransformerDataset(
        ham_train_rows,
        v8.ham_weights(residual_rows),
    )
    ham_val = TransformerDataset(
        ham_val_rows,
        v8.ham_weights(residual_val),
    )

    full_val = TransformerDataset(va)
    full_test = TransformerDataset(te)

    print("[4/6] training spam Transformer ensemble", flush=True)
    spam_models = []
    for i in range(2):
        spam_models.append(
            train_one(
                spam_train,
                spam_val,
                SEED + 1200 + i * 211,
                label=f"spam {i + 1}/2",
            )
        )

    print("[5/6] training ham Transformer ensemble", flush=True)
    ham_models = []
    for i in range(2):
        ham_models.append(
            train_one(
                ham_train,
                ham_val,
                SEED + 7200 + i * 211,
                label=f"ham {i + 1}/2",
            )
        )

    print("[6/6] evaluating gates", flush=True)

    spam_matrix_val = model_matrix(spam_models, full_val)
    ham_matrix_val = model_matrix(ham_models, full_val)
    spam_matrix_test = model_matrix(spam_models, full_test)
    ham_matrix_test = model_matrix(ham_models, full_test)

    feat_val = combined_features(
        ptext_val,
        pctx_val,
        pham_lin_val,
        spam_matrix_val,
        ham_matrix_val,
    )
    feat_test = combined_features(
        ptext_test,
        pctx_test,
        pham_lin_test,
        spam_matrix_test,
        ham_matrix_test,
    )

    safe_gate = choose_gate(va, stage1_val, feat_val, max_added_fp=0)
    balanced_gate = choose_gate(va, stage1_val, feat_val, max_added_fp=1)

    safe_pred = prediction(stage1_test, feat_test, safe_gate)
    balanced_pred = prediction(stage1_test, feat_test, balanced_gate)

    stage1_stats = v5.stats(te, stage1_test)
    safe_test = stats(te, safe_pred, stage1_test)
    balanced_test = stats(te, balanced_pred, stage1_test)

    result = {
        "version": "v12.1-fast-transformer-rescue",
        "dataset": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "testSpam": base["spamTotal"],
            "testHam": base["hamTotal"],
        },
        "architecture": {
            "stage1": "v5 precision linear gate",
            "transformerSpamEnsemble": 2,
            "transformerHamEnsemble": 2,
            "transformerLayers": 1,
            "dModel": D_MODEL,
            "heads": 4,
            "maxSequenceLength": MAX_LEN,
            "metadataBranch": True,
            "uncertaintyGate": True,
            "pretrained": False,
            "testUntouchedWithinRun": True,
        },
        "rspamdBayes": base,
        "stage1ValidationGate": stage1_gate,
        "stage1Test": stage1_stats,
        "safeValidationGate": safe_gate,
        "safeTest": safe_test,
        "balancedValidationGate": balanced_gate,
        "balancedTest": balanced_test,
        "review1pctAfterStage1": review(te, stage1_test, feat_test),
    }

    (REPORTS / "v12_1-benchmark.json").write_text(json.dumps(result, indent=2))

    md = [
        "# MailGuard v12.1 fast Transformer rescue benchmark",
        "",
        f"Final test: {base['spamTotal']} spam + {base['hamTotal']} ham.",
        "",
        "| Mode | Spam recall | FP total | Extra spam vs stage 1 | Extra FP vs stage 1 |",
        "|---|---:|---:|---:|---:|",
        f"| Rspamd + Bayes | {base['recall']:.2%} | {base['falsePositives']}/{base['hamTotal']} | - | - |",
        f"| Stage 1 (v5 precision) | {stage1_stats['recall']:.2%} | "
        f"{stage1_stats['falsePositives']}/{stage1_stats['hamTotal']} | 0 | 0 |",
        f"| v12.1 safe | {safe_test['recall']:.2%} | "
        f"{safe_test['falsePositives']}/{safe_test['hamTotal']} | "
        f"{safe_test['rescuedOverStage1']} | {safe_test['addedFalsePositivesOverStage1']} |",
        f"| v12.1 balanced | {balanced_test['recall']:.2%} | "
        f"{balanced_test['falsePositives']}/{balanced_test['hamTotal']} | "
        f"{balanced_test['rescuedOverStage1']} | {balanced_test['addedFalsePositivesOverStage1']} |",
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
    ]
    text = "\n".join(md) + "\n"
    (REPORTS / "v12_1-benchmark.md").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
