#!/usr/bin/env python3
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

import benchmark_improved as b
import benchmark_v3 as v3
import benchmark_v5 as v5

REPORTS = Path("reports")
SEED = 20261001

URL_RE = re.compile(r"https?://[^\s<>'\"()]+", re.I)
TOKEN_RE = re.compile(r"[\w@.\-]{3,}", re.UNICODE)
IP_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
NUMBER_RE = re.compile(r"\b\d+\b")
HEX_RE = re.compile(r"\b[0-9a-f]{8,}\b", re.I)
SPACE_RE = re.compile(r"\s+")


def safe_domain(value):
    value = (value or "").strip().lower().strip("<>")
    if "@" in value:
        value = value.rsplit("@", 1)[1]
    value = value.strip(".")
    if not value or "." not in value:
        return ""
    return value[-180:]


def normalize_subject(subject):
    text = str(subject or "").lower()
    text = URL_RE.sub(" __url__ ", text)
    text = HEX_RE.sub(" __hex__ ", text)
    text = NUMBER_RE.sub(" __num__ ", text)
    text = re.sub(r"[^\w@.\-]+", " ", text)
    return SPACE_RE.sub(" ", text).strip()[:240]


def extract_ip24(headers):
    for value in headers:
        for ip in IP_RE.findall(str(value)):
            parts = ip.split(".")
            try:
                nums = [int(x) for x in parts]
            except ValueError:
                continue
            if len(nums) != 4 or any(x > 255 for x in nums):
                continue
            if nums[0] in (10, 127) or (nums[0] == 192 and nums[1] == 168):
                continue
            if nums[0] == 172 and 16 <= nums[1] <= 31:
                continue
            return ".".join(parts[:3]) + ".0/24"
    return ""


def simhash_bucket(text):
    tokens = TOKEN_RE.findall(text.lower())[:600]
    if not tokens:
        return ""

    counts = Counter(tokens)
    acc = [0] * 64

    for token, weight in counts.most_common(160):
        digest = hashlib.blake2b(
            token.encode("utf-8", "ignore"),
            digest_size=8,
        ).digest()
        bits = int.from_bytes(digest, "little")
        w = min(weight, 4)
        for bit in range(64):
            acc[bit] += w if ((bits >> bit) & 1) else -w

    value = 0
    for bit, score in enumerate(acc):
        if score >= 0:
            value |= 1 << bit

    # 16-bit coarse bucket: similar campaign templates often collide here,
    # while exact content is not stored.
    return f"{value >> 48:04x}"


def parse_mail(row):
    raw = row["raw"]
    try:
        msg = BytesParser(policy=policy.compat32).parsebytes(raw)
    except Exception:
        msg = None

    if msg is None:
        sender = ""
        sender_domain = ""
        return_domain = ""
        msgid_domain = ""
        subject = ""
        received = []
    else:
        sender = parseaddr(str(msg.get("from", "")))[1].lower()[:220]
        sender_domain = safe_domain(sender)
        return_domain = safe_domain(parseaddr(str(msg.get("return-path", "")))[1])

        msgid = str(msg.get("message-id", "")).strip("<>")
        msgid_domain = safe_domain(msgid.rsplit("@", 1)[1] if "@" in msgid else "")

        subject = normalize_subject(msg.get("subject", ""))
        received = msg.get_all("received", []) or []

    doc = b.extract_document(raw)
    urls = []
    for match in URL_RE.findall(doc):
        try:
            host = (urlparse(match).hostname or "").lower().strip(".")
        except Exception:
            host = ""
        if host and "." in host:
            urls.append(host[-180:])

    url_domains = tuple(sorted(set(urls))[:20])
    ip24 = extract_ip24(received)

    symbol_names = {name.upper() for name, _ in row.get("symbols", [])}

    def has_piece(piece):
        return any(piece in name for name in symbol_names)

    return {
        "sender": sender,
        "senderDomain": sender_domain,
        "returnDomain": return_domain,
        "messageIdDomain": msgid_domain,
        "ip24": ip24,
        "subject": subject,
        "campaign": simhash_bucket(subject + "\n" + doc[:12000]),
        "urlDomains": url_domains,
        "urlCount": len(urls),
        "dkimPass": float(has_piece("DKIM") and has_piece("ALLOW")),
        "dkimBad": float(has_piece("DKIM") and (has_piece("REJECT") or has_piece("FAIL"))),
        "spfPass": float(has_piece("SPF") and has_piece("ALLOW")),
        "spfBad": float(has_piece("SPF") and (has_piece("FAIL") or has_piece("REJECT"))),
        "dmarcBad": float(has_piece("DMARC") and (has_piece("FAIL") or has_piece("REJECT"))),
    }


class ReputationDB:
    def __init__(self):
        self.global_spam = 0
        self.global_total = 0
        self.tables = {
            "sender": defaultdict(lambda: [0, 0]),
            "senderDomain": defaultdict(lambda: [0, 0]),
            "returnDomain": defaultdict(lambda: [0, 0]),
            "messageIdDomain": defaultdict(lambda: [0, 0]),
            "ip24": defaultdict(lambda: [0, 0]),
            "subject": defaultdict(lambda: [0, 0]),
            "campaign": defaultdict(lambda: [0, 0]),
            "url": defaultdict(lambda: [0, 0]),
        }

    def add(self, meta, label):
        y = int(bool(label))
        self.global_spam += y
        self.global_total += 1

        for field in (
            "sender",
            "senderDomain",
            "returnDomain",
            "messageIdDomain",
            "ip24",
            "subject",
            "campaign",
        ):
            key = meta[field]
            if key:
                slot = self.tables[field][key]
                slot[0] += y
                slot[1] += 1

        for domain in meta["urlDomains"]:
            slot = self.tables["url"][domain]
            slot[0] += y
            slot[1] += 1

    def fit(self, metas, labels):
        for meta, label in zip(metas, labels):
            self.add(meta, label)
        return self

    @property
    def prior(self):
        return (self.global_spam + 1.0) / (self.global_total + 2.0)

    def rate_count(self, table, key, strength):
        if not key:
            return self.prior, 0.0, 1.0

        spam, total = self.tables[table].get(key, (0, 0))
        rate = (spam + strength * self.prior) / (total + strength)
        known = 0.0 if total else 1.0
        return rate, math.log1p(total), known

    def vector(self, meta, row):
        values = []

        specs = (
            ("sender", meta["sender"], 2.0),
            ("senderDomain", meta["senderDomain"], 5.0),
            ("returnDomain", meta["returnDomain"], 5.0),
            ("messageIdDomain", meta["messageIdDomain"], 5.0),
            ("ip24", meta["ip24"], 6.0),
            ("subject", meta["subject"], 3.0),
            ("campaign", meta["campaign"], 4.0),
        )

        for table, key, strength in specs:
            rate, count, unknown = self.rate_count(table, key, strength)
            values.extend([rate, count / 6.0, unknown])

        url_rates = []
        url_counts = []
        known_urls = 0

        for domain in meta["urlDomains"]:
            rate, count, unknown = self.rate_count("url", domain, 4.0)
            url_rates.append(rate)
            url_counts.append(count)
            known_urls += int(not unknown)

        if url_rates:
            values.extend([
                max(url_rates),
                float(np.mean(url_rates)),
                max(url_counts) / 6.0,
                known_urls / max(1, len(url_rates)),
            ])
        else:
            values.extend([self.prior, self.prior, 0.0, 0.0])

        required = row.get("required", 0.0) or 0.0
        rscore = row.get("rscore", 0.0) or 0.0

        values.extend([
            min(4.0, max(-4.0, rscore / 10.0)),
            min(4.0, max(-4.0, rscore / required)) if required else 0.0,
            math.log1p(len(row["raw"])) / 12.0,
            min(meta["urlCount"], 20) / 20.0,
            meta["dkimPass"],
            meta["dkimBad"],
            meta["spfPass"],
            meta["spfBad"],
            meta["dmarcBad"],
            self.prior,
        ])

        return np.asarray(values, dtype=np.float64)


def reputation_features_oof(rows, metas):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    out = None

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(np.zeros(len(rows)), y), 1):
        print(f"[4/9] reputation cross-fit {fold}/5", flush=True)
        db = ReputationDB().fit(
            [metas[i] for i in fit_idx],
            y[fit_idx],
        )
        block = np.vstack([db.vector(metas[i], rows[i]) for i in hold_idx])

        if out is None:
            out = np.zeros((len(rows), block.shape[1]), dtype=np.float64)

        out[hold_idx] = block

    return out


def reputation_features(db, rows, metas):
    return np.vstack([db.vector(meta, row) for row, meta in zip(rows, metas)])


def rep_weights(rows):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)

    for i, row in enumerate(rows):
        if not row["y"] and "hard_ham" in row["source"]:
            weights[i] *= 10.0
        if not row["y"] and row["rscore"] >= 4.0:
            weights[i] *= 5.0
        if row["y"] and row["rscore"] <= 4.0:
            weights[i] *= 2.0

    return weights


def fit_reputation_models(x, rows):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    weights = rep_weights(rows)

    logistic = LogisticRegression(
        C=0.7,
        max_iter=800,
        solver="lbfgs",
        random_state=SEED,
    )
    logistic.fit(x, y, sample_weight=weights)

    tree = HistGradientBoostingClassifier(
        learning_rate=0.045,
        max_iter=180,
        max_leaf_nodes=15,
        max_depth=5,
        min_samples_leaf=14,
        l2_regularization=3.0,
        random_state=SEED,
    )
    tree.fit(x, y, sample_weight=weights)

    return logistic, tree


def rep_prob(models, x):
    probs = [model.predict_proba(x)[:, 1] for model in models]
    a = np.clip(probs[0], 1e-6, 1.0)
    bprob = np.clip(probs[1], 1e-6, 1.0)
    return np.sqrt(a * bprob), np.abs(a - bprob)


def combined_features(ptext, pctx, pham, prep, rep_disagreement):
    evidence = np.vstack([
        np.clip(ptext, 1e-8, 1.0),
        np.clip(pctx, 1e-8, 1.0),
        np.clip(prep, 1e-8, 1.0),
        np.clip(1.0 - pham, 1e-8, 1.0),
    ])

    return {
        "score": np.exp(np.mean(np.log(evidence), axis=0)),
        "consensus": evidence.min(axis=0),
        "median": np.median(evidence, axis=0),
        "rep": prep,
        "repDisagreement": rep_disagreement,
        "text": ptext,
        "ctx": pctx,
        "ham": pham,
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


def split_validation(rows):
    a = np.zeros(len(rows), dtype=bool)
    bmask = np.zeros(len(rows), dtype=bool)
    groups = {}

    for i, row in enumerate(rows):
        groups.setdefault(row["source"], []).append(i)

    for idxs in groups.values():
        for j, idx in enumerate(idxs):
            (a if j % 2 == 0 else bmask)[idx] = True

    return a, bmask


def grid(values, fixed, start=0.30, count=28):
    q = np.quantile(values, np.linspace(start, 1.0, count))
    return np.unique(np.concatenate([np.asarray(fixed), q]))


def predict(stage1, feat, gate):
    rescue = (
        (~stage1)
        & (feat["score"] >= gate["scoreThreshold"])
        & (feat["consensus"] >= gate["consensusThreshold"])
        & (feat["median"] >= gate["medianThreshold"])
        & (feat["rep"] >= gate["reputationThreshold"])
        & (feat["text"] >= gate["textThreshold"])
        & (feat["ctx"] >= gate["contextThreshold"])
        & (feat["ham"] <= gate["hamVetoThreshold"])
        & (feat["repDisagreement"] <= gate["maxReputationDisagreement"])
    )
    return stage1 | rescue


def choose_gate(rows, stage1, feat, safe):
    """Fast deterministic reputation gate search.

    The original v16 searched the full Cartesian product of eight thresholds.
    This version anchors gates to real validation spam examples, small groups
    of them, and a compact quantile sweep. The same safe/balanced FP
    constraints remain enforced.
    """
    y = np.asarray([r["y"] for r in rows], dtype=bool)
    ham = ~y
    hard = np.asarray([
        (not r["y"]) and ("hard_ham" in r["source"])
        for r in rows
    ], dtype=bool)

    half_a, half_b = split_validation(rows)

    base_fp = int((ham & stage1).sum())
    base_hard = int((hard & stage1).sum())
    base_a = int((ham & stage1 & half_a).sum())
    base_b = int((ham & stage1 & half_b).sum())

    residual_spam = np.where((~stage1) & y)[0]
    best = None
    seen = set()

    def consider(gate):
        nonlocal best

        gate_key = (
            round(gate["scoreThreshold"], 7),
            round(gate["consensusThreshold"], 7),
            round(gate["medianThreshold"], 7),
            round(gate["reputationThreshold"], 7),
            round(gate["textThreshold"], 7),
            round(gate["contextThreshold"], 7),
            round(gate["hamVetoThreshold"], 7),
            round(gate["maxReputationDisagreement"], 7),
        )
        if gate_key in seen:
            return
        seen.add(gate_key)

        pred = predict(stage1, feat, gate)

        fp = int((ham & pred).sum())
        hard_fp = int((hard & pred).sum())
        a_fp = int((ham & pred & half_a).sum())
        b_fp = int((ham & pred & half_b).sum())

        if hard_fp > base_hard:
            return

        if safe:
            if a_fp > base_a or b_fp > base_b:
                return
        else:
            if fp - base_fp > 1:
                return
            if a_fp - base_a > 1 or b_fp - base_b > 1:
                return

        cur = stats(rows, pred, stage1)
        point = {
            **cur,
            **gate,
            "hardHamFalsePositives": hard_fp,
            "validationHalfAAddedFp": a_fp - base_a,
            "validationHalfBAddedFp": b_fp - base_b,
        }

        rank = (
            point["rescuedOverStage1"],
            -point["addedFalsePositivesOverStage1"],
            point["scoreThreshold"],
            point["reputationThreshold"],
            point["consensusThreshold"],
            point["medianThreshold"],
            -point["hamVetoThreshold"],
            -point["maxReputationDisagreement"],
        )

        if best is None or rank > best[0]:
            best = (rank, point)

    relaxations = (1.0, 0.97, 0.92, 0.85)

    # Exact and slightly relaxed gates induced by each missed validation spam.
    for idx in residual_spam:
        for relax in relaxations:
            consider({
                "scoreThreshold": float(feat["score"][idx] * relax),
                "consensusThreshold": float(feat["consensus"][idx] * relax),
                "medianThreshold": float(feat["median"][idx] * relax),
                "reputationThreshold": float(feat["rep"][idx] * relax),
                "textThreshold": float(feat["text"][idx] * relax),
                "contextThreshold": float(feat["ctx"][idx] * relax),
                "hamVetoThreshold": float(
                    min(1.0, feat["ham"][idx] / max(relax, 1e-6))
                ),
                "maxReputationDisagreement": float(
                    min(
                        1.0,
                        feat["repDisagreement"][idx] / max(relax, 1e-6),
                    )
                ),
            })

    # Search gates capable of rescuing small groups together.
    rnd = np.random.default_rng(SEED + (16 if safe else 116))
    if len(residual_spam):
        for _ in range(22000):
            size = int(rnd.choice([2, 3, 4], p=[0.52, 0.34, 0.14]))
            chosen = rnd.choice(
                residual_spam,
                size=min(size, len(residual_spam)),
                replace=False,
            )
            relax = float(rnd.choice([1.0, 0.97, 0.93, 0.88]))

            consider({
                "scoreThreshold": float(np.min(feat["score"][chosen]) * relax),
                "consensusThreshold": float(
                    np.min(feat["consensus"][chosen]) * relax
                ),
                "medianThreshold": float(np.min(feat["median"][chosen]) * relax),
                "reputationThreshold": float(np.min(feat["rep"][chosen]) * relax),
                "textThreshold": float(np.min(feat["text"][chosen]) * relax),
                "contextThreshold": float(np.min(feat["ctx"][chosen]) * relax),
                "hamVetoThreshold": float(
                    min(
                        1.0,
                        np.max(feat["ham"][chosen]) / max(relax, 1e-6),
                    )
                ),
                "maxReputationDisagreement": float(
                    min(
                        1.0,
                        np.max(feat["repDisagreement"][chosen])
                        / max(relax, 1e-6),
                    )
                ),
            })

    # Compact quantile sweep for gates not neatly anchored to one example.
    if len(residual_spam):
        for q in np.linspace(0.05, 0.95, 19):
            consider({
                "scoreThreshold": float(
                    np.quantile(feat["score"][residual_spam], q)
                ),
                "consensusThreshold": float(
                    np.quantile(feat["consensus"][residual_spam], q)
                ),
                "medianThreshold": float(
                    np.quantile(feat["median"][residual_spam], q)
                ),
                "reputationThreshold": float(
                    np.quantile(feat["rep"][residual_spam], q)
                ),
                "textThreshold": float(
                    np.quantile(feat["text"][residual_spam], q)
                ),
                "contextThreshold": float(
                    np.quantile(feat["ctx"][residual_spam], q)
                ),
                "hamVetoThreshold": float(
                    np.quantile(feat["ham"][residual_spam], 1.0 - q * 0.8)
                ),
                "maxReputationDisagreement": float(
                    np.quantile(
                        feat["repDisagreement"][residual_spam],
                        1.0 - q * 0.8,
                    )
                ),
            })

    print(
        f"fast reputation gate safe={safe} "
        f"candidates={len(seen)} "
        f"best_rescues={0 if best is None else best[1]['rescuedOverStage1']}",
        flush=True,
    )

    if best is not None:
        return best[1]

    cur = stats(rows, stage1.copy(), stage1)
    return {
        **cur,
        "scoreThreshold": 1.000001,
        "consensusThreshold": 1.000001,
        "medianThreshold": 1.000001,
        "reputationThreshold": 1.000001,
        "textThreshold": 1.000001,
        "contextThreshold": 1.000001,
        "hamVetoThreshold": 0.0,
        "maxReputationDisagreement": 0.0,
        "hardHamFalsePositives": base_hard,
        "validationHalfAAddedFp": 0,
        "validationHalfBAddedFp": 0,
    }

def review(rows, stage1, feat):
    residual = [i for i in range(len(rows)) if not stage1[i]]
    budget = min(max(1, math.ceil(len(rows) * .01)), len(residual))

    top = sorted(
        residual,
        key=lambda i: feat["score"][i],
        reverse=True,
    )[:budget]
    top_spam = sum(rows[i]["y"] for i in top)

    rnd = random.Random(SEED)
    values = []

    for _ in range(2000):
        sample = rnd.sample(residual, budget)
        values.append(sum(rows[i]["y"] for i in sample))

    mean = float(np.mean(values))

    return {
        "budget": budget,
        "topRiskSpamFound": int(top_spam),
        "randomSpamFoundMean": mean,
        "lift": top_spam / mean if mean else None,
    }


def main():
    REPORTS.mkdir(exist_ok=True)
    b.wait_rspamd()

    groups = b.prepare()
    train, val, test = v3.build_splits(groups)

    print("TRAIN", len(train), "VAL", len(val), "TEST", len(test), flush=True)
    print("v16: local reputation + campaign intelligence", flush=True)

    b.reset_bayes()
    b.learn(train)

    print("[1/9] scan train", flush=True)
    tr = b.scan_many(train, "v16-train")
    print("[2/9] scan validation", flush=True)
    va = b.scan_many(val, "v16-val")
    print("[3/9] scan test", flush=True)
    te = b.scan_many(test, "v16-test")

    base = b.base_metrics(te)

    xt, xv, xe, ctx_t, ctx_v, ctx_e = v3.matrices(tr, va, te)
    residual_idx = [i for i, row in enumerate(tr) if not row["rspam"]]
    residual_rows = [tr[i] for i in residual_idx]

    print("[4/9] parse local identity/campaign metadata", flush=True)
    residual_meta = [parse_mail(row) for row in residual_rows]
    val_meta = [parse_mail(row) for row in va]
    test_meta = [parse_mail(row) for row in te]

    xrep_train = reputation_features_oof(residual_rows, residual_meta)

    print("[5/9] build training-only reputation database", flush=True)
    y_residual = np.asarray([row["y"] for row in residual_rows], dtype=np.int32)
    rep_db = ReputationDB().fit(residual_meta, y_residual)

    xrep_val = reputation_features(rep_db, va, val_meta)
    xrep_test = reputation_features(rep_db, te, test_meta)

    print("[6/9] train reputation experts", flush=True)
    rep_models = fit_reputation_models(xrep_train, residual_rows)
    prep_val, prep_dis_val = rep_prob(rep_models, xrep_val)
    prep_test, prep_dis_test = rep_prob(rep_models, xrep_test)

    print("[7/9] train stable text/context baseline", flush=True)
    text_models = v3.fit_ensemble(xt[residual_idx], residual_rows)
    ctx_models = v3.fit_ensemble(ctx_t[residual_idx], residual_rows)
    ham_models = v5.fit_ham_ensemble(ctx_t[residual_idx], residual_rows)

    ptext_val = v3.ensemble_predict(text_models, xv)
    pctx_val = v3.ensemble_predict(ctx_models, ctx_v)
    pham_val = v5.avg_prob(ham_models, ctx_v)

    ptext_test = v3.ensemble_predict(text_models, xe)
    pctx_test = v3.ensemble_predict(ctx_models, ctx_e)
    pham_test = v5.avg_prob(ham_models, ctx_e)

    stage1_gate, _ = v5.choose_gate(
        va,
        ptext_val,
        pctx_val,
        pham_val,
    )

    stage1_val = v5.predict_gate(
        va,
        ptext_val,
        pctx_val,
        pham_val,
        stage1_gate["textThreshold"],
        stage1_gate["contextThreshold"],
        stage1_gate["hamVetoThreshold"],
    )

    stage1_test = v5.predict_gate(
        te,
        ptext_test,
        pctx_test,
        pham_test,
        stage1_gate["textThreshold"],
        stage1_gate["contextThreshold"],
        stage1_gate["hamVetoThreshold"],
    )

    feat_val = combined_features(
        ptext_val,
        pctx_val,
        pham_val,
        prep_val,
        prep_dis_val,
    )
    feat_test = combined_features(
        ptext_test,
        pctx_test,
        pham_test,
        prep_test,
        prep_dis_test,
    )

    print("[8/9] select robust reputation rescue gates", flush=True)
    safe_gate = choose_gate(va, stage1_val, feat_val, safe=True)
    balanced_gate = choose_gate(va, stage1_val, feat_val, safe=False)

    safe_pred = predict(stage1_test, feat_test, safe_gate)
    balanced_pred = predict(stage1_test, feat_test, balanced_gate)

    stage1_stats = v5.stats(te, stage1_test)
    safe_stats = stats(te, safe_pred, stage1_test)
    balanced_stats = stats(te, balanced_pred, stage1_test)

    print("[9/9] write report", flush=True)

    result = {
        "version": "v16-local-reputation-campaign",
        "dataset": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "testSpam": base["spamTotal"],
            "testHam": base["hamTotal"],
            "residualTraining": len(residual_rows),
        },
        "architecture": {
            "stage1": "v5 precision text/context/ham gate",
            "reputationEntities": [
                "sender",
                "senderDomain",
                "returnDomain",
                "messageIdDomain",
                "ip24",
                "normalizedSubject",
                "simhashCampaignBucket",
                "urlDomain",
            ],
            "reputationTraining": "5-fold out-of-fold features",
            "reputationModels": [
                "weighted logistic regression",
                "weighted histogram gradient boosting",
            ],
            "cloudRequired": False,
            "rawMailUploaded": False,
            "safeValidation": "zero added FP independently on both validation halves",
            "hardHamAddedFpBudget": 0,
            "testLabelsUsedForTrainingOrThresholds": False,
        },
        "evaluationWarning": (
            "This public corpus is old and randomly split. Sender/domain/campaign "
            "overlap can make reputation features look stronger than they will "
            "on a truly future stream. A time-separated modern lockbox is required."
        ),
        "rspamdBayes": base,
        "stage1ValidationGate": stage1_gate,
        "stage1Test": stage1_stats,
        "safeValidationGate": safe_gate,
        "safeTest": safe_stats,
        "balancedValidationGate": balanced_gate,
        "balancedTest": balanced_stats,
        "review1pctAfterStage1": review(te, stage1_test, feat_test),
    }

    (REPORTS / "v16-benchmark.json").write_text(json.dumps(result, indent=2))

    md = [
        "# MailGuard v16 local reputation + campaign benchmark",
        "",
        f"Final test: {base['spamTotal']} spam + {base['hamTotal']} ham.",
        "",
        "| Mode | Spam recall | FP total | Extra spam vs stage 1 | Extra FP vs stage 1 |",
        "|---|---:|---:|---:|---:|",
        f"| Rspamd + Bayes | {base['recall']:.2%} | "
        f"{base['falsePositives']}/{base['hamTotal']} | - | - |",
        f"| Stage 1 (v5 precision) | {stage1_stats['recall']:.2%} | "
        f"{stage1_stats['falsePositives']}/{stage1_stats['hamTotal']} | 0 | 0 |",
        f"| v16 safe | {safe_stats['recall']:.2%} | "
        f"{safe_stats['falsePositives']}/{safe_stats['hamTotal']} | "
        f"{safe_stats['rescuedOverStage1']} | "
        f"{safe_stats['addedFalsePositivesOverStage1']} |",
        f"| v16 balanced | {balanced_stats['recall']:.2%} | "
        f"{balanced_stats['falsePositives']}/{balanced_stats['hamTotal']} | "
        f"{balanced_stats['rescuedOverStage1']} | "
        f"{balanced_stats['addedFalsePositivesOverStage1']} |",
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
        "Warning: this corpus is not a realistic chronological production stream. "
        "Reputation quality must be re-tested on a modern time-separated lockbox.",
    ]

    report = "\n".join(md) + "\n"
    (REPORTS / "v16-benchmark.md").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
