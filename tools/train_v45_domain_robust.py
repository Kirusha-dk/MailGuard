#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import html
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from email import policy
from email.parser import BytesParser
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from rspamd_benchmark_audit import load_frozen_v40

SEED = 20261007
REPORTS = Path("reports")
MODELS = Path("models/v45")
CACHE = Path(".cache/v45")
TARGET_FPR = 0.004
DEV_FPR = 0.0024

EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
NUM_RE = re.compile(r"\b\d{3,}\b")
WS_RE = re.compile(r"\s+")


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def normalize_text(value):
    s = unicodedata.normalize("NFKC", str(value or ""))
    s = html.unescape(s).replace("\x00", " ")
    s = EMAIL_RE.sub(" <EMAIL> ", s)
    s = URL_RE.sub(" <URL> ", s)
    s = NUM_RE.sub(" <NUM> ", s)
    return WS_RE.sub(" ", s).strip()


def canonical_from_bytes(raw):
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw)
    except Exception:
        return normalize_text(raw.decode("utf-8", "replace"))[:120000]

    subject = normalize_text(msg.get("subject") or "")
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_content_disposition() == "attachment":
                continue
            if part.get_content_type() not in ("text/plain", "text/html"):
                continue
            try:
                parts.append(str(part.get_content()))
            except Exception:
                payload = part.get_payload(decode=True) or b""
                parts.append(payload.decode("utf-8", "replace"))
    else:
        try:
            parts.append(str(msg.get_content()))
        except Exception:
            payload = msg.get_payload(decode=True) or raw
            parts.append(payload.decode("utf-8", "replace"))

    body = normalize_text("\n".join(parts))
    return ("SUBJECT " + subject + "\nBODY " + body)[:120000]


def content_identity(subject, body):
    s = normalize_text(subject).casefold() + "\n" + normalize_text(body).casefold()
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()


def add_public_v43(rows):
    from datasets import load_dataset

    seen = {hashlib.sha256(r["text"].encode("utf-8", "replace")).hexdigest() for r in rows}
    out = list(rows)

    def add(source, y, subject, body):
        text = "SUBJECT " + normalize_text(subject) + "\nBODY " + normalize_text(body)
        if len(text) < 40:
            return
        key = hashlib.sha256(text.casefold().encode("utf-8", "replace")).hexdigest()
        if key in seen:
            return
        seen.add(key)
        out.append({"text": text[:120000], "y": int(y), "source": source})

    joe = load_dataset("renemel/joephishing_labeled_phishing", split="train")
    for row in joe:
        subject = row.get("Subject") or ""
        body = row.get("Body") or ""
        if "FOLDER INTERNAL DATA" in str(subject).upper():
            continue
        if "not a real message" in str(body).lower():
            continue
        add("JoePhishing-2021", 1, subject, body)

    biz = load_dataset("wardacoder/business-email-dataset", split="train")
    for row in biz:
        body = row.get("output") or ""
        m = re.search(r"(?im)^\s*subject\s*:\s*(.+?)\s*$", str(body))
        subject = m.group(1) if m else "Business correspondence"
        add("BusinessSynthetic-2025", 0, subject, body)

    tur = load_dataset("anilguven/turkish_spam_email")
    for split in tur:
        for row in tur[split]:
            add("TurkishMail", int(row.get("labels") or 0), "", row.get("text") or "")

    return out


def load_engineering_rows():
    frozen, audit = load_frozen_v40(".cache/v40-fresh50k")
    rows = []
    for split in ("train", "val"):
        for r in frozen[split]:
            rows.append({
                "text": canonical_from_bytes(r["path"].read_bytes()),
                "y": int(r["y"]),
                "source": str(r["source"]),
            })
    rows = add_public_v43(rows)
    return rows, audit


def vectorizers():
    char = HashingVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), n_features=2**19,
        alternate_sign=False, norm="l2", lowercase=True,
    )
    word = HashingVectorizer(
        analyzer="word", ngram_range=(1, 2), n_features=2**18,
        alternate_sign=False, norm="l2", lowercase=True,
        token_pattern=r"(?u)\b\w[\w<>_-]+\b",
    )
    return char, word


def transform(texts, char, word):
    return hstack([char.transform(texts), word.transform(texts)], format="csr")


def source_class_weights(rows, ham_penalty):
    counts = Counter((r["source"], r["y"]) for r in rows)
    totals = defaultdict(float)
    for (src, y), n in counts.items():
        totals[y] += n
    w = np.empty(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        n = counts[(r["source"], r["y"])]
        class_groups = sum(1 for (s, y) in counts if y == r["y"])
        base = 1.0 / max(1, n * class_groups)
        w[i] = base * (ham_penalty if r["y"] == 0 else 1.0)
    w *= len(rows) / max(1e-12, w.sum())
    return w


def source_fold(source):
    table = {
        "TREC-05": 0, "JoePhishing-2021": 0,
        "CEAS-08": 1, "BusinessSynthetic-2025": 1,
        "TREC-07": 2, "TurkishMail": 2,
        "Enron": 3,
        "TREC-06": 4, "Assassin": 4, "Ling": 4,
    }
    return table.get(source, int(hashlib.sha256(source.encode()).hexdigest()[:8], 16) % 5)


def metrics_at(y, p, threshold, sources):
    pred = p >= threshold
    yb = y.astype(bool)
    tp = int((yb & pred).sum())
    fn = int((yb & ~pred).sum())
    fp = int((~yb & pred).sum())
    tn = int((~yb & ~pred).sum())
    by_source = {}
    for src in sorted(set(sources)):
        m = np.asarray([s == src for s in sources])
        sy = yb[m]
        sp = pred[m]
        spam = int(sy.sum())
        ham = int((~sy).sum())
        by_source[src] = {
            "n": int(m.sum()),
            "spam": spam,
            "ham": ham,
            "recall": (int((sy & sp).sum()) / spam) if spam else None,
            "fp": int(((~sy) & sp).sum()),
            "fpr": (int(((~sy) & sp).sum()) / ham) if ham else None,
        }
    return {
        "threshold": float(threshold), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": tp / max(1, tp + fn),
        "precision": tp / max(1, tp + fp),
        "fpr": fp / max(1, fp + tn),
        "bySource": by_source,
    }


def select_threshold(y, p, sources, max_fpr):
    yb = y.astype(bool)
    ham_p = p[~yb]
    max_fp = max(1, int(math.floor((~yb).sum() * max_fpr)))
    qs = np.linspace(0.90, 1.0, 700, endpoint=False)
    candidates = np.unique(np.concatenate([
        np.asarray([0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.000001]),
        np.quantile(ham_p, qs) if len(ham_p) else np.asarray([1.000001]),
    ]))
    best = None
    for t in candidates:
        m = metrics_at(y, p, float(t), sources)
        if m["fp"] > max_fp:
            continue
        spam_recalls = [
            v["recall"] for v in m["bySource"].values()
            if v["spam"] >= 20 and v["recall"] is not None
        ]
        worst = min(spam_recalls) if spam_recalls else 0.0
        key = (m["recall"], worst, -m["fp"], t)
        if best is None or key > best[0]:
            best = (key, m)
    return best[1] if best else metrics_at(y, p, 1.000001, sources)


def train_one(x, y, rows, ham_penalty, seed):
    w = source_class_weights(rows, ham_penalty)
    model = SGDClassifier(
        loss="log_loss", penalty="elasticnet", alpha=2.5e-6,
        l1_ratio=0.04, max_iter=45, tol=1e-4, random_state=seed,
        average=True,
    )
    model.fit(x, y, sample_weight=w)

    train_p = model.predict_proba(x)[:, 1]
    hard = np.ones(len(rows), dtype=np.float64)
    hard[(y == 0) & (train_p >= np.quantile(train_p[y == 0], 0.92))] *= 3.0
    hard[(y == 1) & (train_p <= np.quantile(train_p[y == 1], 0.18))] *= 1.8
    model2 = SGDClassifier(
        loss="log_loss", penalty="elasticnet", alpha=2.5e-6,
        l1_ratio=0.04, max_iter=55, tol=8e-5, random_state=seed + 1000,
        average=True,
    )
    model2.fit(x, y, sample_weight=w * hard)
    return model2


def run_oof(rows, x, ham_penalty):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    sources = [r["source"] for r in rows]
    folds = np.asarray([source_fold(s) for s in sources], dtype=np.int32)
    p = np.zeros(len(rows), dtype=np.float64)

    for fid in range(5):
        tr = folds != fid
        va = folds == fid
        train_rows = [rows[i] for i in np.where(tr)[0]]
        print("v45 fold", fid, "train", int(tr.sum()), "valid", int(va.sum()), flush=True)
        model = train_one(x[tr], y[tr], train_rows, ham_penalty, SEED + fid)
        p[va] = model.predict_proba(x[va])[:, 1]

    selected = select_threshold(y, p, sources, DEV_FPR)
    selected["ap"] = float(average_precision_score(y, p))
    selected["auc"] = float(roc_auc_score(y, p))
    return selected, p


def main():
    seed_all()
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    rows, audit = load_engineering_rows()
    texts = [r["text"] for r in rows]
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    sources = [r["source"] for r in rows]
    print("v45 engineering rows", len(rows), Counter((r["source"], r["y"]) for r in rows), flush=True)

    char, word = vectorizers()
    x = transform(texts, char, word)

    candidates = []
    for hp in (2.0, 3.5, 5.0):
        result, p = run_oof(rows, x, hp)
        result["hamPenalty"] = hp
        spam_recalls = [
            v["recall"] for v in result["bySource"].values()
            if v["spam"] >= 20 and v["recall"] is not None
        ]
        result["worstSpamSourceRecall"] = min(spam_recalls) if spam_recalls else 0.0
        candidates.append((result, p))
        print("v45 candidate", hp, result["recall"], result["fp"], result["fpr"],
              result["worstSpamSourceRecall"], flush=True)

    candidates.sort(key=lambda z: (
        z[0]["recall"], z[0]["worstSpamSourceRecall"], -z[0]["fp"]
    ), reverse=True)
    best, oof_p = candidates[0]
    hp = best["hamPenalty"]

    final_model = train_one(x, y, rows, hp, SEED + 99)
    artifact = {
        "version": "v45-domain-robust-hashed-text",
        "model": final_model,
        "threshold": float(best["threshold"]),
        "targetFpr": TARGET_FPR,
        "devFpr": DEV_FPR,
        "hamPenalty": hp,
        "featureConfig": {
            "char": {"ngram": [3, 5], "features": 2**19},
            "word": {"ngram": [1, 2], "features": 2**18},
            "canonical": "subject+body only; neutralize emails/urls/long numbers; no source headers",
        },
    }
    joblib.dump(artifact, MODELS / "v45.joblib", compress=3)

    report = {
        "version": "v45-domain-robust-hashed-text",
        "goal": {"recall": 0.90, "maxFpPer25kHam": 100, "fpr": TARGET_FPR},
        "selectionDevFpr": DEV_FPR,
        "engineeringData": {
            "count": len(rows),
            "v40Audit": audit,
            "sources": dict(Counter(r["source"] for r in rows)),
            "note": "v43 labels are engineering data now; OOF is source-held-out, not a pristine final lockbox.",
        },
        "bestOof": best,
        "allCandidates": [x[0] for x in candidates],
        "method": {
            "representation": "label-blind canonical subject/body + hashed char and word ngrams",
            "domainBalance": "equal total weight per source x class group",
            "hardMining": "retrain with top 8% suspicious ham x3 and hardest 18% spam x1.8",
            "validation": "5-fold source-family held-out OOF",
            "threshold": "maximize OOF recall under 0.24% ham FPR development budget",
        },
    }
    (REPORTS / "v45-domain-robust.json").write_text(json.dumps(report, indent=2))

    b = best
    lines = [
        "# MailGuard v45 domain-robust fast candidate", "",
        "Engineering OOF only; this is not the final unseen 50k lockbox.", "",
        "| Metric | Result |", "|---|---:|",
        f"| Recall | {b['recall']:.2%} |",
        f"| False negatives | {b['fn']} |",
        f"| False positives | {b['fp']} |",
        f"| FPR | {b['fpr']:.3%} |",
        f"| Precision | {b['precision']:.3%} |",
        f"| Worst spam-source recall | {b['worstSpamSourceRecall']:.2%} |",
        f"| AP | {b['ap']:.5f} |",
        f"| AUC | {b['auc']:.5f} |",
        f"| Selected ham penalty | {hp:.1f} |", "",
        "Target final gate: >=90% recall and <=100 FP per 25,000 ham (0.4% FPR).",
        "Development threshold is intentionally stricter at 0.24% OOF FPR.",
    ]
    (REPORTS / "v45-domain-robust.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
