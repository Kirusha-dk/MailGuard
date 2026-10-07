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
from urllib.parse import urlsplit

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from rspamd_benchmark_audit import load_frozen_v40

SEED = 20261007
REPORTS = Path("reports")
MODELS = Path("models/v46")
CACHE = Path(".cache/v46")
DEV_FPR = 0.0024
TARGET_FPR = 0.004
MAX_PER_SOURCE_CLASS = 1400
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>'\"]+")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})\b")
TAG_RE = re.compile(r"(?s)<[^>]+>")
FORM_RE = re.compile(r"(?is)<\s*form\b")
WS_RE = re.compile(r"\s+")


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def norm(value):
    s = unicodedata.normalize("NFKC", str(value or ""))
    s = html.unescape(s).replace("\x00", " ")
    return WS_RE.sub(" ", s).strip()


def host_from_url(value):
    v = value.strip().rstrip(".,);]>")
    if v.lower().startswith("www."):
        v = "http://" + v
    try:
        host = (urlsplit(v).hostname or "").lower().strip(".")
    except Exception:
        host = ""
    return host


def canonical_fields(subject, body, sender=""):
    subject = norm(subject)
    body_raw = str(body or "")
    body = norm(TAG_RE.sub(" ", body_raw))

    urls = URL_RE.findall(body_raw)
    hosts = [host_from_url(x) for x in urls]
    hosts = [h for h in hosts if h]
    email_domains = [x.lower() for x in EMAIL_RE.findall(subject + " " + body_raw)]
    sender_match = EMAIL_RE.search(sender or "")
    sender_domain = sender_match.group(1).lower() if sender_match else ""

    host_tokens = " ".join(sorted(set(hosts))[:24])
    email_tokens = " ".join(sorted(set(email_domains))[:12])

    # Two deterministic semantic windows: subject/head and tail. Preserve domains.
    head = body[:2400]
    tail = body[-1600:] if len(body) > 1600 else body
    common = (
        f"SUBJECT: {subject}\n"
        f"URL_HOSTS: {host_tokens}\n"
        f"EMAIL_DOMAINS: {email_tokens}\n"
        f"SENDER_DOMAIN: {sender_domain}\n"
    )
    semantic_head = (common + "BODY: " + head)[:5000]
    semantic_tail = (common + "BODY_TAIL: " + tail)[:4200]
    ngram_text = (common + "BODY: " + body)[:120000]

    structural = np.asarray([
        math.log1p(len(body)),
        math.log1p(len(subject)),
        min(20, len(urls)) / 20.0,
        min(20, len(set(hosts))) / 20.0,
        min(20, len(email_domains)) / 20.0,
        float(bool(FORM_RE.search(body_raw))),
        min(1.0, len(TAG_RE.findall(body_raw)) / 80.0),
        float("xn--" in body_raw.lower()),
        float(any(h.count(".") >= 3 for h in hosts)),
        float(bool(sender_domain and hosts and sender_domain not in set(hosts))),
    ], dtype=np.float32)

    ident = hashlib.sha256(
        (subject.casefold() + "\n" + body.casefold()).encode("utf-8", "replace")
    ).hexdigest()

    return {
        "semantic_head": semantic_head,
        "semantic_tail": semantic_tail,
        "ngram_text": ngram_text,
        "struct": structural,
        "identity": ident,
    }


def from_eml(raw):
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw)
    except Exception:
        return canonical_fields("", raw.decode("utf-8", "replace"))

    subject = str(msg.get("subject") or "")
    sender = str(msg.get("from") or "")
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
    return canonical_fields(subject, "\n".join(parts), sender)


def add_row(out, seen, source, y, subject, body, sender=""):
    c = canonical_fields(subject, body, sender)
    if len(c["ngram_text"]) < 40 or c["identity"] in seen:
        return
    seen.add(c["identity"])
    c.update({"source": str(source), "y": int(y)})
    out.append(c)


def load_rows():
    from datasets import load_dataset

    frozen, audit = load_frozen_v40(".cache/v40-fresh50k")
    rows, seen = [], set()
    for split in ("train", "val"):
        for r in frozen[split]:
            c = from_eml(r["path"].read_bytes())
            if c["identity"] in seen:
                continue
            seen.add(c["identity"])
            c.update({"source": str(r["source"]), "y": int(r["y"])})
            rows.append(c)

    joe = load_dataset("renemel/joephishing_labeled_phishing", split="train")
    for r in joe:
        subject, body = r.get("Subject") or "", r.get("Body") or ""
        if "FOLDER INTERNAL DATA" in str(subject).upper():
            continue
        if "not a real message" in str(body).lower():
            continue
        add_row(rows, seen, "JoePhishing-2021", 1, subject, body)

    biz = load_dataset("wardacoder/business-email-dataset", split="train")
    for r in biz:
        body = r.get("output") or ""
        m = re.search(r"(?im)^\s*subject\s*:\s*(.+?)\s*$", str(body))
        add_row(rows, seen, "BusinessSynthetic-2025", 0,
                m.group(1) if m else "Business correspondence", body)

    tur = load_dataset("anilguven/turkish_spam_email")
    for split in tur:
        for r in tur[split]:
            add_row(rows, seen, "TurkishMail", int(r.get("labels") or 0),
                    "", r.get("text") or "")

    # Deterministic source x class cap for a fast semantic smoke run.
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["source"], r["y"])].append(r)
    capped = []
    for key, arr in sorted(buckets.items()):
        arr.sort(key=lambda r: r["identity"])
        capped.extend(arr[:MAX_PER_SOURCE_CLASS])

    print("v46 rows", len(capped), Counter((r["source"], r["y"]) for r in capped), flush=True)
    return capped, audit


def source_fold(source):
    table = {
        "TREC-05": 0, "JoePhishing-2021": 0,
        "CEAS-08": 1, "BusinessSynthetic-2025": 1,
        "TREC-07": 2, "TurkishMail": 2,
        "Enron": 3,
        "TREC-06": 4, "Assassin": 4, "Ling": 4,
    }
    return table.get(source, int(hashlib.sha256(source.encode()).hexdigest()[:8], 16) % 5)


def source_class_weights(rows, ham_penalty=3.0):
    counts = Counter((r["source"], r["y"]) for r in rows)
    w = np.empty(len(rows), dtype=np.float64)
    groups_per_class = Counter(y for _, y in counts)
    for i, r in enumerate(rows):
        base = 1.0 / max(1, counts[(r["source"], r["y"])] * groups_per_class[r["y"]])
        w[i] = base * (ham_penalty if r["y"] == 0 else 1.0)
    w *= len(rows) / max(1e-12, w.sum())
    return w


def metrics_at(y, p, threshold, sources):
    pred = p >= threshold
    yb = y.astype(bool)
    tp, fn = int((yb & pred).sum()), int((yb & ~pred).sum())
    fp, tn = int((~yb & pred).sum()), int((~yb & ~pred).sum())
    by = {}
    for src in sorted(set(sources)):
        m = np.asarray([s == src for s in sources], dtype=bool)
        sy, sp = yb[m], pred[m]
        spam, ham = int(sy.sum()), int((~sy).sum())
        sfp = int(((~sy) & sp).sum())
        by[src] = {
            "n": int(m.sum()), "spam": spam, "ham": ham,
            "recall": (int((sy & sp).sum()) / spam) if spam else None,
            "fp": sfp, "fpr": sfp / ham if ham else None,
        }
    recs = [v["recall"] for v in by.values() if v["spam"] >= 100 and v["recall"] is not None]
    return {
        "threshold": float(threshold), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": tp / max(1, tp + fn),
        "precision": tp / max(1, tp + fp),
        "fpr": fp / max(1, fp + tn),
        "worstSpamSourceRecall": min(recs) if recs else 0.0,
        "bySource": by,
    }


def select_threshold(y, p, sources, max_fpr=DEV_FPR):
    yb = y.astype(bool)
    max_fp = max(1, int(math.floor((~yb).sum() * max_fpr)))
    ham_p = p[~yb]
    candidates = np.unique(np.concatenate([
        np.linspace(0.5, 0.999, 180),
        np.quantile(ham_p, np.linspace(0.88, 0.9999, 500)) if len(ham_p) else np.asarray([1.0]),
        np.asarray([1.000001]),
    ]))
    best = None
    for t in candidates:
        m = metrics_at(y, p, float(t), sources)
        if m["fp"] > max_fp:
            continue
        # Worst-source first prevents another Joe/Turkish collapse.
        key = (m["worstSpamSourceRecall"], m["recall"], -m["fp"], float(t))
        if best is None or key > best[0]:
            best = (key, m)
    return best[1]


def build_ngram(texts):
    char = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                             n_features=2**18, alternate_sign=False, norm="l2")
    word = HashingVectorizer(analyzer="word", ngram_range=(1, 2),
                             n_features=2**17, alternate_sign=False, norm="l2")
    return hstack([char.transform(texts), word.transform(texts)], format="csr")


def build_semantic(rows):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = 192
    heads = [r["semantic_head"] for r in rows]
    tails = [r["semantic_tail"] for r in rows]
    print("v46 encoding head windows", len(heads), flush=True)
    eh = model.encode(heads, batch_size=96, show_progress_bar=True,
                      normalize_embeddings=True, convert_to_numpy=True)
    print("v46 encoding tail windows", len(tails), flush=True)
    et = model.encode(tails, batch_size=96, show_progress_bar=True,
                      normalize_embeddings=True, convert_to_numpy=True)
    struct = np.vstack([r["struct"] for r in rows]).astype(np.float32)
    scaler = StandardScaler()
    struct = scaler.fit_transform(struct).astype(np.float32)
    x = np.hstack([eh, et, np.abs(eh - et), struct]).astype(np.float32)
    return x, scaler


def fit_ngram(x, y, rows, seed):
    w = source_class_weights(rows, 3.0)
    m = SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=3e-6,
                      l1_ratio=0.03, max_iter=50, tol=1e-4,
                      random_state=seed, average=True)
    m.fit(x, y, sample_weight=w)
    return m


def fit_semantic(x, y, rows, seed):
    w = source_class_weights(rows, 2.4)
    m = LogisticRegression(C=0.22, penalty="l2", max_iter=1600,
                           solver="lbfgs", random_state=seed)
    m.fit(x, y, sample_weight=w)
    return m


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def main():
    seed_all()
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    rows, audit = load_rows()
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    sources = [r["source"] for r in rows]
    folds = np.asarray([source_fold(s) for s in sources], dtype=np.int32)

    x_ng = build_ngram([r["ngram_text"] for r in rows])
    x_sem, struct_scaler = build_semantic(rows)

    p_ng = np.zeros(len(rows), dtype=np.float64)
    p_sem = np.zeros(len(rows), dtype=np.float64)

    for fid in range(5):
        tr, va = folds != fid, folds == fid
        tr_rows = [rows[i] for i in np.where(tr)[0]]
        print("v46 fold", fid, "train", int(tr.sum()), "valid", int(va.sum()), flush=True)
        ng = fit_ngram(x_ng[tr], y[tr], tr_rows, SEED + fid)
        sem = fit_semantic(x_sem[tr], y[tr], tr_rows, SEED + 100 + fid)
        p_ng[va] = ng.predict_proba(x_ng[va])[:, 1]
        p_sem[va] = sem.predict_proba(x_sem[va])[:, 1]

    candidates = []
    for name, p in [("ngram-preserve-domains", p_ng), ("semantic-multilingual", p_sem)]:
        m = select_threshold(y, p, sources)
        m.update({
            "name": name,
            "ap": float(average_precision_score(y, p)),
            "auc": float(roc_auc_score(y, p)),
        })
        candidates.append((m, p))
        print("v46", name, m["recall"], m["fp"], m["fpr"], m["worstSpamSourceRecall"], flush=True)

    # Blend logits; every component is OOF for each row.
    for alpha in np.linspace(0.0, 1.0, 21):
        p = sigmoid(alpha * logit(p_sem) + (1.0 - alpha) * logit(p_ng))
        m = select_threshold(y, p, sources)
        m.update({
            "name": f"blend-semantic-{alpha:.2f}",
            "semanticWeight": float(alpha),
            "ap": float(average_precision_score(y, p)),
            "auc": float(roc_auc_score(y, p)),
        })
        candidates.append((m, p))

    candidates.sort(key=lambda z: (
        z[0]["worstSpamSourceRecall"], z[0]["recall"], -z[0]["fp"]
    ), reverse=True)
    best, best_p = candidates[0]

    # Final experts are fit on all engineering rows. OOF scores define replay weights.
    hard = np.ones(len(rows), dtype=np.float64)
    ham_cut = np.quantile(best_p[y == 0], 0.97)
    spam_cut = np.quantile(best_p[y == 1], 0.20)
    hard[(y == 0) & (best_p >= ham_cut)] *= 3.0
    hard[(y == 1) & (best_p <= spam_cut)] *= 1.8

    w_ng = source_class_weights(rows, 3.0) * hard
    final_ng = SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=3e-6,
                             l1_ratio=0.03, max_iter=60, tol=8e-5,
                             random_state=SEED + 999, average=True)
    final_ng.fit(x_ng, y, sample_weight=w_ng)

    w_sem = source_class_weights(rows, 2.4) * hard
    final_sem = LogisticRegression(C=0.22, penalty="l2", max_iter=1800,
                                   solver="lbfgs", random_state=SEED + 1999)
    final_sem.fit(x_sem, y, sample_weight=w_sem)

    artifact = {
        "version": "v46-semantic-hybrid-smoke",
        "sentenceModel": MODEL_NAME,
        "maxSeqLength": 192,
        "ngramModel": final_ng,
        "semanticModel": final_sem,
        "structScaler": struct_scaler,
        "threshold": float(best["threshold"]),
        "semanticWeight": float(best.get("semanticWeight", 1.0 if best["name"] == "semantic-multilingual" else 0.0)),
        "devFpr": DEV_FPR,
        "targetFpr": TARGET_FPR,
        "note": "Fast source-held-out engineering candidate; not final lockbox evidence.",
    }
    joblib.dump(artifact, MODELS / "v46-hybrid.joblib", compress=3)

    report = {
        "version": "v46-semantic-hybrid-smoke",
        "goal": {"recall": 0.90, "maxFpPer25kHam": 100, "fpr": TARGET_FPR},
        "engineering": {
            "rows": len(rows), "maxPerSourceClass": MAX_PER_SOURCE_CLASS,
            "v40Audit": audit,
            "counts": {f"{s}|{y0}": n for (s, y0), n in sorted(Counter((r["source"], r["y"]) for r in rows).items())},
            "warning": "Joe/Business/Turkish labels are engineering data; this is not a new unseen lockbox.",
        },
        "method": {
            "canonical": "source-blind visible text preserving URL hosts/email domains",
            "semantic": MODEL_NAME + " head+tail embeddings",
            "structural": "URL/email/domain/html/form/punycode features",
            "validation": "5 source-family-held-out folds",
            "selector": "worst-source recall first under 0.24% OOF FPR",
            "hardReplay": "OOF top 3% ham and bottom 20% spam for final fit only",
        },
        "bestOof": best,
        "topCandidates": [m for m, _ in candidates[:8]],
    }
    (REPORTS / "v46-semantic-hybrid.json").write_text(json.dumps(report, indent=2))

    b = best
    lines = [
        "# MailGuard v46 semantic hybrid smoke", "",
        "Source-held-out engineering OOF only; not a pristine 50k lockbox.", "",
        "| Metric | Result |", "|---|---:|",
        f"| Best variant | {b['name']} |",
        f"| Recall | {b['recall']:.2%} |",
        f"| False negatives | {b['fn']} |",
        f"| False positives | {b['fp']} |",
        f"| FPR | {b['fpr']:.3%} |",
        f"| Precision | {b['precision']:.3%} |",
        f"| Worst spam-source recall | {b['worstSpamSourceRecall']:.2%} |",
        f"| AP | {b['ap']:.5f} |",
        f"| AUC | {b['auc']:.5f} |", "",
        "Final target remains >=90% recall with <=100 FP / 25k ham.",
    ]
    (REPORTS / "v46-semantic-hybrid.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
