#!/usr/bin/env python3
import csv
import io
import json
import math
import random
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

import benchmark_improved as b
import benchmark_v3 as v3
import benchmark_v5 as v5
import benchmark_v15_hard_mining as v15

REPORTS = Path("reports")
CACHE = Path(".cache/enron-spam-large")
ZIP_PATH = CACHE / "enron_spam_data.zip"
EML_DIR = CACHE / "eml"
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "MWiechmann/enron_spam_data/master/enron_spam_data.zip"
)
SEED = 20261001


def download():
    CACHE.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        print("Enron archive cache hit", flush=True)
        return

    print("download Enron-Spam dataset", flush=True)
    req = Request(
        DATA_URL,
        headers={"User-Agent": "MailGuard-large-benchmark/1.0"},
    )
    with urlopen(req, timeout=180) as response:
        ZIP_PATH.write_bytes(response.read())


def clean_header(value):
    return " ".join((value or "").replace("\r", " ").replace("\n", " ").split())


def prepare_enron():
    download()
    marker = CACHE / "manifest.json"

    if marker.exists() and EML_DIR.exists():
        manifest = json.loads(marker.read_text())
        rows = [
            {
                "path": EML_DIR / item["file"],
                "y": int(item["y"]),
                "source": "enron_spam" if item["y"] else "enron_ham",
            }
            for item in manifest
        ]
        if rows and all(row["path"].exists() for row in rows[:50]):
            print("Enron prepared cache", len(rows), flush=True)
            return rows

    EML_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        raw_csv = zf.read("enron_spam_data.csv").decode("utf-8", "replace")

    reader = csv.DictReader(io.StringIO(raw_csv))
    rows = []
    seen = set()

    for idx, item in enumerate(reader):
        label = (item.get("Spam/Ham") or "").strip().lower()
        if label not in {"spam", "ham"}:
            continue

        subject = clean_header(item.get("Subject") or "")
        message = item.get("Message") or ""
        date = clean_header(item.get("Date") or "")

        # Keep every unique Enron-Spam record. Duplicate protection is based on
        # normalized content, not the synthetic RFC822 envelope added below.
        key = (
            label,
            subject.strip().lower(),
            " ".join(message.lower().split()),
        )
        if key in seen:
            continue
        seen.add(key)

        y = 1 if label == "spam" else 0
        filename = f"{len(rows):05d}.eml"
        path = EML_DIR / filename

        raw = (
            f"Subject: {subject}\r\n"
            f"From: enron-benchmark-{len(rows)}@example.invalid\r\n"
            f"To: mailguard-benchmark@example.invalid\r\n"
            f"X-MailGuard-Dataset: enron-spam\r\n"
            f"X-MailGuard-Original-Date: {date}\r\n"
            "MIME-Version: 1.0\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            "Content-Transfer-Encoding: 8bit\r\n"
            "\r\n"
        ).encode("utf-8") + message.encode("utf-8", "replace")

        path.write_bytes(raw)
        rows.append({
            "path": path,
            "y": y,
            "source": "enron_spam" if y else "enron_ham",
        })

    marker.write_text(json.dumps([
        {"file": row["path"].name, "y": row["y"]}
        for row in rows
    ]))

    print(
        "Enron unique prepared",
        len(rows),
        "spam",
        sum(row["y"] for row in rows),
        "ham",
        sum(not row["y"] for row in rows),
        flush=True,
    )
    return rows


def report_stats(rows, pred, stage1):
    y = np.asarray([row["y"] for row in rows], dtype=bool)
    spam_total = int(y.sum())
    ham_total = int((~y).sum())

    tp = int((y & pred).sum())
    fp = int(((~y) & pred).sum())
    stage_tp = int((y & stage1).sum())
    stage_fp = int(((~y) & stage1).sum())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, spam_total)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "spamTotal": spam_total,
        "hamTotal": ham_total,
        "spamDetected": tp,
        "falseNegatives": spam_total - tp,
        "falsePositives": fp,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fpr": fp / max(1, ham_total),
        "rescuedOverStage1": tp - stage_tp,
        "addedFalsePositivesOverStage1": fp - stage_fp,
    }


def main():
    REPORTS.mkdir(exist_ok=True)
    b.wait_rspamd()

    # Current SpamAssassin corpus remains the only training/validation source.
    groups = b.prepare()
    train, val, _old_test = v3.build_splits(groups)

    # Large independent corpus is test-only.
    large_test = prepare_enron()

    print(
        "TRAIN", len(train),
        "VAL", len(val),
        "LARGE TEST", len(large_test),
        flush=True,
    )
    print(
        "Large-domain-shift evaluation: train/thresholds never use Enron labels",
        flush=True,
    )

    b.reset_bayes()
    b.learn(train)

    print("[1/8] scan train", flush=True)
    tr = b.scan_many(train, "large-v15-train", workers=12)
    print("[2/8] scan validation", flush=True)
    va = b.scan_many(val, "large-v15-val", workers=12)
    print("[3/8] scan Enron large test", flush=True)
    te = b.scan_many(large_test, "large-v15-enron", workers=16)

    base = b.base_metrics(te)

    xt, xv, xe, ctx_t, ctx_v, ctx_e = v3.matrices(tr, va, te)
    residual_idx = [i for i, row in enumerate(tr) if not row["rspam"]]
    residual_rows = [tr[i] for i in residual_idx]

    xr_text = xt[residual_idx]
    xr_ctx = ctx_t[residual_idx]

    print("[4/8] cross-fitted hard-example mining", flush=True)
    hardness, _ = v15.crossfit_hardness(xr_text, xr_ctx, residual_rows)

    base_text_models = v3.fit_ensemble(xr_text, residual_rows)
    base_ctx_models = v3.fit_ensemble(xr_ctx, residual_rows)
    ham_models = v5.fit_ham_ensemble(xr_ctx, residual_rows)

    spec_text_models = v15.fit_specialist_ensemble(
        xr_text, residual_rows, hardness, 2100
    )
    spec_ctx_models = v15.fit_specialist_ensemble(
        xr_ctx, residual_rows, hardness, 3100
    )

    print("[5/8] train hard-only experts", flush=True)
    error_text_models, error_ctx_models, mined_count = v15.fit_error_experts(
        xr_text,
        xr_ctx,
        residual_rows,
        hardness,
    )

    def all_probs(x_text, x_ctx):
        return (
            v3.ensemble_predict(base_text_models, x_text),
            v3.ensemble_predict(base_ctx_models, x_ctx),
            v15.avg_prob(spec_text_models, x_text),
            v15.avg_prob(spec_ctx_models, x_ctx),
            v15.avg_prob(error_text_models, x_text),
            v15.avg_prob(error_ctx_models, x_ctx),
            v5.avg_prob(ham_models, x_ctx),
        )

    print("[6/8] score validation and large test", flush=True)
    val_probs = all_probs(xv, ctx_v)
    test_probs = all_probs(xe, ctx_e)

    pbase_text_val, pbase_ctx_val, _, _, _, _, pham_val = val_probs
    pbase_text_test, pbase_ctx_test, _, _, _, _, pham_test = test_probs

    stage1_gate, _ = v5.choose_gate(
        va,
        pbase_text_val,
        pbase_ctx_val,
        pham_val,
    )

    stage1_val = v5.predict_gate(
        va,
        pbase_text_val,
        pbase_ctx_val,
        pham_val,
        stage1_gate["textThreshold"],
        stage1_gate["contextThreshold"],
        stage1_gate["hamVetoThreshold"],
    )
    stage1_test = v5.predict_gate(
        te,
        pbase_text_test,
        pbase_ctx_test,
        pham_test,
        stage1_gate["textThreshold"],
        stage1_gate["contextThreshold"],
        stage1_gate["hamVetoThreshold"],
    )

    feat_val = v15.features(*val_probs)
    feat_test = v15.features(*test_probs)

    print("[7/8] select v15 safe gate on validation only", flush=True)
    safe_gate = v15.choose_gate(va, stage1_val, feat_val, safe=True)
    balanced_gate = v15.choose_gate(va, stage1_val, feat_val, safe=False)

    safe_pred = v15.predict(stage1_test, feat_test, safe_gate)
    balanced_pred = v15.predict(stage1_test, feat_test, balanced_gate)

    stage1_stats = report_stats(te, stage1_test, stage1_test)
    safe_stats = report_stats(te, safe_pred, stage1_test)
    balanced_stats = report_stats(te, balanced_pred, stage1_test)

    print("[8/8] write report", flush=True)

    result = {
        "version": "v15-large-enron-domain-shift",
        "dataset": {
            "trainingCorpus": "SpamAssassin public corpus",
            "largeTestCorpus": "Enron-Spam",
            "largeTestTotal": len(te),
            "largeTestSpam": base["spamTotal"],
            "largeTestHam": base["hamTotal"],
            "residualTraining": len(residual_rows),
            "hardSubsetTraining": mined_count,
        },
        "methodology": {
            "largeTestUsedForTraining": False,
            "largeTestUsedForThresholdSelection": False,
            "largeTestUniqueRecords": True,
            "syntheticEnvelope": True,
            "note": (
                "Enron rows contain subject/body labels rather than complete modern "
                "SMTP metadata. RFC822 envelope headers are synthesized so Rspamd "
                "can scan the messages. This is a domain-shift quality/stress test, "
                "not a production-quality estimate."
            ),
        },
        "rspamdBayes": base,
        "stage1Gate": stage1_gate,
        "stage1LargeTest": stage1_stats,
        "safeValidationGate": safe_gate,
        "safeLargeTest": safe_stats,
        "balancedValidationGate": balanced_gate,
        "balancedLargeTest": balanced_stats,
    }

    (REPORTS / "v15-large-enron.json").write_text(json.dumps(result, indent=2))

    md = [
        "# MailGuard v15 large independent Enron-Spam benchmark",
        "",
        f"Large test: {len(te)} unique emails "
        f"({base['spamTotal']} spam + {base['hamTotal']} ham).",
        "",
        "Enron labels are test-only: they are not used for training or threshold selection.",
        "",
        "| Mode | Spam detected | Recall | FN | FP | FP rate | Precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Rspamd + Bayes | {base['spamDetected']}/{base['spamTotal']} | "
        f"{base['recall']:.2%} | {base['spamTotal'] - base['spamDetected']} | "
        f"{base['falsePositives']} | {base['fpr']:.3%} | - |",
        f"| Stage 1 | {stage1_stats['spamDetected']}/{stage1_stats['spamTotal']} | "
        f"{stage1_stats['recall']:.2%} | {stage1_stats['falseNegatives']} | "
        f"{stage1_stats['falsePositives']} | {stage1_stats['fpr']:.3%} | "
        f"{stage1_stats['precision']:.3%} |",
        f"| v15 safe | {safe_stats['spamDetected']}/{safe_stats['spamTotal']} | "
        f"{safe_stats['recall']:.2%} | {safe_stats['falseNegatives']} | "
        f"{safe_stats['falsePositives']} | {safe_stats['fpr']:.3%} | "
        f"{safe_stats['precision']:.3%} |",
        f"| v15 balanced | "
        f"{balanced_stats['spamDetected']}/{balanced_stats['spamTotal']} | "
        f"{balanced_stats['recall']:.2%} | {balanced_stats['falseNegatives']} | "
        f"{balanced_stats['falsePositives']} | {balanced_stats['fpr']:.3%} | "
        f"{balanced_stats['precision']:.3%} |",
        "",
        "Important: Enron-Spam is an old external corpus and lacks complete modern "
        "mail-routing metadata. This run is useful for scale and domain-shift evidence, "
        "not as a direct comparison with the customer's production report.",
    ]

    report = "\n".join(md) + "\n"
    (REPORTS / "v15-large-enron.md").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
