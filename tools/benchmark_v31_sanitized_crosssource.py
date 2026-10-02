#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import benchmark_v18_overnight as v18
import benchmark_v25_soft_veto as v25
import benchmark_v27_multidomain as md
import benchmark_v28_crosssource as v28
import benchmark_v30_balanced_lowfp as v30

base = v25.base
v3 = base.v3

SEED = v30.SEED
REPORT_JSON = Path("reports/v31-sanitized-crosssource-50k.json")
REPORT_MD = Path("reports/v31-sanitized-crosssource-50k.md")


def clean(value):
    if value is None:
        return ""
    return " ".join(re.sub(r"[\r\n\x00]+", " ", str(value)).split())


def sanitized_render(item, source, idx):
    """
    Render only mail-like fields.

    Previous cross-source benchmarks embedded the dataset name in an X- header
    and, when a sender was missing, in the fallback sender domain
    (unknown@trec-07.invalid, unknown@ceas-08.invalid, ...). Those are benchmark
    artifacts, not real spam signals, and they can teach reputation/context
    branches to recognize corpora instead of mail.

    Source identity is still retained in row metadata for leave-one-source-out
    validation, but it is never exposed to the classifier through the raw EML.
    """
    sender = clean(item.get("sender")) or "unknown@example.invalid"
    receiver = clean(item.get("receiver")) or "recipient@example.invalid"
    subject = clean(item.get("subject"))
    date = clean(item.get("date"))
    body = str(item.get("text") or "")

    headers = [
        f"Subject: {subject}",
        f"From: {sender}",
        f"To: {receiver}",
    ]
    if date:
        headers.append(f"Date: {date}")
    headers.extend([
        "MIME-Version: 1.0",
        "Content-Type: text/plain; charset=utf-8",
        "Content-Transfer-Encoding: 8bit",
        "",
        "",
    ])
    return ("\r\n".join(headers)).encode("utf-8", "replace") + body.encode(
        "utf-8", "replace"
    )


def write_report():
    src_json = Path("reports/v24-crossfit.json")
    src_md = Path("reports/v24-crossfit.md")

    if src_json.exists():
        obj = json.loads(src_json.read_text())
        obj["version"] = "v31-sanitized-crosssource-lowfp"
        obj["dataset"]["training"] = v28.EXPECTED_TRAIN
        obj["dataset"]["validation"] = v28.EXPECTED_VAL
        obj["dataset"]["lockedTest"] = v28.EXPECTED_TEST
        obj["dataset"]["trainingSources"] = [
            "TREC-07", "Enron", "TREC-06", "Assassin", "Ling"
        ]
        obj["dataset"]["testSources"] = ["TREC-05", "CEAS-08"]
        obj["method"]["sourceArtifactSanitization"] = (
            "dataset-name X headers removed; missing sender/receiver use one "
            "source-neutral fallback; source labels remain only outside raw EML"
        )
        obj["method"]["metaValidation"] = (
            "5-fold leave-one-source-family-out validation"
        )
        obj["method"]["gate"] = (
            "v30 two-tier low-FP gate with hard total/per-source FP budgets"
        )
        obj["method"]["testLabelsUsedForTrainingOrThresholds"] = False
        obj["method"]["individualTestErrorsInspected"] = False
        obj["warning"] = (
            "The v28/v30 test source pair is reused for engineering comparison and "
            "is not a pristine final lockbox. A new untouched corpus is required for "
            "the final generalization claim."
        )
        REPORT_JSON.write_text(json.dumps(obj, indent=2))

    if src_md.exists():
        txt = src_md.read_text()
        txt = txt.replace(
            "# MailGuard v24 hard-ham-veto benchmark",
            "# MailGuard v31 sanitized cross-source low-FP 50k benchmark",
        )
        txt = txt.replace("v24 ", "v31 ")
        txt += (
            "\nV31 removes benchmark-source leakage from the raw EML: no dataset-name "
            "X headers and no source-specific fallback sender domains. Source identity "
            "is retained only for leave-one-source-out validation. The v30 low-FP gate "
            "is kept so this run isolates the metadata-sanitization change.\n"
        )
        REPORT_MD.write_text(txt)


def main():
    # Critical fix: never expose corpus identity to the classifier.
    md.render = sanitized_render

    # Force a clean re-render; old v28 cache contains source-identifying raw EML.
    md.CACHE = Path(".cache/v31-sanitized-crosssource")
    md.EML_DIR = md.CACHE / "eml"
    md.MANIFEST = md.CACHE / "manifest.json"
    md.TRAIN_PLAN = v28.TRAIN_PLAN
    md.VAL_PLAN = v28.VAL_PLAN
    md.TEST_PLAN = v28.TEST_PLAN
    md.EXPECTED_TRAIN = v28.EXPECTED_TRAIN
    md.EXPECTED_VAL = v28.EXPECTED_VAL
    md.EXPECTED_TEST = v28.EXPECTED_TEST

    data = md.prepare()

    v18.enron.prepare_enron = lambda: data["train"] + data["val"] + data["test"]
    v18.split_enron = lambda _rows: (data["train"], data["val"], data["test"])
    v3.build_splits = lambda _groups: ([], [], [])

    base.select_dual_guard = v30.select_balanced_guard
    base.apply_dual_guard = v30.apply_balanced_guard
    base.chronological_folds = v28.source_heldout_folds
    base.MODELS = Path("models/v31")
    base.SEED = SEED

    base.main()
    write_report()


if __name__ == "__main__":
    main()
