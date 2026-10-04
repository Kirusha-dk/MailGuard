#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

import benchmark_v18_overnight as v18
import benchmark_v19_combined as v19
import benchmark_v25_soft_veto as v25
import benchmark_v28_crosssource as v28
import benchmark_v30_balanced_lowfp as v30
import benchmark_v31_sanitized_crosssource as v31
import benchmark_v33_overnight_neural as v33
import benchmark_v34_hybrid_rescue as v34
import benchmark_v36_bugfix_hardmine as v36

base = v25.base
v3 = base.v3

SEED = 20261020
CACHE = Path(".cache/v40-fresh50k")
EML_DIR = CACHE / "eml"
MANIFEST = CACHE / "manifest.json"
OUT_MODELS = Path("models/v40")
REPORT_JSON = Path("reports/v40-fresh50k.json")
REPORT_MD = Path("reports/v40-fresh50k.md")
ERROR_JSON = Path("reports/v40-error-analysis.json")

# V40 deliberately reuses part of the already-observed v39 lockbox as
# engineering/training data, then evaluates on 50,000 later, content-unique rows.
# The new test rows are disjoint from every v39 train/val/test row.
AUG_TRAIN = {
    ("TREC-05", 0): (0, 2500),
    ("TREC-05", 1): (0, 2500),
    ("CEAS-08", 0): (0, 2500),
    ("CEAS-08", 1): (0, 2500),
}
AUG_VAL = {
    ("TREC-05", 0): (2500, 3400),
    ("TREC-05", 1): (2500, 3400),
    ("CEAS-08", 0): (2500, 3400),
    ("CEAS-08", 1): (2500, 3400),
}

# Remaining unique rows after the entire v39 split:
#   TREC-07 11221/13231, Enron 5441/4370, TREC-06 4185/405,
#   Assassin 733/197, Ling 126/15, TREC-05 12987/5372,
#   CEAS-08 1768/5364 (ham/spam).
FRESH_TEST_PLAN = {
    ("TREC-07", 0): 8000, ("TREC-07", 1): 12000,
    ("TREC-05", 0): 8000, ("TREC-05", 1): 4500,
    ("Enron", 0): 4000, ("Enron", 1): 3500,
    ("TREC-06", 0): 3000, ("TREC-06", 1): 400,
    ("CEAS-08", 0): 1500, ("CEAS-08", 1): 4500,
    ("Assassin", 0): 400, ("Assassin", 1): 90,
    ("Ling", 0): 100, ("Ling", 1): 10,
}

EXPECTED_TRAIN = 50000
EXPECTED_VAL = 10000
EXPECTED_TEST = 50000

_METRIC_CALLS = []
_ORIGINAL_METRICS = v19.metrics


def _load_cache():
    if not MANIFEST.exists() or not EML_DIR.exists():
        return None
    try:
        obj = json.loads(MANIFEST.read_text())
    except Exception:
        return None
    if obj.get("version") != 40:
        return None

    out = {}
    for split in ("train", "val", "test"):
        rows = [
            {
                "path": EML_DIR / x["file"],
                "y": int(x["y"]),
                "source": x["source"],
            }
            for x in obj.get(split, [])
        ]
        if not rows or not all(r["path"].exists() for r in rows[:100]):
            return None
        out[split] = rows

    if (
        len(out["train"]) != EXPECTED_TRAIN
        or len(out["val"]) != EXPECTED_VAL
        or len(out["test"]) != EXPECTED_TEST
    ):
        return None
    return out


def _old_used_count(bucket_key):
    return (
        int(v28.TRAIN_PLAN.get(bucket_key, 0))
        + int(v28.VAL_PLAN.get(bucket_key, 0))
        + int(v28.TEST_PLAN.get(bucket_key, 0))
    )


def prepare_v40():
    cached = _load_cache()
    if cached is not None:
        print(
            "v40 cache",
            len(cached["train"]), len(cached["val"]), len(cached["test"]),
            flush=True,
        )
        return cached

    from datasets import load_dataset

    ds = load_dataset("JinqiangDing/seven-phishing-email-datasets")["train"]
    print("HF rows", len(ds), flush=True)

    counts = Counter()
    labels_by_key = defaultdict(set)
    for i, item in enumerate(ds):
        k = v36.content_key(item)
        counts[k] += 1
        labels_by_key[k].add(int(item.get("label") or 0))
        if (i + 1) % 25000 == 0:
            print("v40 content-dedupe pass", i + 1, "/", len(ds), flush=True)

    conflict_keys = {k for k, labels in labels_by_key.items() if len(labels) > 1}
    duplicate_keys = {k for k, n in counts.items() if n > 1}

    buckets = defaultdict(list)
    for i, item in enumerate(ds):
        k = v36.content_key(item)
        if k in conflict_keys or k in duplicate_keys:
            continue
        source = v36.md.clean(item.get("dataset_name"))
        y = int(item.get("label") or 0)
        if y in (0, 1):
            buckets[(source, y)].append((k, i))

    for arr in buckets.values():
        arr.sort(key=lambda x: x[0])

    for key in sorted(buckets):
        if key in set(v28.TRAIN_PLAN) | set(v28.VAL_PLAN) | set(v28.TEST_PLAN) | set(FRESH_TEST_PLAN):
            print("v40 bucket", key, len(buckets[key]), "old_used", _old_used_count(key), flush=True)

    # Reconstruct the exact corrected v39 training and validation slices first.
    cursor = defaultdict(int)
    picked = {"train": [], "val": [], "test": []}

    def take_old_plan(split, plan):
        for bucket_key, n in plan.items():
            arr = buckets[bucket_key]
            start = cursor[bucket_key]
            end = start + n
            if end > len(arr):
                raise RuntimeError(f"not enough rows for {bucket_key}: need {end}, have {len(arr)}")
            for _, idx in arr[start:end]:
                picked[split].append((bucket_key[0], bucket_key[1], idx))
            cursor[bucket_key] = end

    take_old_plan("train", v28.TRAIN_PLAN)
    take_old_plan("val", v28.VAL_PLAN)

    # Add 10k/3.6k already-observed TREC-05 + CEAS-08 examples.
    # These were part of v39's old test, so using them here is engineering iteration,
    # not leakage into the new v40 lockbox.
    for bucket_key, (start, end) in AUG_TRAIN.items():
        arr = buckets[bucket_key]
        for _, idx in arr[start:end]:
            picked["train"].append((bucket_key[0], bucket_key[1], idx))

    for bucket_key, (start, end) in AUG_VAL.items():
        arr = buckets[bucket_key]
        for _, idx in arr[start:end]:
            picked["val"].append((bucket_key[0], bucket_key[1], idx))

    # Fresh test starts strictly after every row range consumed by v39.
    for bucket_key, n in FRESH_TEST_PLAN.items():
        arr = buckets[bucket_key]
        start = _old_used_count(bucket_key)
        end = start + n
        if end > len(arr):
            raise RuntimeError(
                f"v40 not enough fresh rows for {bucket_key}: "
                f"need through {end}, have {len(arr)}"
            )
        for _, idx in arr[start:end]:
            picked["test"].append((bucket_key[0], bucket_key[1], idx))

    if len(picked["train"]) != EXPECTED_TRAIN:
        raise RuntimeError(f"bad v40 train size {len(picked['train'])}")
    if len(picked["val"]) != EXPECTED_VAL:
        raise RuntimeError(f"bad v40 val size {len(picked['val'])}")
    if len(picked["test"]) != EXPECTED_TEST:
        raise RuntimeError(f"bad v40 test size {len(picked['test'])}")

    EML_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 40,
        "contentKeyExcludesLabel": True,
        "freshTestStartsAfterV39UsedRanges": True,
        "oldV39TestUsedForEngineering": True,
        "train": [], "val": [], "test": [],
    }
    out = {"train": [], "val": [], "test": []}

    for split in ("train", "val", "test"):
        picked[split].sort(
            key=lambda x: hashlib.sha256(f"{x[0]}:{x[2]}".encode()).hexdigest()
        )
        for j, (source, y, idx) in enumerate(picked[split]):
            filename = f"{split}-{j:05d}.eml"
            path = EML_DIR / filename
            path.write_bytes(v31.sanitized_render(ds[idx], source, idx))
            out[split].append({"path": path, "y": y, "source": source})
            manifest[split].append({"file": filename, "y": y, "source": source})
            if (j + 1) % 5000 == 0:
                print("v40 render", split, j + 1, "/", len(picked[split]), flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest))

    for split in ("train", "val", "test"):
        print(
            "v40", split, len(out[split]),
            "spam", sum(r["y"] for r in out[split]),
            "ham", sum(not r["y"] for r in out[split]),
            "sources", dict(Counter(r["source"] for r in out[split])),
            flush=True,
        )
    return out


def v40_sample_weights(rows):
    y = np.asarray([r["y"] for r in rows], dtype=np.int32)
    w = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)

    for i, row in enumerate(rows):
        symbols = {name.upper() for name, _ in row["symbols"]}
        rscore = float(row["rscore"] or 0.0)

        if not row["y"]:
            # Stronger than v39: explicitly spend less FP budget.
            w[i] *= 7.0
            if rscore >= 1.5:
                w[i] *= 1.55
            if rscore >= 3.0:
                w[i] *= 1.25
            if "BAYES_HAM" in symbols:
                w[i] *= 1.32
            if "MIME_GOOD" in symbols:
                w[i] *= 1.12
        elif rscore <= 3.5:
            # Keep recall pressure so the lower FP target does not collapse recall.
            w[i] *= 2.45
            if rscore <= 1.0:
                w[i] *= 1.18

    return w


def fit_v40_ham_veto(x, rows, seed):
    y_ham = np.asarray([0 if r["y"] else 1 for r in rows], dtype=np.int32)
    w = compute_sample_weight(class_weight="balanced", y=y_ham).astype(np.float64)

    for i, row in enumerate(rows):
        symbols = {name.upper() for name, _ in row["symbols"]}
        rscore = float(row["rscore"] or 0.0)

        if y_ham[i]:
            w[i] *= 2.90
            if rscore >= 1.5:
                w[i] *= 1.55
            if rscore >= 3.0:
                w[i] *= 1.30
            if "BAYES_HAM" in symbols:
                w[i] *= 1.28
            if "MIME_GOOD" in symbols:
                w[i] *= 1.15
        elif rscore <= 3.5:
            w[i] *= 1.45

    model = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=260,
        max_leaf_nodes=13,
        min_samples_leaf=28,
        l2_regularization=2.8,
        random_state=seed,
    )
    model.fit(x, y_ham, sample_weight=w)
    return model


def v40_meta_score(models, x, raw_probs):
    linear, tree = models
    pl = np.clip(linear.predict_proba(x)[:, 1], 1e-8, 1.0)
    pt = np.clip(tree.predict_proba(x)[:, 1], 1e-8, 1.0)

    pair = np.sqrt(pl * pt)
    median = np.median(raw_probs, axis=0)
    q75 = np.quantile(raw_probs, 0.75, axis=0)
    top3 = np.mean(np.sort(raw_probs, axis=0)[-3:, :], axis=0)

    agree80 = np.sum(raw_probs >= 0.80, axis=0)
    agree72 = np.sum(raw_probs >= 0.72, axis=0).astype(np.int32)

    # Conservative core + rescue only when many independent branches agree.
    core = pair * (0.72 + 0.28 * median) * (
        0.84 + 0.16 * agree80 / raw_probs.shape[0]
    )
    rescue = 0.94 * np.sqrt(np.clip(q75 * top3, 1e-10, 1.0))
    rescue *= 0.88 + 0.12 * np.sqrt(pair)

    rescue_ok = (agree72 >= 5) & (q75 >= 0.76) & (top3 >= 0.84)
    score = np.where(rescue_ok, np.maximum(core, rescue), core)
    return np.clip(score, 0.0, 1.0), pl, pt, agree72


def v40_profile(name):
    # Validation has 5,300 ham. V39 target-95 allowed 16 FP on 3,500 ham and
    # produced 376 FP on 25k unseen ham. V40 roughly halves the allowed FP rate.
    if name == "ultra-safe":
        return {"fp": 0, "fold_fp": 0, "worst": 0.30}
    if name == "safe":
        return {"fp": 2, "fold_fp": 1, "worst": 0.55}
    if name == "target-93":
        return {"fp": 5, "fold_fp": 2, "worst": 0.72}
    if name == "target-95":
        return {"fp": 8, "fold_fp": 3, "worst": 0.78}
    return {"fp": 5, "fold_fp": 2, "worst": 0.72}


def select_v40_guard(
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
    cfg = v40_profile(name)
    y = np.asarray([bool(r["y"]) for r in rows], dtype=bool)
    fold_values = sorted(set(fold_id.tolist()))

    primary_scores = np.asarray([
        .35, .40, .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .93, .95, .97
    ])
    primary_ham = np.asarray([
        .010, .018, .028, .040, .055, .075, .10, .13, .16
    ])
    primary_agree = (2, 3, 4, 5, 6)

    override_scores = np.asarray([
        .60, .66, .72, .78, .84, .88, .92, .95, .97, .985, .995
    ])
    override_ham = np.asarray([
        .020, .035, .055, .08, .12, .18, .25
    ])
    override_agree = (5, 6, 7, 8)

    primary = []
    for agree in primary_agree:
        for hmax in primary_ham:
            eligible = (agreement >= agree) & (ham_risk <= hmax)
            for threshold in primary_scores:
                m = eligible & (spam_score >= threshold)
                if int(((~y) & m).sum()) <= cfg["fp"]:
                    primary.append((threshold, hmax, agree, m))
    primary.append((1.000001, 0.0, 99, np.zeros(len(rows), dtype=bool)))

    override = []
    for agree in override_agree:
        for hmax in override_ham:
            eligible = (agreement >= agree) & (ham_risk <= hmax)
            for threshold in override_scores:
                m = eligible & (spam_score >= threshold)
                if int(((~y) & m).sum()) <= cfg["fp"]:
                    override.append((threshold, hmax, agree, m))
    override.append((1.000001, 0.0, 99, np.zeros(len(rows), dtype=bool)))

    best_stable = None
    best_any = None

    for ps, ph, pa, pm in primary:
        for os, oh, oa, om in override:
            pred = pm | om
            fp_total = int(((~y) & pred).sum())
            if fp_total > cfg["fp"]:
                continue

            stats = []
            recalls = []
            stable = True
            for fid in fold_values:
                fm = fold_id == fid
                fy = y[fm]
                fpred = pred[fm]
                fp = int(((~fy) & fpred).sum())
                if fp > cfg["fold_fp"]:
                    stable = False
                    break
                spam = int(fy.sum())
                tp = int((fy & fpred).sum())
                rec = tp / max(1, spam)
                recalls.append(rec)
                stats.append({
                    "fold": int(fid), "spam": spam, "tp": tp, "fp": fp, "recall": rec
                })
            if not stable:
                continue

            tp_total = int((y & pred).sum())
            recall = tp_total / max(1, int(y.sum()))
            worst = min(recalls) if recalls else 0.0
            mean = float(np.mean(recalls)) if recalls else 0.0

            point = {
                "name": name,
                "primaryThreshold": float(ps),
                "primaryMaxHamRisk": float(ph),
                "primaryMinAgreement": int(pa),
                "overrideThreshold": float(os),
                "overrideMaxHamRisk": float(oh),
                "overrideMinAgreement": int(oa),
                "spamDetected": tp_total,
                "falsePositives": fp_total,
                "recall": recall,
                "worstFoldRecall": worst,
                "meanFoldRecall": mean,
                "foldStats": stats,
                "validationFpBudget": cfg["fp"],
                "validationFoldFpBudget": cfg["fold_fp"],
                "selectionObjective": "max recall under reduced FP budget",
            }
            key = (
                tp_total,
                round(worst, 8),
                round(mean, 8),
                -fp_total,
                pa + oa,
                ps + os,
            )
            if best_any is None or key > best_any[0]:
                best_any = (key, point)
            if worst >= cfg["worst"]:
                if best_stable is None or key > best_stable[0]:
                    best_stable = (key, point)

    if best_stable is not None:
        return best_stable[1]
    if best_any is not None:
        return best_any[1]

    return {
        "name": name,
        "primaryThreshold": 1.000001,
        "primaryMaxHamRisk": 0.0,
        "primaryMinAgreement": 99,
        "overrideThreshold": 1.000001,
        "overrideMaxHamRisk": 0.0,
        "overrideMinAgreement": 99,
        "spamDetected": 0,
        "falsePositives": 0,
        "recall": 0.0,
        "worstFoldRecall": 0.0,
        "meanFoldRecall": 0.0,
        "foldStats": [],
        "validationFpBudget": cfg["fp"],
        "validationFoldFpBudget": cfg["fold_fp"],
        "selectionObjective": "max recall under reduced FP budget",
    }


def source_group_folds(rows, n_folds=5):
    # Five domain groups. TREC-05 and CEAS-08 each get their own held-out fold
    # because they caused the large v38/v39 validation-to-test gap.
    group = {
        "TREC-05": 0,
        "CEAS-08": 1,
        "TREC-07": 2,
        "Enron": 3,
        "TREC-06": 4,
        "Assassin": 4,
        "Ling": 4,
    }
    fold_id = np.asarray([group.get(str(r.get("source", "")), 4) for r in rows], dtype=np.int32)

    for fid in range(5):
        m = fold_id == fid
        yy = np.asarray([int(bool(rows[i]["y"])) for i in np.where(m)[0]], dtype=np.int32)
        ham = int((yy == 0).sum())
        spam = int((yy == 1).sum())
        if ham == 0 or spam == 0:
            raise RuntimeError(f"v40 fold {fid} lacks class ham={ham} spam={spam}")
        print("v40 source-group fold", fid, "ham", ham, "spam", spam, flush=True)
    return fold_id


def capture_metrics(rows, pred):
    s = _ORIGINAL_METRICS(rows, pred)
    _METRIC_CALLS.append((rows, np.asarray(pred, dtype=bool).copy(), dict(s)))
    return s


def _score_bin(x):
    x = float(x or 0.0)
    if x <= 0.0:
        return "<=0"
    if x <= 1.5:
        return "0..1.5"
    if x <= 3.5:
        return "1.5..3.5"
    return ">3.5"


def write_error_analysis():
    # v24 base.main records: rspamd, stage1, v16, near, ultra, safe, target93, target95.
    if len(_METRIC_CALLS) < 8:
        return {}
    rows, pred, _ = _METRIC_CALLS[-1]
    y = np.asarray([bool(r["y"]) for r in rows], dtype=bool)

    by_source = {}
    for source in sorted({str(r.get("source", "unknown")) for r in rows}):
        m = np.asarray([str(r.get("source", "unknown")) == source for r in rows], dtype=bool)
        sy = y[m]
        sp = pred[m]
        spam = int(sy.sum())
        ham = int((~sy).sum())
        tp = int((sy & sp).sum())
        fp = int(((~sy) & sp).sum())
        by_source[source] = {
            "spam": spam,
            "ham": ham,
            "tp": tp,
            "fn": spam - tp,
            "fp": fp,
            "recall": tp / max(1, spam),
            "fpr": fp / max(1, ham),
            "precision": tp / max(1, tp + fp),
        }

    fp_idx = np.where((~y) & pred)[0]
    fn_idx = np.where(y & (~pred))[0]

    fp_bins = Counter(_score_bin(rows[i].get("rscore")) for i in fp_idx)
    fn_bins = Counter(_score_bin(rows[i].get("rscore")) for i in fn_idx)

    fp_symbols = Counter()
    fn_symbols = Counter()
    for i in fp_idx:
        fp_symbols.update(name for name, _ in rows[i].get("symbols", []))
    for i in fn_idx:
        fn_symbols.update(name for name, _ in rows[i].get("symbols", []))

    out = {
        "falsePositives": int(len(fp_idx)),
        "falseNegatives": int(len(fn_idx)),
        "bySource": by_source,
        "fpRspamdScoreBins": dict(fp_bins),
        "fnRspamdScoreBins": dict(fn_bins),
        "topFpSymbols": fp_symbols.most_common(20),
        "topFnSymbols": fn_symbols.most_common(20),
    }
    ERROR_JSON.write_text(json.dumps(out, indent=2))
    return out


def write_report():
    src_json = Path("reports/v24-crossfit.json")
    src_md = Path("reports/v24-crossfit.md")
    errors = write_error_analysis()

    if src_json.exists():
        obj = json.loads(src_json.read_text())
        obj["version"] = "v40-fresh50k-lowfp"
        obj["dataset"]["training"] = EXPECTED_TRAIN
        obj["dataset"]["validation"] = EXPECTED_VAL
        obj["dataset"]["lockedTest"] = EXPECTED_TEST
        obj["dataset"]["freshTestAfterAllV39Ranges"] = True
        obj["dataset"]["oldV39LockboxReusedForEngineering"] = True
        obj["dataset"]["testSources"] = sorted({k[0] for k in FRESH_TEST_PLAN})
        obj["method"]["dedupeIdentity"] = "normalized sender+receiver+subject+body SHA256; label excluded"
        obj["method"]["sourceArtifactSanitization"] = True
        obj["method"]["v39ErrorResponse"] = (
            "add observed TREC-05/CEAS-08 examples to train/validation; stronger ham "
            "weight/veto; reduce validation FP budgets; conservative consensus rescue"
        )
        obj["method"]["validationFpBudgets"] = {
            "ultraSafe": 0, "safe": 2, "target93": 5, "target95": 8
        }
        obj["method"]["testLabelsUsedForTrainingOrThresholds"] = False
        obj["freshTestErrorAnalysis"] = errors
        REPORT_JSON.write_text(json.dumps(obj, indent=2))

    if src_md.exists():
        txt = src_md.read_text()
        txt = txt.replace(
            "# MailGuard v24 hard-ham-veto benchmark",
            "# MailGuard v40 fresh 50k low-FP benchmark",
        )
        txt = txt.replace("v24 ", "v40 ")
        txt = txt.replace(
            "The 10k test labels are not used",
            "The fresh 50k test labels are not used",
        )
        txt += (
            "\nV40 uses the previously observed v39 TREC-05/CEAS-08 lockbox only as "
            "engineering/training data, then evaluates on 50,000 later content-unique "
            "rows that begin strictly after every v39-used bucket range. The FP response "
            "is stronger ham weighting/veto, a conservative consensus rescue, and lower "
            "validation FP budgets (target-95: 8 instead of v39's 16).\n"
        )
        if errors:
            txt += "\nFresh-test per-source target-95 diagnostics:\n\n"
            txt += "| Source | Recall | FN | FP | FP rate | Precision |\n"
            txt += "|---|---:|---:|---:|---:|---:|\n"
            for source, s in errors["bySource"].items():
                txt += (
                    f"| {source} | {s['recall']:.2%} | {s['fn']} | {s['fp']} | "
                    f"{s['fpr']:.3%} | {s['precision']:.3%} |\n"
                )
            txt += f"\nFP Rspamd-score bins: {errors['fpRspamdScoreBins']}\n"
            txt += f"FN Rspamd-score bins: {errors['fnRspamdScoreBins']}\n"
            txt += f"Top FP symbols: {errors['topFpSymbols'][:10]}\n"
            txt += f"Top FN symbols: {errors['topFnSymbols'][:10]}\n"
        REPORT_MD.write_text(txt)


def main():
    data = prepare_v40()

    v33.SOURCE_BY_PATH = {
        str(row["path"]): row.get("source", "unknown")
        for split in ("train", "val", "test")
        for row in data[split]
    }

    v18.enron.prepare_enron = lambda: data["train"] + data["val"] + data["test"]
    v18.split_enron = lambda _rows: (data["train"], data["val"], data["test"])
    v3.build_splits = lambda _groups: ([], [], [])

    if not hasattr(v18, "_v33_original_base_weights"):
        v18._v33_original_base_weights = v18.base_weights

    v18.base_weights = v33.mild_source_weights
    v18.DeepMailNet = v33.OvernightMailNet
    v18.train_model = v33.overnight_train_model
    v18.MODELS = OUT_MODELS
    v33.OUT_MODELS = OUT_MODELS

    base.fit_forensic_text_experts = v34.fit_hybrid_forensic
    base.sample_weights = v40_sample_weights
    base.fit_ham_veto = fit_v40_ham_veto
    base.meta_score = v40_meta_score
    base.select_dual_guard = select_v40_guard
    base.apply_dual_guard = v30.apply_balanced_guard
    base.chronological_folds = source_group_folds
    base.MODELS = OUT_MODELS
    base.SEED = SEED

    # Capture final fresh-test FP/FN diagnostics without using them for fitting.
    v19.metrics = capture_metrics

    base.main()
    write_report()


if __name__ == "__main__":
    main()
