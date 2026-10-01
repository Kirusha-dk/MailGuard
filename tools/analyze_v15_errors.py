#!/usr/bin/env python3
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import benchmark_improved as b
import benchmark_v3 as v3
import benchmark_v5 as v5
import benchmark_v15_hard_mining as v15

REPORTS = Path("reports")
URL_RE = re.compile(r"https?://", re.I)


def subject_of(raw):
    try:
        text = raw.decode("utf-8", "ignore")
    except Exception:
        return ""
    for line in text.splitlines()[:80]:
        if line.lower().startswith("subject:"):
            return line.split(":", 1)[1].strip()[:160]
    return ""


def message_shape(row):
    doc = b.extract_document(row["raw"])
    lower = doc.lower()
    return {
        "bytes": len(row["raw"]),
        "chars": len(doc),
        "urlCount": len(URL_RE.findall(doc)),
        "hasHtml": int("<html" in lower or "text/html" in lower),
        "rspamdScore": float(row["rscore"]),
        "required": float(row["required"]),
        "action": row["action"],
        "subject": subject_of(row["raw"]),
        "source": row["source"],
    }


def top_symbols(rows, mask, limit=20):
    counts = Counter()
    signed = Counter()
    for row, use in zip(rows, mask):
        if not use:
            continue
        for name, score in row["symbols"]:
            counts[name] += 1
            if score > 0:
                signed[name + " (+)"] += 1
            elif score < 0:
                signed[name + " (-)"] += 1
    total = max(1, int(np.sum(mask)))
    return [
        {
            "symbol": name,
            "count": count,
            "share": count / total,
        }
        for name, count in counts.most_common(limit)
    ], [
        {
            "symbol": name,
            "count": count,
            "share": count / total,
        }
        for name, count in signed.most_common(limit)
    ]


def aggregate_shapes(rows, mask):
    selected = [message_shape(row) for row, use in zip(rows, mask) if use]
    if not selected:
        return {}

    def stat(key):
        values = np.asarray([x[key] for x in selected], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, .90)),
        }

    actions = Counter(x["action"] for x in selected)
    sources = Counter(x["source"] for x in selected)

    return {
        "count": len(selected),
        "bytes": stat("bytes"),
        "chars": stat("chars"),
        "urlCount": stat("urlCount"),
        "rspamdScore": stat("rspamdScore"),
        "htmlShare": sum(x["hasHtml"] for x in selected) / len(selected),
        "actions": dict(actions.most_common()),
        "sources": dict(sources.most_common()),
    }


def blockers(feat, gate, indices):
    reasons = Counter()
    combinations = Counter()
    details = []

    checks = [
        ("score", lambda i: feat["score"][i] >= gate["scoreThreshold"]),
        ("consensus", lambda i: feat["consensus"][i] >= gate["consensusThreshold"]),
        ("median", lambda i: feat["median"][i] >= gate["medianThreshold"]),
        ("specialist", lambda i: feat["specialist"][i] >= gate["specialistThreshold"]),
        ("errorExpert", lambda i: feat["errorExpert"][i] >= gate["errorExpertThreshold"]),
        ("hamVeto", lambda i: feat["ham"][i] <= gate["hamVetoThreshold"]),
        ("agreement", lambda i: feat["agreement"][i] >= gate["minAgreement"]),
    ]

    for i in indices:
        failed = [name for name, fn in checks if not fn(i)]
        for reason in failed:
            reasons[reason] += 1
        combinations[" + ".join(failed) if failed else "none"] += 1

        details.append({
            "index": int(i),
            "failed": failed,
            "score": float(feat["score"][i]),
            "consensus": float(feat["consensus"][i]),
            "median": float(feat["median"][i]),
            "specialist": float(feat["specialist"][i]),
            "errorExpert": float(feat["errorExpert"][i]),
            "ham": float(feat["ham"][i]),
            "agreement": int(feat["agreement"][i]),
        })

    return {
        "reasonCounts": dict(reasons.most_common()),
        "topCombinations": dict(combinations.most_common(20)),
        "details": details,
    }


def main():
    REPORTS.mkdir(exist_ok=True)
    b.wait_rspamd()

    groups = b.prepare()
    train, val, test = v3.build_splits(groups)

    b.reset_bayes()
    b.learn(train)

    print("[1/7] scan train", flush=True)
    tr = b.scan_many(train, "v15-analysis-train")
    print("[2/7] scan validation", flush=True)
    va = b.scan_many(val, "v15-analysis-val")
    print("[3/7] scan test", flush=True)
    te = b.scan_many(test, "v15-analysis-test")

    xt, xv, xe, ctx_t, ctx_v, ctx_e = v3.matrices(tr, va, te)
    residual_idx = [i for i, row in enumerate(tr) if not row["rspam"]]
    residual_rows = [tr[i] for i in residual_idx]
    xr_text = xt[residual_idx]
    xr_ctx = ctx_t[residual_idx]

    print("[4/7] train v15 experts", flush=True)
    hardness, _ = v15.crossfit_hardness(xr_text, xr_ctx, residual_rows)

    base_text_models = v3.fit_ensemble(xr_text, residual_rows)
    base_ctx_models = v3.fit_ensemble(xr_ctx, residual_rows)
    ham_models = v5.fit_ham_ensemble(xr_ctx, residual_rows)
    spec_text_models = v15.fit_specialist_ensemble(xr_text, residual_rows, hardness, 2100)
    spec_ctx_models = v15.fit_specialist_ensemble(xr_ctx, residual_rows, hardness, 3100)
    error_text_models, error_ctx_models, _ = v15.fit_error_experts(
        xr_text, xr_ctx, residual_rows, hardness
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

    print("[5/7] score validation and test", flush=True)
    val_probs = all_probs(xv, ctx_v)
    test_probs = all_probs(xe, ctx_e)

    pbase_text_val, pbase_ctx_val, _, _, _, _, pham_val = val_probs
    pbase_text_test, pbase_ctx_test, _, _, _, _, pham_test = test_probs

    stage1_gate, _ = v5.choose_gate(
        va, pbase_text_val, pbase_ctx_val, pham_val
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

    safe_gate = v15.choose_gate(va, stage1_val, feat_val, safe=True)
    pred = v15.predict(stage1_test, feat_test, safe_gate)

    y = np.asarray([r["y"] for r in te], dtype=bool)
    fn = y & (~pred)
    fp = (~y) & pred
    tp = y & pred
    tn = (~y) & (~pred)
    stage1_fn = y & (~stage1_test)
    rescued = y & pred & (~stage1_test)

    print("[6/7] analyze blockers and error profiles", flush=True)
    fn_idx = np.where(fn)[0]
    fp_idx = np.where(fp)[0]

    fn_symbols, fn_signed = top_symbols(te, fn)
    fp_symbols, fp_signed = top_symbols(te, fp)
    tp_symbols, _ = top_symbols(te, tp)

    block = blockers(feat_test, safe_gate, fn_idx)

    # Which FN are near the chosen gate versus fundamentally low-confidence?
    near = 0
    deep = 0
    for item in block["details"]:
        failed = item["failed"]
        if len(failed) <= 2:
            near += 1
        if item["score"] < safe_gate["scoreThreshold"] * 0.35:
            deep += 1

    # Collect compact examples for manual inspection. Public corpus only.
    examples_fn = []
    for i in sorted(
        fn_idx,
        key=lambda k: feat_test["score"][k],
        reverse=True,
    )[:25]:
        shape = message_shape(te[i])
        examples_fn.append({
            "subject": shape["subject"],
            "source": te[i]["source"],
            "rspamdScore": float(te[i]["rscore"]),
            "action": te[i]["action"],
            "score": float(feat_test["score"][i]),
            "consensus": float(feat_test["consensus"][i]),
            "specialist": float(feat_test["specialist"][i]),
            "errorExpert": float(feat_test["errorExpert"][i]),
            "hamProbability": float(feat_test["ham"][i]),
            "agreement": int(feat_test["agreement"][i]),
            "bytes": len(te[i]["raw"]),
        })

    examples_fp = []
    for i in sorted(
        fp_idx,
        key=lambda k: feat_test["score"][k],
        reverse=True,
    )[:25]:
        shape = message_shape(te[i])
        examples_fp.append({
            "subject": shape["subject"],
            "source": te[i]["source"],
            "rspamdScore": float(te[i]["rscore"]),
            "action": te[i]["action"],
            "stage1": bool(stage1_test[i]),
            "score": float(feat_test["score"][i]),
            "consensus": float(feat_test["consensus"][i]),
            "specialist": float(feat_test["specialist"][i]),
            "errorExpert": float(feat_test["errorExpert"][i]),
            "hamProbability": float(feat_test["ham"][i]),
            "agreement": int(feat_test["agreement"][i]),
            "bytes": len(te[i]["raw"]),
        })

    result = {
        "version": "v15-error-analysis",
        "counts": {
            "spam": int(y.sum()),
            "ham": int((~y).sum()),
            "truePositive": int(tp.sum()),
            "falseNegative": int(fn.sum()),
            "falsePositive": int(fp.sum()),
            "trueNegative": int(tn.sum()),
            "stage1FalseNegative": int(stage1_fn.sum()),
            "rescuedByV15": int(rescued.sum()),
        },
        "safeGate": safe_gate,
        "falseNegativeBlockers": block,
        "falseNegativeNearGate": near,
        "falseNegativeDeepLowScore": deep,
        "falseNegativeShapes": aggregate_shapes(te, fn),
        "falsePositiveShapes": aggregate_shapes(te, fp),
        "truePositiveShapes": aggregate_shapes(te, tp),
        "falseNegativeTopSymbols": fn_symbols,
        "falseNegativeTopSignedSymbols": fn_signed,
        "falsePositiveTopSymbols": fp_symbols,
        "falsePositiveTopSignedSymbols": fp_signed,
        "truePositiveTopSymbols": tp_symbols,
        "falseNegativeExamples": examples_fn,
        "falsePositiveExamples": examples_fp,
        "methodologyWarning": (
            "This is post-hoc diagnosis on an already observed public test set. "
            "Use the findings to design features, but validate any new model on "
            "a fresh lockbox before claiming improvement."
        ),
    }

    (REPORTS / "v15-error-analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )

    print("[7/7] write summary", flush=True)
    reasons = block["reasonCounts"]
    reason_lines = [
        f"- {name}: {count}/{int(fn.sum())} ({count/max(1,int(fn.sum())):.1%})"
        for name, count in reasons.items()
    ]

    md = [
        "# MailGuard v15 error analysis",
        "",
        f"Errors on current public test: **{int(fn.sum())} FN** and **{int(fp.sum())} FP**.",
        f"Stage 1 had {int(stage1_fn.sum())} FN; v15 rescued {int(rescued.sum())} of them.",
        "",
        "## Why the remaining spam is missed",
        "",
        *reason_lines,
        "",
        f"Near-gate FN (fail <=2 rescue checks): {near}/{int(fn.sum())}.",
        f"Deep low-score FN (<35% of score threshold): {deep}/{int(fn.sum())}.",
        "",
        "## Shape comparison",
        "",
        f"FN median Rspamd score: {result['falseNegativeShapes']['rspamdScore']['median']:.3f}.",
        f"TP median Rspamd score: {result['truePositiveShapes']['rspamdScore']['median']:.3f}.",
        f"FN median size: {result['falseNegativeShapes']['bytes']['median']:.0f} bytes.",
        f"TP median size: {result['truePositiveShapes']['bytes']['median']:.0f} bytes.",
        f"FN HTML share: {result['falseNegativeShapes']['htmlShare']:.1%}.",
        f"TP HTML share: {result['truePositiveShapes']['htmlShare']:.1%}.",
        "",
        "## Top symbols among FN",
        "",
    ]

    for item in fn_symbols[:12]:
        md.append(
            f"- {item['symbol']}: {item['count']} ({item['share']:.1%})"
        )

    md += [
        "",
        "## Top symbols among FP",
        "",
    ]
    for item in fp_symbols[:12]:
        md.append(
            f"- {item['symbol']}: {item['count']} ({item['share']:.1%})"
        )

    md += [
        "",
        "This analysis is diagnostic only. Any feature changes derived from it "
        "must be checked on a fresh holdout.",
    ]

    text = "\n".join(md) + "\n"
    (REPORTS / "v15-error-analysis.md").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
