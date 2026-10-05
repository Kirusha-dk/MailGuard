#!/usr/bin/env python3
"""
MailGuard v44: domain-shift diagnostics and conservative adaptive rescue.

Important: v43 new-source labels are used here for ENGINEERING/DIAGNOSTICS because
v43 has already been opened. Therefore v44 is not a new lockbox claim. It searches
for a rescue rule that maximizes worst mixed-source recall while explicitly
constraining ham false positives, and reports the full Pareto frontier. A future
v45 must validate the chosen recipe on never-opened source families.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import benchmark_v43_newsource_v41_v42 as v43

REPORTS = Path("reports")
PRED = REPORTS / "v43-newsource-predictions.jsonl"
OUT_JSON = REPORTS / "v44-domainshift.json"
OUT_MD = REPORTS / "v44-domainshift.md"


def metrics(rows, pred):
    y = np.asarray([bool(r["label"]) for r in rows])
    p = np.asarray(pred, dtype=bool)
    tp = int((y & p).sum()); fn = int((y & ~p).sum())
    fp = int((~y & p).sum()); tn = int((~y & ~p).sum())
    return dict(tp=tp, fn=fn, fp=fp, tn=tn,
                recall=tp/max(1,tp+fn), fpr=fp/max(1,fp+tn),
                precision=tp/max(1,tp+fp))


def per_source(rows, pred):
    out={}
    for src in sorted({r["source"] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r["source"]==src]
        out[src]=metrics([rows[i] for i in ids], np.asarray(pred)[ids])
    return out


def main():
    if not PRED.exists():
        raise RuntimeError("v43 predictions missing; run v43 first")
    rows=[json.loads(x) for x in PRED.read_text().splitlines() if x.strip()]
    y=np.asarray([bool(r["label"]) for r in rows])
    v41=np.asarray([bool(r["v41Spam"]) for r in rows])
    p42=np.asarray([float(r["v42Probability"]) for r in rows])
    rs=np.asarray([float(r["rspamdScore"]) for r in rows])

    # Diagnostics: probability distributions explain whether v42 is calibratable
    # across domains or whether the representation itself has shifted.
    diag={}
    for src in sorted({r["source"] for r in rows}):
        for label in (0,1):
            idx=np.asarray([r["source"]==src and int(r["label"])==label for r in rows])
            if not idx.any():
                continue
            vals=p42[idx]
            diag[f"{src}|{label}"]={
                "n":int(idx.sum()), "p10":float(np.quantile(vals,.10)),
                "p50":float(np.quantile(vals,.50)), "p90":float(np.quantile(vals,.90)),
                "p99":float(np.quantile(vals,.99)), "mean":float(vals.mean()),
            }

    # Search a transparent adaptive rescue. V41 remains the precision anchor.
    # V42 can rescue only above a threshold and, optionally, with supporting
    # Rspamd evidence. We keep all Pareto points instead of hiding trade-offs.
    thresholds=np.unique(np.r_[np.linspace(.50,.99999,240), np.quantile(p42,np.linspace(.5,.999,180))])
    rspamd_min=[-999.0,-2.0,-1.0,0.0,0.5,1.0,2.0,3.0,4.0,5.0]
    candidates=[]
    for t in thresholds:
        for rmin in rspamd_min:
            pred=v41 | ((p42>=t) & (rs>=rmin))
            m=metrics(rows,pred)
            src=per_source(rows,pred)
            mixed=[s for s,d in src.items() if d["tp"]+d["fn"]>0 and d["fp"]+d["tn"]>0]
            spam_sources=[s for s,d in src.items() if d["tp"]+d["fn"]>0]
            worst_recall=min(src[s]["recall"] for s in spam_sources)
            worst_mixed_fpr=max(src[s]["fpr"] for s in mixed) if mixed else m["fpr"]
            candidates.append(dict(threshold=float(t),rspamdMin=float(rmin),
                worstSourceRecall=float(worst_recall),worstMixedFpr=float(worst_mixed_fpr),
                **m,bySource=src))

    # Engineering profiles. Crucially rank WORST source first, unlike v40's
    # aggregate-TP-first selector.
    profiles={}
    for name,max_fpr in [("strict",.01),("lowfp",.03),("balanced",.05),("exploratory",.10)]:
        ok=[c for c in candidates if c["fpr"]<=max_fpr and c["worstMixedFpr"]<=max_fpr*1.5]
        if ok:
            profiles[name]=max(ok,key=lambda c:(c["worstSourceRecall"],c["recall"],-c["fp"]))
        else:
            profiles[name]=min(candidates,key=lambda c:(c["fpr"],-c["worstSourceRecall"]))

    # Also expose whether 90% per spam source is mathematically reachable by
    # threshold/rspamd gating on this already-opened engineering set.
    reaches90=[c for c in candidates if c["worstSourceRecall"]>=.90]
    best90=min(reaches90,key=lambda c:(c["fpr"],c["worstMixedFpr"],-c["precision"])) if reaches90 else None

    report={"version":"v44-domainshift-engineering","labelsOpened":True,
      "warning":"v43 labels are engineering data now; validate any chosen recipe on fresh v45 sources",
      "diagnostics":diag,"profiles":profiles,"best90WorstSource":best90}
    OUT_JSON.write_text(json.dumps(report,indent=2))

    lines=["# MailGuard v44 domain-shift engineering","",
      "V43 labels are now engineering data. This is NOT a fresh lockbox result; v45 must use unseen families.","",
      "| Profile | Recall | FN | FP | FPR | Precision | Worst spam-source recall |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for name,c in profiles.items():
        lines.append(f"| {name} | {c['recall']:.2%} | {c['fn']} | {c['fp']} | {c['fpr']:.3%} | {c['precision']:.3%} | {c['worstSourceRecall']:.2%} |")
    lines+=["","V42 probability calibration by source/class:","",
      "| Source/class | N | p10 | p50 | p90 | p99 | mean |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for k,d in diag.items():
        lines.append(f"| {k} | {d['n']} | {d['p10']:.4f} | {d['p50']:.4f} | {d['p90']:.4f} | {d['p99']:.4f} | {d['mean']:.4f} |")
    if best90:
        lines+=["",f"90% worst-source recall is reachable on opened v43 data only at overall FPR **{best90['fpr']:.3%}** (FP={best90['fp']}). This is diagnostic, not a production claim."]
    else:
        lines+=["","No searched transparent v41+v42+Rspamd gate reaches 90% recall on every spam source. Representation/training must change; threshold tuning alone is insufficient."]
    OUT_MD.write_text("\n".join(lines)+"\n")
    print(OUT_MD.read_text(),flush=True)


if __name__=="__main__":
    main()

# trigger v44 Actions
