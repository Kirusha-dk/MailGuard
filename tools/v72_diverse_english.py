#!/usr/bin/env python3
"""MailGuard v72: diverse English-source training, source-wise CAL FPR controls.

CSDMC2010 was first evaluated in v70 and is now DEV data. It is partitioned
by source-family into TRAIN and CAL, after exact/prefix screening against the
v68 source pool. It is NOT independent TEST anymore. Historical v71 test is
replayed exactly once for comparison, and is NOT another first-use evaluation.
No training/calibration from v71, no source or threshold selection on v71.

RU is unchanged frozen v69; EN retrains NB-SVM word+char text experts.
Original CSDMC redistribution rights unclear, so model research-only.
"""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import joblib
import v68_ru_en_research as v68
import v69_ru_en_nbsvm as v69
import v70_external_transfer as v70
import v71_second_external_10k as v71

VERSION="v72-diverse-english-research"
TRAIN_EXT_MULTIPLIER=4
SOURCE_SPLIT_SEED="v72-CSDMC-train-cal-family-2026-10-16"
CAL_SOURCE_CAP={"en-known":.001,"csdmc":.001}
PROFILES=("zero-cal-fp","research-0p1pct-cal-fpr")
CANDIDATES=v70.CANDIDATES
MIN_TRAIN_EXT=1800
MIN_CAL_EXT=350
PINNED_V71_SHA="c24e30ac874480ae4f9eca99be20505b123276709a86a5b9a23800a557c841a9"
PINNED_V71_SAMPLE="838cff416c33aff20eb5e63032e896875eb5e57043c7f97bdb601199923ca647"
PINNED_V70_SHA={"ham.zip":"ae4e7e46387fa98ba3e4dbb2b11618fe63e0d2fd10966ac85707413372ce98c8",
                 "spam.zip":"9d2d5a4f6d85768f53cde4ae7eba365929c8818a43d5ec085cff25815f434247"}

def partition_external(rows):
    """Freeze family-linked TRAIN/CAL groups without using classifier outcomes."""
    splits={"train":[],"cal":[]};assignment={}
    for r in rows:
        fam=r["family"]
        bucket=int(v68.digest(SOURCE_SPLIT_SEED+fam)[:12],16)%100
        split="train" if bucket<70 else "cal"
        if assignment.setdefault(fam,split)!=split:
            raise ValueError("Family crossed split")
        splits[split].append(r)
    if len(splits["train"])<MIN_TRAIN_EXT or len(splits["cal"])<MIN_CAL_EXT:
        raise ValueError("Unexpected external partition counts")
    for split in splits:
        if len(set(r["identity"] for r in splits[split]))!=len(splits[split]):
            raise ValueError("External duplicate within split")
        if len({r["y"] for r in splits[split]})!=2:
            raise ValueError("Both labels needed")
    return splits

def fingerprint(rows):
    return v68.digest(json.dumps(sorted(
        (r["identity"],r["family"]) for r in rows),separators=(",",":")))

def training_mix(original, external):
    # Deterministic, predeclared external repeats to avoid 45k known source
    # overwhelming ~2.7k CSDMC. No v71 or CAL message can enter.
    ids={r["identity"] for r in original}
    for r in external:
        if r["identity"] in ids:
            raise ValueError("TRAIN exact overlap")
    combined=list(original)+[r for r in external for _ in range(TRAIN_EXT_MULTIPLIER)]
    return combined

def pool_scores(experts, rows):
    w=v69.infer_one(experts["word"],rows)
    c=v69.infer_one(experts["char"],rows)
    result=v69.combine(w,c)
    result["min-expert"]=np.minimum(w,c)
    result["soft-consensus"]=.5*(w+c)-.25*np.abs(w-c)
    return result

def choose_multi_source(cals,profile):
    """Pick a common EN threshold by constraints on BOTH CAL sources.

    each cal group: (candidate score dict, binary label vector).
    Ranking favors worst-source SPAM recall; never read any TEST outcomes.
    """
    if profile not in PROFILES:raise ValueError("Unknown profile")
    best=None
    for candidate in CANDIDATES:
        # Threshold from BOTH groups, applied globally. We start from the
        # stricter per-source cutoff, never lower it to rescue CAL spam.
        cutoff_options=[];hams={};spams={};limits={}
        for source,(d,y) in cals.items():
            sig=d[candidate]
            hams[source]=np.sort(np.asarray(sig)[y==0])[::-1]
            spams[source]=np.asarray(sig)[y==1]
            budget=0 if profile=="zero-cal-fp" else int(len(hams[source])*CAL_SOURCE_CAP[source])
            if len(hams[source])<MIN_CAL_EXT or len(spams[source])<100:
                raise ValueError("Incomplete CAL source")
            if budget>=len(hams[source]):raise ValueError("Unsafe budget")
            cutoff_options.append(float(np.nextafter(hams[source][budget],np.inf)))
            limits[source]=budget
        cutoff=max(cutoff_options)
        measured={}
        for source in cals:
            tp=int(np.count_nonzero(spams[source]>=cutoff))
            fp=int(np.count_nonzero(hams[source]>=cutoff))
            if fp>limits[source]:raise ValueError("CAL source FP cap exceeded")
            measured[source]=dict(tp=tp,spam=len(spams[source]),fp=fp,ham=len(hams[source]),
                                  recall=tp/len(spams[source]))
        minimum=min(x["recall"] for x in measured.values())
        pooled_tp=sum(v["tp"] for v in measured.values())
        # Prioritize minimum per-source recall, then pooled recall.
        rank=(minimum,pooled_tp,-sum(v["fp"] for v in measured.values()))
        if best is None or rank>best["_rank"]:
            best=dict(candidate=candidate,cutoff=cutoff,sourceCAL=measured,_rank=rank)
    return {k:v for k,v in best.items() if k!="_rank"}

def evaluate(bundle, selection, rows):
    scores=pool_scores(bundle,rows)
    truth=np.asarray([r["y"] for r in rows],dtype=np.int8)
    return {profile:v68.confusion(scores[choice["candidate"]],truth,choice["cutoff"])
            for profile,choice in selection.items()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--old-model",default="frozen-v69/model.joblib")
    p.add_argument("--out",default="research-v72")
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)

    existing,revisions,_=v68.acquire()
    if revisions!=v69.EXPECTED_REVISIONS:raise ValueError("Known datasets changed")
    partitions=v68.partition(existing)
    hashed=v68.sample_fingerprint(partitions)
    for lang in ("ru","en"):
        if hashed[lang]["test"]["identityFamilySha256"]!=v69.EXPECTED_TEST_IDS[lang]:
            raise ValueError("Known v68 split changed")

    external,external_sha,external_raw=v70.source_rows()
    if external_sha!=PINNED_V70_SHA:
        raise ValueError("CSDMC changed after its first-use evaluation")
    known_ids,known_fam=v70.known_index(existing)
    external,overlaps=v70.filter_prior(external,known_ids,known_fam)
    epart=partition_external(external)
    source_freeze=dict(v68Sources=revisions,v68Partitions=hashed,
        externalSource=v70.SOURCE_REPO,externalArchivesSHA256=external_sha,
        externalCounts=external_raw,existingOverlap=overlaps,
        externalSplitSeed=SOURCE_SPLIT_SEED,
        extTrainSize=len(epart["train"]),extCalSize=len(epart["cal"]),
        extTrainFingerprint=fingerprint(epart["train"]),
        extCalFingerprint=fingerprint(epart["cal"]),
        extTrainRepeatFactor=TRAIN_EXT_MULTIPLIER,
        externalFirstUseAlreadyConsumedInV70=True,
        priorV71TestAlreadyConsumed=True,
        warning="v72 research only. CSDMC source copyright uncertain; CSDMC already assessed in v70. v71 previously evaluated, retrospective only.")
    (out/"development-freeze.json").write_text(json.dumps(source_freeze,indent=2))
    print("V72 DEV PARTITIONS FROZEN",json.dumps(source_freeze),flush=True)

    old=Path(args.old_model)
    if v70.digest_bytes(old.read_bytes())!=v70.V69_SHA:raise ValueError("Old model SHA mismatch")
    v69_bundle=joblib.load(old)
    if v69_bundle["version"]!=v69.VERSION:raise ValueError("Old model wrong version")

    known_train=sorted(partitions["en"]["train"],key=lambda x:v68.digest("v69-train:"+x["identity"]))[:v69.TRAIN_LIMIT["en"]]
    mixed=training_mix(known_train,epart["train"])
    print("V72 EN diverse TRAIN",len(mixed),"rows (includes repeated external TRAIN)",flush=True)
    experts={k:v69.fit_expert(mixed,k) for k in v69.EXPERTS}
    cal_groups={
        "en-known":(pool_scores(experts,partitions["en"]["cal"]),
                    np.asarray([r["y"] for r in partitions["en"]["cal"]],dtype=np.int8)),
        "csdmc":(pool_scores(experts,epart["cal"]),
                 np.asarray([r["y"] for r in epart["cal"]],dtype=np.int8))}
    selection={p:choose_multi_source(cal_groups,p) for p in PROFILES}
    bundle=dict(version=VERSION,experts={"en":experts,"ru":v69_bundle["experts"]["ru"]},
                selection={"en":selection,"ru":v69_bundle["selection"]["ru"]},
                sourceFreeze=source_freeze,researchOnly=True,
                customerCommercialDeploymentAllowed=False,
                modelChoiceDidNotReadV71Labels=True)
    locked=out/"model.joblib"
    joblib.dump(bundle,locked,compress=3)
    model_sha=v70.digest_bytes(locked.read_bytes())
    (out/"calibration-before-retrospective-test.json").write_text(json.dumps(
        dict(modelSHA256=model_sha,selections=selection,
             noV71SourceTrainedOrCalibrated=True),indent=2))
    print("V72 MODEL & CAL CUTS FROZEN BEFORE READING V71 RETRO TEST",model_sha,flush=True)

    # Old v71 is the only retrospective eval. This test is NOT fresh after v71.
    blob=v71.download_streamed()
    if v70.digest_bytes(blob)!=PINNED_V71_SHA:raise ValueError("v71 archive drift")
    candidate,_=v71.read_dataset(blob)
    baseline_prior=list(existing)+list(external)
    known_ids,known_fam=v70.known_index(baseline_prior)
    test,exc=v70.filter_prior(candidate,known_ids,known_fam)
    fixed=v71.select_fixed(test)
    if v71.fingerprint(fixed)!=PINNED_V71_SAMPLE:
        raise ValueError("v71 retrospective cohort changed")
    scores_new=evaluate(experts,selection,fixed)
    scores_old=v70.scores(v69_bundle,fixed)
    y=np.asarray([r["y"] for r in fixed],dtype=np.int8)
    scores_baseline={p:v68.confusion(scores_old[v69_bundle["selection"]["en"][p]["candidate"]],
                      y,v69_bundle["selection"]["en"][p]["cutoff"]) for p in PROFILES}
    if scores_baseline["research-0p1pct-cal-fpr"]["tp"]!=4026 or scores_baseline["research-0p1pct-cal-fpr"]["fp"]!=11:
        raise ValueError("v69 baseline does not match v71 historical result")
    if v70.digest_bytes(locked.read_bytes())!=model_sha:raise ValueError("Frozen v72 model mutated")
    result=dict(version=72,modelSHA256=model_sha,oldModelSHA256=v70.V69_SHA,
                evaluation="RETROSPECTIVE v71 dataset, not first-use",
                thresholdSelection="Only distinct in-domain + CSDMC train-family-disjoint CAL, two-provider caps",
                sourceFreeze=source_freeze,selection=selection,
                retrospectiveV71Fingerprint=PINNED_V71_SAMPLE,
                resultsV69=scores_baseline,resultsV72=scores_new,
                passedAnyNewSourceFirstUse=False,hasValidated90PercentAny10k=False,
                productionFpr01PercentValidated=False,
                russianEmailValidation=False,licenseUncertainty=True)
    (out/"result.json").write_text(json.dumps(result,indent=2))
    md=["# MailGuard v72 — diverse-source English adaptation, retrospective v71 test","",
        "**Existing v71 TEST used AGAIN; this is NOT another first-use result.**",
        "",f"Locked v72 SHA256: {model_sha}",
        f"External CSDMC train/cal: {len(epart['train'])} / {len(epart['cal'])} (family-disjoint).",
        f"External train repeated {TRAIN_EXT_MULTIPLIER} times alongside {len(known_train)} in-domain English training samples.",
        "Both CAL source FP caps applied jointly. Old RU v69 model unchanged.",
        "", "| Source | Policy | Recall | TP / SPAM | FP / HAM | FPR |",
        "|---|---|---:|---:|---:|---:|"]
    for ver,data in [("v69",scores_baseline),("v72",scores_new)]:
        for profile,d in data.items():
            md.append(f"| {ver} | {profile} | {d['recall']:.2%} | "
                      f"{d['tp']} / {d['spam']} | {d['fp']} / {d['ham']} | {d['fpr']:.3%} |")
    md.extend(["", "This is not production validation; never change the model "
                    "to maximize the historical v71 TEST. Acquire different genuinely "
                    "first-use source before promotion.", "CSDMC original data license uncertain. "
                    "No client email used, no raw email stored in artifacts.",""])
    (out/"result.md").write_text("\n".join(md))
    print("\n".join(md),flush=True)

if __name__=="__main__":main()
