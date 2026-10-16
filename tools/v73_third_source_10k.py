#!/usr/bin/env python3
"""v73: frozen v69 vs frozen v72 on a new republisher (not necessarily new origin).

HuggingFace locuoco/the-biggest-spam-ham-phish-email-dataset-300000 is a
preprocessed mixture of older public Enron/spam/phish sources: source-level
independence is NOT assured. Frozen before scoring, hashed identity sampling
without label selection, exact and same-prefix content screen versus every
v68 source row and prior v70/v71 external source rows. No tuning from TEST,
no 90%-any-mail or commercial FPR claim. No raw mail in reports or artifacts.

Because repository data is already parsed, this does NOT validate full MIME
attachments, sender metadata or live production Rspamd integration.
"""
import argparse
import hashlib
import heapq
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
import joblib
from huggingface_hub import HfApi
from datasets import load_dataset
import v68_ru_en_research as v68
import v69_ru_en_nbsvm as v69
import v70_external_transfer as v70
import v71_second_external_10k as v71
import v72_diverse_english as v72

SOURCE="locuoco/the-biggest-spam-ham-phish-email-dataset-300000"
EXPECTED_ROWS=365448
SAMPLE=10000
HEAP_CAP=12000
SAMPLE_SEED="v73-fixed-label-blind-source-first-test-2026-10-16"
EXPECTED_V72_SHA="cc99c9e2b2c1d6e36fdfa0c8404c74191e7cfb3a4f2316cad9722b06146eb737"
RESULTS_P=("zero-cal-fp","research-0p1pct-cal-fpr")
MAX_RAW_LENGTH=150000

def lexical_cohort_filter(value):
    if not isinstance(value,str) or len(value)>MAX_RAW_LENGTH:return None
    if v68.language_fraction(value)[1]<.55:return None
    c=v68.clean(value)[:6000]
    if len(c)<30:return None
    return c

def rank(identity):
    return int(hashlib.sha256((SAMPLE_SEED+":"+identity).encode("ascii")).hexdigest(),16)

def source_screen(c,knownids,knownfamilies):
    i=v68.digest(c);f=v68.digest(c[:160])
    if i in knownids:return "exact_prior",i,f
    if f in knownfamilies:return "prefix_prior",i,f
    return "eligible",i,f

def keep_top(heap,identity,family,text,label,maxkeep=HEAP_CAP):
    score=rank(identity)
    row=dict(identity=identity,family=family,text=text,y=label)
    item=(-score,identity,row)
    if len(heap)<maxkeep:
        heapq.heappush(heap,item)
    elif item[0]>heap[0][0]:
        heapq.heapreplace(heap,item)

def stream_selection(revision,knownids,knownfamilies):
    raw_counts=Counter();classes=Counter();seen={};conflicts=set();heap=[]
    # Reading one fixed snapshot. No label influence in rank, overlap or sample.
    ds=load_dataset(SOURCE,split="train",streaming=True,revision=revision)
    for record in ds:
        raw_counts["rows"]+=1
        if not {"text","label"}<=set(record):
            raise ValueError("Upstream data schema changed")
        raw_label=record["label"]
        if type(raw_label) is not int or raw_label not in (0,1,2):
            raise ValueError("Unknown label scheme: 0=HAM, 1=PHISH, 2=SPAM")
        # Both phishing and spam are unwanted for a binary spam filter.
        y=0 if raw_label==0 else 1
        raw_counts["label_"+str(raw_label)]+=1
        c=lexical_cohort_filter(record["text"])
        if c is None:
            raw_counts["invalid_lang_or_length"]+=1
            continue
        status,identity,family=source_screen(c,knownids,knownfamilies)
        if status!="eligible":
            raw_counts[status]+=1
            continue
        if identity in seen:
            raw_counts["duplicate_inside_source"]+=1
            if seen[identity]!=y:
                raw_counts["conflicting_inside_source"]+=1
                conflicts.add(identity)
            continue
        seen[identity]=y
        raw_counts["eligible_preconflict"]+=1
        classes[y]+=1
        keep_top(heap,identity,family,c,y)
        if raw_counts["rows"]%50000==0:
            print("V73 source audit rows",raw_counts["rows"],"eligible",
                  raw_counts["eligible_preconflict"],flush=True)
    if raw_counts["rows"]!=EXPECTED_ROWS:
        raise ValueError(f"Dataset row count changed: {raw_counts['rows']}")
    surviving=[x[2] for x in heap if x[1] not in conflicts]
    if len(surviving)<SAMPLE:raise ValueError("Too few clean sample items after dedup/overlap")
    fixed=sorted(surviving,key=lambda r:(rank(r["identity"]),r["identity"]))[:SAMPLE]
    if len(fixed)!=SAMPLE or len({r["identity"] for r in fixed})!=SAMPLE:
        raise ValueError("Not 10000 unique rows")
    ham=sum(r["y"]==0 for r in fixed);spam=SAMPLE-ham
    if min(spam,ham)<1000:raise ValueError("10k sample has insufficient HAM or SPAM")
    raw_counts["removed_label_conflicts"]=len(conflicts)
    return fixed,dict(raw_counts),dict(eligibleClassCounts=classes,selectedHAM=ham,selectedSPAM=spam)

def fingerprint(rows):
    return v68.digest(json.dumps(sorted(
        (r["identity"],r["family"]) for r in rows),separators=(",",":")))

def score_old(old,rows):
    values=v70.scores(old,rows)
    truth=np.asarray([r["y"] for r in rows],dtype=np.int8)
    policies=old["selection"]["en"]
    return {name:v68.confusion(values[policies[name]["candidate"]],
                               truth,policies[name]["cutoff"]) for name in RESULTS_P}

def score_new(new,rows):
    values=v72.pool_scores(new["experts"]["en"],rows)
    truth=np.asarray([r["y"] for r in rows],dtype=np.int8)
    choices=new["selection"]["en"]
    return {name:v68.confusion(values[choices[name]["candidate"]],
                               truth,choices[name]["cutoff"]) for name in RESULTS_P}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--old-model",default="frozen-v69/model.joblib")
    p.add_argument("--new-model",default="frozen-v72/model.joblib")
    p.add_argument("--out",default="research-v73")
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    existing,revisions,_=v68.acquire()
    if revisions!=v69.EXPECTED_REVISIONS:
        raise ValueError("v68 baseline dataset revisions drifted")
    c2010,hash2010,_=v70.source_rows()
    if hash2010!=v72.PINNED_V70_SHA:raise ValueError("CSDMC baseline source drifted")
    zip71=v71.download_streamed()
    if v70.digest_bytes(zip71)!=v72.PINNED_V71_SHA:
        raise ValueError("v71 previously evaluated data changed")
    c71,_=v71.read_dataset(zip71)
    oldids,oldfam=v70.known_index(existing+c2010+c71)
    revision=HfApi().dataset_info(SOURCE).sha
    if not isinstance(revision,str) or len(revision)!=40:
        raise ValueError("No immutable HuggingFace SHA")
    rows,raw,counts=stream_selection(revision,oldids,oldfam)
    manifest=dict(source=SOURCE,sourceRevision=revision,
        sourceAdvertisedLicense="MIT (original underlying corpora rights not independently checked)",
        reusedPublicOriginsPossible=True,
        datasetType="merged preprocessed English spam/phishing/HAM",
        labelMapping="0 HAM, 1 or 2 unwanted",
        scannedRawRows=raw["rows"],sourceCounts=raw,counts=counts,
        screenPrior=("v68 all datasets","CSDMC2010","v71 provider all"),
        labelBlindSampleSeed=SAMPLE_SEED,sampleCount=SAMPLE,
        sampleFingerprint=fingerprint(rows),
        sampledInputIsNotRepresentativeOfAny10k=True,
        provenanceLimitations="The dataset aggregates older public repositories, so semantic and campaign overlap with prior MailGuard models may remain despite exact and prefix content rejection.")
    (out/"new-republisher-freeze-before-scoring.json").write_text(json.dumps(manifest,indent=2,default=int))
    print("V73 NEW RELEASE FROZEN",json.dumps(manifest,default=int),flush=True)
    oldp=Path(args.old_model);newp=Path(args.new_model)
    if v70.digest_bytes(oldp.read_bytes())!=v70.V69_SHA:raise ValueError("Wrong v69 model")
    if v70.digest_bytes(newp.read_bytes())!=EXPECTED_V72_SHA:raise ValueError("Wrong v72 model")
    old=joblib.load(oldp);new=joblib.load(newp)
    if old.get("version")!=v69.VERSION or new.get("version")!=v72.VERSION:
        raise ValueError("Unexpected model format")
    old_result=score_old(old,rows)
    new_result=score_new(new,rows)
    for group in (old_result,new_result):
        if any(x["n"]!=SAMPLE for x in group.values()):
            raise ValueError("Missing score rows")
    if (v70.digest_bytes(oldp.read_bytes()),v70.digest_bytes(newp.read_bytes()))!=(
            v70.V69_SHA,EXPECTED_V72_SHA):raise ValueError("Model altered during test")
    result=dict(version=73,evaluationKind="First-use NEW REPUBLISHER, NOT origin-independent",
                freeze=manifest,oldModelSHA256=v70.V69_SHA,
                newModelSHA256=EXPECTED_V72_SHA,
                baselineV69=old_result,diverseV72=new_result,
                trainedOnChallenge=False,calibratedOnChallenge=False,
                selectionAfterTestResults=False,
                validated90OnAnyNew10000=False,productionFPRTargetValidated=False,
                validatedRussianEmail=False,
                copyrightOriginalCorporaUnconfirmed=True)
    (out/"result.json").write_text(json.dumps(result,indent=2))
    md=["# MailGuard v73 — previously unscored 10k republisher sample","",
        "**Tested without retraining or adjusting frozen v69/v72 cutoffs.**",
        "Not a new independent *original provider*: compiled corpora can have overlap beyond hashed-content screening.",
        "",f'Publisher: {SOURCE} @ {revision}',
        f'Available messages scanned: {raw["rows"]}.',
        f'Prior exact exclusions: {raw.get("exact_prior",0)}; prefix exclusions: {raw.get("prefix_prior",0)}.',
        f'Fixed sample: {SAMPLE}, SPAM/PHISH {counts["selectedSPAM"]}, HAM {counts["selectedHAM"]}.',
        "", "| Model | Policy | Recall | TP / SPAM | FP / HAM | FPR |",
        "|---|---|---:|---:|---:|---:|"]
    for model,groups in (("v69",old_result),("v72",new_result)):
        for name,d in groups.items():
            md.append(f"| {model} | {name} | {d['recall']:.2%} | "
                      f"{d['tp']} / {d['spam']} | {d['fp']} / {d['ham']} | {d['fpr']:.3%} |")
    md.extend(["","This does not demonstrate 90% on arbitrary incoming 10k mail, "
        "and not a production 0.01% FPR guarantee. Missing verified Russian MIME "
        "samples and large independent HAM data.",
        "Reprocessed/aggregated mail may share original underlying corpora with training. "
        "Do not retune models on the outcomes of this challenge.",
        "No raw email text, predictions or message IDs saved.", ""])
    (out/"result.md").write_text("\n".join(md))
    print("\n".join(md),flush=True)

if __name__=="__main__":main()
