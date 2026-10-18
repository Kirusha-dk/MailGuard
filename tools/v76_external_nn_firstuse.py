#!/usr/bin/env python3
"""MailGuard v76: first scoring of frozen v75 ensemble on separate 2024 academic email source.

Pretrained v75 neural checkpoint and ensemble/policies are read-only. All
previously examined v68/v70/v71/v73 mail is screened for canonical exact and
first-160-char overlaps before a 10k label-blind sample is frozen.
This is not new production/customer mail and publisher originals/licensing
are not independently verifiable; do not re-calibrate on these labels.
"""
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from datasets import load_dataset
import v68_ru_en_research as v68
import v69_ru_en_nbsvm as v69
import v70_external_transfer as v70
import v71_second_external_10k as v71
import v73_third_source_10k as v73
import v74_transformer_finetune as v74
import v75_neural_continuation as v75
import v76_source_audit as audit

EXPECTED={
"final_cleaned_safe_1.csv":"5dfec12e84f4f91714d169259477f2697be3429849a72d21016dba555fed2c5b",
"final_cleaned_phis1.csv":"a274fe9efce5deb9cb52d0b823c138d4bc2f23870823c90c82d2871e1841342c",
"final_cleaned_phis2.csv":"55e1f238ec1243237236e7e1ff055b30a5c0820d4ad2000cb7fe739825681170",
"final_cleaned_phis_3.csv":"eb059239b03a5190adc3b8655e09108f5b20e470fed675e60bddfc447aaaa140",
"final_cleaned_phis_4.csv":"d510ef99b2a874d3742af372bf2ba6a5b9b3ec6ed742963214116274b8eae93b",
}
V75_SHA="f4c4dfa7a25b7b4daa60af0807a5f404b8969f594cab1b7e59b2842e73771698"
COHORT=9000
SEED="v76-firstuse-academic-2024-fixed-identity-order"
PREVIOUS_V73_REV="22fa6d1c1362518821bfb25564654348014303e4"
PROFILES=v75.PROFILES
MAX_BODY=150000

def classify_label(record,filename):
    expected=0 if "safe" in filename else 1
    c=str(record.get("category_text","")).strip().lower()
    known_ham={"ham","safe","normal","benign","legitimate","non-spam","not spam"}
    known_spam={"spam","phish","phis","phishing","malicious","fraud","scam"}
    if c in known_ham:label=0
    elif c in known_spam:label=1
    else:raise ValueError("Unknown or untrusted source label text: "+repr(c[:30]))
    if label!=expected:raise ValueError("File category contradicts row's label text")
    try:numeric=float(record.get("category_numeric",""))
    except (ValueError,TypeError):raise ValueError("Unparseable numerical class")
    if not math.isfinite(numeric) or numeric not in (0,1,2):
        raise ValueError("Invalid numerical source class")
    return label,numeric

def canonical_row(rec,filename):
    label,numeric=classify_label(rec,filename)
    subj=rec.get("subject") or ""
    body=rec.get("body") or ""
    if len(subj)+len(body)>MAX_BODY:
        return None,"oversized"
    raw="subject "+subj+" body "+body
    if v68.language_fraction(raw)[1]<.55:
        return None,"non_english"
    clean=v68.clean(raw)[:6000]
    if len(clean)<30:
        return None,"short"
    identity=v68.digest(clean)
    family=v68.digest(clean[:160])
    return dict(identity=identity,family=family,y=label,text=clean,lang="en"),None

def collect_new():
    samples=[]
    seen={}
    conflicted=set()
    input_stats=Counter()
    sha={}
    values_by_file={}
    for fname in audit.FILES:
        raw=audit.fetch(fname)
        sha[fname]=hashlib.sha256(raw).hexdigest()
        if sha[fname]!=EXPECTED[fname]:
            raise ValueError("Academic original source changed: "+fname)
        csv.field_size_limit(1000000)
        reader=csv.DictReader(io.StringIO(raw.decode("utf-8-sig","replace")))
        if list(reader.fieldnames or [])!=["body","subject","category_numeric","category_text"]:
            raise ValueError("CSV publisher schema changed")
        labels=Counter()
        for rec in reader:
            input_stats["raw_rows"]+=1
            # The publisher includes partially labelled rows; exclude them
            # rather than inferring ground truth from folder / filename.
            if not str(rec.get("category_text") or "").strip():
                input_stats["missing_label"]+=1
                continue
            row,reason=canonical_row(rec,fname)
            if reason:
                input_stats[reason]+=1;continue
            labels[(row["y"],rec["category_text"].strip().lower(),
                    str(rec["category_numeric"]).strip())]+=1
            ident=row["identity"]
            if ident in seen:
                input_stats["duplicate_inside_source"]+=1
                if seen[ident]!=row["y"]:
                    conflicted.add(ident)
                    input_stats["label_conflicting_duplicate"]+=1
                continue
            seen[ident]=row["y"]
            samples.append(row)
        values_by_file[fname]=dict((str(k),v) for k,v in labels.items())
    if conflicted:
        samples=[r for r in samples if r["identity"] not in conflicted]
    input_stats["excluded_conflicting_identities"]=len(conflicted)
    input_stats["unique_eligible"]=len(samples)
    return samples,sha,dict(input_stats),values_by_file

def canonical_known(source,ids,families):
    for r in source:
        text=r["text"]
        ids.add(v68.digest(text))
        families.add(v68.digest(text[:160]))

def build_known():
    ids=set();family=set();totals={}
    old,revisions,_=v68.acquire()
    if revisions!=v69.EXPECTED_REVISIONS:raise ValueError("Past data revision changed")
    canonical_known(old,ids,family);totals["v68_en_ru"]=len(old)
    del old
    archive,archives,_=v70.source_rows()
    if archives!={
        "ham.zip":"ae4e7e46387fa98ba3e4dbb2b11618fe63e0d2fd10966ac85707413372ce98c8",
        "spam.zip":"9d2d5a4f6d85768f53cde4ae7eba365929c8818a43d5ec085cff25815f434247"
    }:raise ValueError("CSDMC changed")
    canonical_known(archive,ids,family);totals["v70_CSDMC"]=len(archive)
    del archive
    zip71=v71.download_streamed()
    if hashlib.sha256(zip71).hexdigest()!="c24e30ac874480ae4f9eca99be20505b123276709a86a5b9a23800a557c841a9":
        raise ValueError("v71 source changed")
    old71,_=v71.read_dataset(zip71)
    canonical_known(old71,ids,family);totals["v71"]=len(old71)
    del old71,zip71
    # Historical v73 merged corpus is source-examined, even though it
    # was never used for training. Screen it too, before independent scoring.
    scanned=0
    ds=load_dataset(v73.SOURCE,split="train",streaming=True,revision=PREVIOUS_V73_REV)
    for rec in ds:
        scanned+=1
        text=v73.lexical_cohort_filter(rec.get("text"))
        if text is not None:
            ids.add(v68.digest(text))
            family.add(v68.digest(text[:160]))
        if scanned%100000==0:print("Prior v73 overlap index",scanned,flush=True)
    if scanned!=v73.EXPECTED_ROWS:raise ValueError("Historical v73 corpus row count changed")
    totals["v73_merged_rows_scanned"]=scanned
    return ids,family,totals

def screen(rows,ids,families):
    kept=[];removed=Counter()
    for r in rows:
        if r["identity"] in ids:removed["exact_prior"]+=1;continue
        if r["family"] in families:removed["prefix_prior"]+=1;continue
        kept.append(r)
    return kept,dict(removed)

def select_fixed(rows,n=COHORT):
    if len(rows)<n:
        raise ValueError(f"Only {len(rows)} source-disjoint candidates, fewer than {n}")
    # Do NOT use y, filename, classifier output or language to pick the sample.
    chosen=sorted(rows,key=lambda r:hashlib.sha256(
        (SEED+":"+r["identity"]).encode("ascii")).digest())[:n]
    if len({r["identity"] for r in chosen})!=n:
        raise ValueError("Selected duplicate")
    return chosen

def fingerprint(rows):
    return v68.digest(json.dumps(sorted((r["identity"],r["family"])
                                       for r in rows),separators=(",",":")))

def metrics(score,labels,threshold):
    return v68.confusion(score,labels,threshold)

def paired(out_old,out_new,labels):
    cnt=Counter()
    for old,new,y in zip(out_old,out_new,labels):
        key="spam" if y else "ham"
        if old and not new:cnt[key+"_old_only"]+=1
        elif new and not old:cnt[key+"_new_only"]+=1
        elif old and new:cnt[key+"_both_positive"]+=1
        else:cnt[key+"_both_negative"]+=1
    return dict(cnt)

def run():
    import argparse
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    parser=argparse.ArgumentParser()
    parser.add_argument("--v69",default="frozen-v69/model.joblib")
    parser.add_argument("--v75",default="frozen-v75/checkpoint")
    parser.add_argument("--policies",default="reports/v75-neural-policies.json")
    parser.add_argument("--output",default="research-v76")
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(3)
    sample,source_sha,rawstats,labels=collect_new()
    oldids,oldfam,priorstats=build_known()
    clean,overlap=screen(sample,oldids,oldfam)
    fixed=select_fixed(clean)
    spam=sum(x["y"] for x in fixed)
    ham=COHORT-spam
    if min(spam,ham)<1200:
        raise ValueError("Both classes too small; cannot perform balanced source audit")
    freeze=dict(source=audit.SOURCE,revision=audit.REV,
                originalsUnverified=True,licenseUnverified=True,
                sourceFilesSha256=source_sha,inputCounts=rawstats,
                publisherClassValues=labels,priorSourcesScreened=priorstats,
                priorExclusions=overlap,eligibleAfterPrior=len(clean),
                fixedCount=COHORT,selectedSpam=spam,selectedHam=ham,
                selection="smallest SHA256("+SEED+":content_identity), label-blind",
                fixedIdentityFamilyFingerprint=fingerprint(fixed),
                rawMessageTextPersisted=False,
                warning="2024 published research CSV, NOT fresh live mail; provenance and rights uncertain. First scoring of this republisher; original-source semantic overlap cannot be ruled out.")
    (out/"source-before-first-score.json").write_text(json.dumps(freeze,indent=2))
    print("V76 FROZEN UNSEEN ACADEMIC SOURCE BEFORE INFERENCE",json.dumps(
          {k:v for k,v in freeze.items() if k not in ("publisherClassValues",)},indent=2),flush=True)
    del sample,clean,oldids,oldfam
    model69=Path(args.v69)
    chk=Path(args.v75)/"model.safetensors"
    if v74.file_sha(model69)!=v75.V69_SHA or v74.file_sha(chk)!=V75_SHA:
        raise ValueError("Frozen v69/v75 model SHA differs from trained run")
    policy=json.loads(Path(args.policies).read_text())
    if policy["weightsSha256"]!=V75_SHA or not policy.get("historicalTestNotUsedForSelection"):
        raise ValueError("v75 policies not frozen prior to original test")
    for p in PROFILES:
        if p not in policy["alphaChosenOnlyOnSelect"]["en"] or p not in policy["thresholdOnlyOnDisjointCalibration"]["en"]:
            raise ValueError("Missing frozen policy")
    base=joblib.load(model69)
    if base.get("version")!=v69.VERSION:raise ValueError("Unexpected v69 classifier")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=AutoModelForSequenceClassification.from_pretrained(
        str(Path(args.v75)),local_files_only=True).to(device).eval()
    tokenizer=AutoTokenizer.from_pretrained(str(Path(args.v75)),local_files_only=True)
    v74.MAX_TOKENS=160
    labels=np.asarray([r["y"] for r in fixed],dtype=np.int8)
    # Raw scores are computed after input freeze. All policies remain unchanged.
    nn=v74.predict(net,tokenizer,fixed,device)
    base_scores,conservative_cut=v75.baseline_scores(base,"en",fixed,PROFILES[0])
    if not np.isfinite(nn).all() or not np.isfinite(base_scores).all():
        raise ValueError("NaN inference")
    all_results={}
    comparisons={}
    for profile in PROFILES:
        baseline,old_cut=v75.baseline_scores(base,"en",fixed,profile)
        chosen=policy["alphaChosenOnlyOnSelect"]["en"][profile]
        normalizer=policy["scoreNormalizationOnlyOnSelect"]["en"]
        chosen_cut=policy["thresholdOnlyOnDisjointCalibration"]["en"][profile]["cutoff"]
        blended=v75.blend(base_scores,nn,normalizer,chosen["alpha"])
        old_bool=baseline>=old_cut
        new_bool=blended>=chosen_cut
        all_results[profile]=dict(
            v69=metrics(baseline,labels,old_cut),
            v75=metrics(blended,labels,chosen_cut),
            neuralWeight=float(chosen["alpha"]),
            deltaPaired=paired(old_bool,new_bool,labels))
        if all_results[profile]["v69"]["n"]!=COHORT or all_results[profile]["v75"]["n"]!=COHORT:
            raise ValueError("Incomplete model comparison")
    if v74.file_sha(chk)!=V75_SHA or v74.file_sha(model69)!=v75.V69_SHA:
        raise ValueError("Model weights mutated after first scoring")
    report=dict(version=76,task="New 2024 academic English phishing vs HAM cohort",
                evaluationKind="FIRST SCORE of this republisher by frozen v75/v69; independent original message provenance UNVERIFIED",
                source=freeze,lockedV75SHA256=V75_SHA,lockedV69SHA256=v75.V69_SHA,
                policiesChosenBeforeSource=True,notCalibratedOnExternalSource=True,
                alreadySawOriginalUnderlyingCorporaPossible=True,
                results=all_results,englishOnly=True,russianTrueEmailValidated=False,
                productionRecall90Validated=False,productionFPR001Validated=False)
    (out/"result.json").write_text(json.dumps(report,indent=2))
    lines=["# MailGuard v76 — frozen v75 neural versus v69 on new academic EN source",
          "", "**First time scoring the 2024 study republisher: 9,000 fixed messages.**",
          "No model fitting, threshold, ensemble blending, or source selection based on these TEST outcomes.",
          "",f"Source {audit.SOURCE} pinned revision {audit.REV}.",
          f"Labels: {spam} malicious, {ham} HAM. Prior overlaps excluded: {overlap}.",
          "","| Profile | Model | Recall | TP / SPAM | FP / HAM | FPR |",
          "|---|---|---:|---:|---:|---:|"]
    for name,profiles in all_results.items():
        for key in ("v69","v75"):
            m=profiles[key]
            lines.append(f"| {name} | {key} | {m['recall']:.2%} | {m['tp']}/{m['spam']} | "
                         f"{m['fp']}/{m['ham']} | {m['fpr']:.3%} |")
        lines.append(f"Paired deltas for {name}: {profiles['deltaPaired']}.")
    lines.extend(["",
        "2024 academic publisher mirror; original email origins, label quality and licensing NOT independently verified.",
        "Prior exact and same-prefix overlaps screened versus all known v68/v70/v71/v73. Semantic family overlap is still possible.",
        "English only, cleaned CSV text, not raw customer .eml or Rspamd metadata.",
        "0.01% FP target NOT validated by 9k; do not tune with this challenge result.",
        "Do not promote a model solely because this result is favorable. No raw emails were committed.",""])
    (out/"result.md").write_text("\n".join(lines))
    print("\n".join(lines),flush=True)

if __name__=="__main__":run()
