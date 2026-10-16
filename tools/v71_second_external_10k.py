#!/usr/bin/env python3
"""v71: 10k first-use English mail from another external provider, frozen v69.

The GitHub dataset realprogrammersusevim/email-dataset is labeled directory 1=HAM,
2=SPAM and reports 19,528 ASCII-cleaned messages. Original mail rights/source
lineage cannot be verified; this is a public research-only transfer challenge.
SHA-fixed upstream commit, source-screened vs all v68 training/development/test
and CSDMC2010 before predictions, identity-blind 10k. v69 frozen model.
Do not adjust cutoffs on the resulting TEST; never claim arbitrary 10k mail.
"""
import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import v68_ru_en_research as v68
import v69_ru_en_nbsvm as v69
import v70_external_transfer as v70

REPO="realprogrammersusevim/email-dataset"
REV="84209612df4c55831f074a65b7c137a4b980059d"
URL="https://codeload.github.com/"+REPO+"/tar.gz/"+REV
SELECTION_SEED="v71-firstuse-english-another-provider-predeclared"
SAMPLE=10000
MAX_DOWNLOAD=90000000
MAX_FILE=300000
RESULT_P =("zero-cal-fp","research-0p1pct-cal-fpr")


def download_streamed():
    req=urllib.request.Request(URL,headers={"User-Agent":"MailGuard-v71-firstuse-evaluation"})
    with urllib.request.urlopen(req,timeout=90) as response:
        buf=response.read(MAX_DOWNLOAD+1)
    if len(buf)>MAX_DOWNLOAD:raise ValueError("Source tar.gz too large")
    return buf


def read_dataset(blob):
    candidate=[];seen={};conflicted=set();counter=Counter()
    with tarfile.open(fileobj=io.BytesIO(blob),mode="r:gz") as archive:
        members=archive.getmembers()
        if len(members)>25000:raise ValueError("Unreasonable member count")
        for member in members:
            if not member.isfile():continue
            p=Path(member.name)
            parts=p.parts
            if p.is_absolute() or ".." in parts:raise ValueError("Unsafe path")
            if "dataset" not in parts:continue
            at=parts.index("dataset")
            if len(parts)<at+3 or parts[at+1] not in ("1","2"):continue
            if member.size>MAX_FILE:
                counter["oversized"]+=1
                continue
            raw=archive.extractfile(member).read(MAX_FILE+1)
            if len(raw)>MAX_FILE:raise ValueError("Overexpanded text")
            text=v70.raw_to_text(raw)
            cyr,lat=v68.language_fraction(text)
            if lat<.55:
                counter["non_english"]+=1
                continue
            content=v68.clean(text)[:6000]
            if len(content)<30:
                counter["short"]+=1
                continue
            identity=v68.digest(content);family=v68.digest(content[:160])
            y=int(parts[at+1])-1
            if identity in seen:
                counter["duplicate"]+=1
                if seen[identity]!=y:
                    counter["label_conflict"]+=1
                    conflicted.add(identity)
                continue
            seen[identity]=y
            candidate.append(dict(text=content,identity=identity,family=family,
                                  y=y,source=REPO))
    candidate=[r for r in candidate if r["identity"] not in conflicted]
    counter["retained_raw"]=len(candidate)
    counter["retained_ham"]=sum(r["y"]==0 for r in candidate)
    counter["retained_spam"]=sum(r["y"]==1 for r in candidate)
    counter["excluded_conflicting_identities"]=len(conflicted)
    if len(candidate)<SAMPLE:
        raise ValueError(f"Not enough source emails before overlap screen {len(candidate)}")
    return candidate,dict(counter)


def select_fixed(rows,n=SAMPLE):
    if len(rows)<n:raise ValueError("Insufficient unique source mail to sample 10k")
    # Identity blind to class and outcome; fixed BEFORE any score is computed.
    return sorted(rows,key=lambda r:hashlib.sha256(
        (SELECTION_SEED+":"+r["identity"]).encode("ascii")).digest())[:n]


def external_screen(source,prior):
    ids,families=v70.known_index(prior)
    csdmc,_,_=v70.source_rows()
    csdmc_ids,csdmc_families=v70.known_index(csdmc)
    ids.update(csdmc_ids);families.update(csdmc_families)
    return v70.filter_prior(source,ids,families)


def fingerprint(rows):
    return v68.digest(json.dumps(sorted((r["identity"],r["family"])
             for r in rows),separators=(",",":")))


def report_md(report):
    lines=["# MailGuard v71 — 10,000 English mail, second first-use source",
           "","**Frozen preexisting v69 model, unmodified thresholds.**",
           "This is one additional public source, not production validation.",
           "",f'Input source: {REPO} pinned to {REV}.',
           f'Archive SHA256: {report["sourceFreeze"]["archiveSha256"]}.',
           f'Original eligible before exclusions: {report["sourceFreeze"]["rawEligible"]}.',
           f'Overlap exclusions: {report["sourceFreeze"]["exclusions"]}.',
           f'Fixed sample: {SAMPLE} ({report["sourceFreeze"]["sampleSPAM"]} SPAM,'
           f' {report["sourceFreeze"]["sampleHAM"]} HAM).',"",
           "| Frozen v69 profile | Recall | TP / SPAM | FP / HAM | FPR |",
           "|---|---:|---:|---:|---:|"]
    for label,m in report["metrics"].items():
        lines.append(f'| {label} | {m["recall"]:.2%} | {m["tp"]} / {m["spam"]} | '
                     f'{m["fp"]} / {m["ham"]} | {m["fpr"]:.3%} |')
    lines.extend(["","**Important**: source data was ASCII cleaned; original providers and"
                  " original mail licensing are not independently verified.",
                  "Same release may aggregate older public corpora; exact/prefix overlaps screened, "
                  "semantic near-duplicates cannot be completely excluded.",
                  "This is one provider, not random external suppliers and not the customer's mail.",
                  "English only; no inference about Russian email quality.",
                  "No threshold tuning allowed after these first-use measurements.",
                  "Raw emails and individual predictions/identities are never committed or uploaded.",""])
    return "\n".join(lines)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="frozen-v69/model.joblib")
    p.add_argument("--out",default="research-v71")
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    blob=download_streamed()
    arch_sha=v70.digest_bytes(blob)
    source,raw_counts=read_dataset(blob)
    prior,revisions,_=v68.acquire()
    if revisions!=v69.EXPECTED_REVISIONS:raise ValueError("Source revision of training prior changed")
    selected,exclusions=external_screen(source,prior)
    sample=select_fixed(selected)
    if sum(r["y"]==0 for r in sample)<2000 or sum(r["y"]==1 for r in sample)<2000:
        raise ValueError("Not both classes adequately represented")
    freeze=dict(source=REPO,revision=REV,archiveSha256=arch_sha,
                rawEligible=len(source),rawCounts=raw_counts,
                exclusions=exclusions,afterExclusion=len(selected),
                sampleCount=len(sample),sampleSPAM=sum(r["y"] for r in sample),
                sampleHAM=sum(1-r["y"] for r in sample),
                sampleFingerprint=fingerprint(sample),
                sampleRule=SELECTION_SEED+", smallest SHA256 over identity",
                modelDigestExpected=v70.V69_SHA,
                sourceDisjointFromKnownExactAndPrefix=True,
                allSeenSourcesExcluded=("v68-ru","v68-en","CSDMC2010"),
                caveat="New repository previously unscored; publisher may aggregate prior sources, semantic overlap not ruled out, copyright unclear, research only.")
    (out/"first-use-freeze.json").write_text(json.dumps(freeze,indent=2))
    print("V71 FIRST USE SOURCE FROZEN",json.dumps(freeze),flush=True)
    model_file=Path(args.model)
    if v70.digest_bytes(model_file.read_bytes())!=v70.V69_SHA:
        raise ValueError("Unverified v69 model SHA")
    bundle=joblib.load(model_file)
    if bundle["version"]!=v69.VERSION:raise ValueError("Wrong version")
    if bundle["testIdentityFingerprints"]!=v69.EXPECTED_TEST_IDS:raise ValueError("Wrong dataset provenance")
    candidates=v70.scores(bundle,sample)
    truth=np.asarray([r["y"] for r in sample],dtype=np.int8)
    outcomes={}
    for name in RESULT_P:
        policy=bundle["selection"]["en"][name]
        outcomes[name]=v68.confusion(candidates[policy["candidate"]],
                                     truth,policy["cutoff"])
        if outcomes[name]["n"]!=SAMPLE:raise ValueError("Scoring incomplete")
    if v70.digest_bytes(model_file.read_bytes())!=v70.V69_SHA:raise ValueError("Model modified")
    report=dict(version=71,evaluationKind="First-use external PUBLIC provider English 10000, no adaptation",
                sourceFreeze=freeze,modelSHA256=v70.V69_SHA,
                metrics=outcomes,
                trainedOnThisSource=False,
                adjustedThresholdsBasedOnChallenge=False,
                resultsNotProductionOrArbitrary10000=True,
                russianMailNotEvaluated=True,
                sourceOriginalLicenseUnknown=True)
    (out/"result.json").write_text(json.dumps(report,indent=2))
    md=report_md(report)
    (out/"result.md").write_text(md)
    print(md,flush=True)

if __name__=="__main__":
    main()
