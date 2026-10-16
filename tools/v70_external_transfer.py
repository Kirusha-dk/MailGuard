#!/usr/bin/env python3
"""v70: frozen v69 RU/EN classifier, CAL-only conservative alternative, first-use CSDMC2010 audit.

External CSDMC2010: source zrz1996/Spam-Email-Classifier-DataSet pinned
36b5435af66a037a5cd474874950ca38c837e953 (a repackaged version).
Read source, freeze SHA of entire inputs/eligible identity set before scoring.
Never fit, calibrate, choose policies, or change v69 weights with CSDMC labels.
No test mail contents saved. CSDMC is OLD public mail, not production data.
"""
import argparse
import email
import hashlib
import html
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter
from email import policy
from pathlib import Path

import joblib
import numpy as np
import v68_ru_en_research as v68
import v69_ru_en_nbsvm as v69

SOURCE_REPO="zrz1996/Spam-Email-Classifier-DataSet"
SOURCE_REV="36b5435af66a037a5cd474874950ca38c837e953"
SOURCE_BASE="https://raw.githubusercontent.com/"+SOURCE_REPO+"/"+SOURCE_REV+"/"
V69_SHA="1a0b228cf86b43d06a62a98e4dc1039bb1788c4eeb14ea77efaa6eb1d9ddb228"
ARCHIVES={"ham.zip":0,"spam.zip":1}
V69_PROFILE="research-0p1pct-cal-fpr"
CANDIDATES=("mean","word","char","char-heavy","min-expert","soft-consensus")
PROFILE_POLICY=("conservative-0cal-fp","sensitivity-0p05pct-cal-fp")
MIN_TOKEN_CHARS=30
MAX_UNCOMPRESSED=400000
MAX_ARCHIVE_BYTES=10000000
HTML_TAG=re.compile(r"<[^>]*>")
SCRIPT_TAG=re.compile(r"(?is)<(script|style)\b[^>]*>.*?</\1>")
HTML_BREAK=re.compile(r"(?i)<(?:br|p|div|tr|li|h[1-6])\b[^>]*>")
SPACE=re.compile(r"\s+")
MAX_TEXT=6000

def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()

def raw_to_text(raw):
    """Parse RFC822 mail when possible, while handling source archives already stripped of tags."""
    obj=email.message_from_bytes(raw,policy=policy.default)
    parts=[]
    try:
        subj=str(obj.get("Subject") or "")
    except Exception:
        subj=""
    if subj:parts.append(subj)
    iterable=obj.walk() if obj.is_multipart() else (obj,)
    for item in iterable:
        typ=item.get_content_type()
        if typ not in ("text/plain","text/html"):continue
        if item.get_content_disposition()=="attachment":continue
        try:
            body=item.get_content()
            if isinstance(body,bytes):body=body.decode("utf-8","replace")
        except Exception:
            body=item.get_payload(decode=True)
            if isinstance(body,bytes):body=body.decode("utf-8","replace")
        if body and isinstance(body,str):
            if typ=="text/html":
                body=SCRIPT_TAG.sub(" ",body)
                body=HTML_BREAK.sub(" ",body)
                body=HTML_TAG.sub(" ",body)
                body=html.unescape(body)
            parts.append(body)
    return SPACE.sub(" "," ".join(parts)).strip() if parts else SPACE.sub(" ",raw.decode("utf-8","replace")).strip()

def download_archive(name):
    url=SOURCE_BASE+name
    with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"MailGuard-first-use-v70-audit"}),timeout=60) as response:
        raw=response.read(MAX_ARCHIVE_BYTES+1)
    if len(raw)>MAX_ARCHIVE_BYTES:raise ValueError("Archive too large")
    if not zipfile.is_zipfile(io.BytesIO(raw)):raise ValueError("Not ZIP")
    return raw

def source_rows():
    rows=[];archive_sha={};stats=Counter();identity_set=set();family_labels={};conflicts=set()
    for name,y in ARCHIVES.items():
        archive=download_archive(name)
        archive_sha[name]=digest_bytes(archive)
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            names=sorted(z.infolist(),key=lambda i:i.filename)
            if len(names)>6500:raise ValueError("Too many members")
            for meta in names:
                if meta.is_dir():continue
                if meta.file_size>MAX_UNCOMPRESSED:stats["oversized_member"]+=1;continue
                if meta.flag_bits & 1:raise ValueError("Encrypted mail member")
                if meta.filename.startswith("/") or ".." in Path(meta.filename).parts:
                    raise ValueError("Unsafe ZIP paths")
                raw=z.read(meta)
                if len(raw)>MAX_UNCOMPRESSED:raise ValueError("Unsafe expansion")
                parsed=raw_to_text(raw)
                cyr,lat=v68.language_fraction(parsed)
                if lat<.55:stats["not_english"]+=1;continue
                canonical=v68.clean(parsed)[:MAX_TEXT]
                if len(canonical)<MIN_TOKEN_CHARS:stats["too_short"]+=1;continue
                identity=v68.digest(canonical)
                family=v68.digest(canonical[:160])
                if identity in identity_set:
                    stats["source_duplicate"]+=1
                    if family_labels[identity]!=y:
                        conflicts.add(identity)
                        stats["conflicting_label_identity_seen"]+=1
                    continue
                family_labels[identity]=y
                identity_set.add(identity)
                # message text only in process memory, never exported
                rows.append(dict(y=y,text=canonical,identity=identity,
                                 family=family,source=SOURCE_REPO))
    # Label contradictions invalidate BOTH copies, never preferentially keep
    # whichever archive appeared first. This happens before any model scoring.
    if conflicts:
        rows=[r for r in rows if r["identity"] not in conflicts]
        stats["excluded_conflicting_identities"]=len(conflicts)
    stats["retained_spam"]=sum(r["y"] for r in rows)
    stats["retained_ham"]=sum(1-r["y"] for r in rows)
    if len(rows)<2500 or sum(r["y"]==0 for r in rows)<1000 or sum(r["y"]==1 for r in rows)<500:
        raise ValueError("Source too small for meaningful first-use comparison")
    return rows,archive_sha,dict(stats)

def known_index(existing):
    identities=set();families=set()
    for r in existing:
        txt=r["text"]
        identities.add(v68.digest(txt))
        families.add(v68.digest(txt[:160]))
    return identities,families

def filter_prior(candidate,known_ids,known_families):
    out=[];excluded=Counter()
    for r in candidate:
        if r["identity"] in known_ids:
            excluded["exact_content_overlap"]+=1
            continue
        if r["family"] in known_families:
            excluded["first160_character_overlap"]+=1
            continue
        out.append(r)
    return out,dict(excluded)

def freeze_source(rows,sha,counts,excluded):
    return dict(source=SOURCE_REPO,sourceRevision=SOURCE_REV,
                republisherLicense="MIT on code repo, original corpus rights not independently verified",
                originalCorpus="CSDMC2010, repackaged publicly with preprocessing",
                sha256Archives=sha,sourceCounts=counts,excludedKnown=excluded,
                eligibleCount=len(rows),
                spamCount=sum(r["y"] for r in rows),
                hamCount=sum(1-r["y"] for r in rows),
                identityFamilyFingerprint=v68.digest(json.dumps(sorted(
                    (r["identity"],r["family"]) for r in rows),separators=(",",":"))),
                firstUseAtScoring=True,
                warning="No previous scoring on this corpus by this experiment; cannot rule out all semantic near-duplicates or historical indirect model exposure. Old repackaged public mail not current/customer production. Not a random 10k cohort.")

def scores(model,rows):
    experts=model["experts"]["en"]
    w=v69.infer_one(experts["word"],rows)
    c=v69.infer_one(experts["char"],rows)
    combos=v69.combine(w,c)
    combos["min-expert"]=np.minimum(w,c)
    combos["soft-consensus"]=.5*(w+c)-.25*np.abs(w-c)
    return combos

def cal_cap(n_ham,profile):
    if profile=="conservative-0cal-fp":return 0
    if profile=="sensitivity-0p05pct-cal-fp":return int(.0005*n_ham)
    raise ValueError("Unknown predeclared policy")

def calibrate(candidates,ys,allowed):
    ys=np.asarray(ys)
    ham_count=int(np.count_nonzero(ys==0))
    if ham_count<10000:raise ValueError("Insufficient calibration HAM")
    best=None
    # Predeclare a small collection, tie-break in tuple order; no TEST involved.
    for kind in CANDIDATES:
        sig=candidates[kind]
        ham=np.sort(sig[ys==0])[::-1]
        positive=sig[ys==1]
        if len(ham)<=allowed:raise ValueError("Too much FP budget")
        cutoff=float(np.nextafter(ham[allowed],float("inf")))
        tp=int(np.count_nonzero(positive>=cutoff))
        fp=int(np.count_nonzero(ham>=cutoff))
        if fp>allowed:raise ValueError("CAL FP allowance broken")
        key=(tp,-fp)
        if best is None or key>best["rank"]:
            best=dict(candidate=kind,threshold=cutoff,
                      calTP=tp,calSPAM=len(positive),
                      calFP=fp,calHAM=len(ham),rank=key)
    return {k:v for k,v in best.items() if k!="rank"}

def count_scores(candidates,ys,selection):
    return v68.confusion(candidates[selection["candidate"]],
                         ys,selection["threshold"])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="frozen-v69/model.joblib")
    p.add_argument("--out",default="research-v70")
    opt=p.parse_args()
    out=Path(opt.out);out.mkdir(parents=True,exist_ok=False)
    # Before any v70 challenge score, record upstream archive hashes and filter
    # exact source overlaps using already known public v68 inputs.
    source,raw_hash,raw_count=source_rows()
    existing,revisions,inputs=v68.acquire()
    if revisions!=v69.EXPECTED_REVISIONS:raise ValueError("Upstream v68 provenance drift")
    known_ids,known_families=known_index(existing)
    source,excluded=filter_prior(source,known_ids,known_families)
    freezer=freeze_source(source,raw_hash,raw_count,excluded)
    (out/"firstuse-freeze.json").write_text(json.dumps(freezer,indent=2))
    print("V70 FROZEN CHALLENGE INPUTS BEFORE ANY SCORING",json.dumps(freezer),flush=True)
    if len(source)<2000:raise ValueError("Not enough independent candidate mail after screening")
    partitions=v68.partition(existing)
    f=v68.sample_fingerprint(partitions)
    for lang in ("ru","en"):
        if f[lang]["test"]["identityFamilySha256"]!=v69.EXPECTED_TEST_IDS[lang]:
            raise ValueError("Previously frozen v68 inputs changed")
    modelpath=Path(opt.model)
    if digest_bytes(modelpath.read_bytes())!=V69_SHA:raise ValueError("Wrong v69 frozen model")
    model=joblib.load(modelpath)
    if model["version"]!=v69.VERSION or model["testIdentityFingerprints"]!=v69.EXPECTED_TEST_IDS:
        raise ValueError("Wrong trained model")
    cal=partitions["en"]["cal"]
    cal_y=np.asarray([r["y"] for r in cal],dtype=np.int8)
    cal_scores=scores(model,cal)
    policies={name:calibrate(cal_scores,cal_y,cal_cap(int(sum(cal_y==0)),name))
              for name in PROFILE_POLICY}
    cal_metrics={name:count_scores(cal_scores,cal_y,pick)
                 for name,pick in policies.items()}
    # Freeze every policy before the CSDMC challenge.
    selection=dict(version="v70-cal-only-policy-research",
                   frozenModelSHA256=V69_SHA,
                   policies=policies,calibration=cal_metrics,
                   calibrationSource="known in-domain English v68 CAL",
                   challengeWasNotUsedToChoosePolicy=True)
    (out/"policies-before-challenge.json").write_text(json.dumps(selection,indent=2))
    print("V70 FROZEN CAL-ONLY ENGLISH POLICIES",json.dumps(selection),flush=True)
    chall_scores=scores(model,source)
    chall_y=np.asarray([r["y"] for r in source],dtype=np.int8)
    baseline={}
    for name,settings in model["selection"]["en"].items():
        baseline[name]=v68.confusion(chall_scores[settings["candidate"]],
                                   chall_y,settings["cutoff"])
    improved={name:count_scores(chall_scores,chall_y,pick)
              for name,pick in policies.items()}
    if digest_bytes(modelpath.read_bytes())!=V69_SHA:raise ValueError("Model mutated after evaluation")
    metrics=dict(experiment="v70 external CSDMC2010 first-use transfer audit",
                 status="FROZEN v69 model and CAL-only v70 policy against FIRST-USE external repackaged CSDMC",
                 modelSHA256=V69_SHA,source=freezer,policies=policies,
                 calibration=cal_metrics,v69Baseline=baseline,v70Policies=improved,
                 customerProductionValidated=False,claims90OnAny10000Validated=False,
                 russianEmailQualityValidated=False,
                 selectionAgainstExternalChallenge=False,
                 caveats=[
                    "External corpus CSDMC2010, 2010-era republished and processed, not new customer mail.",
                    "Original publisher and original corpus redistribution rights not independently confirmed.",
                    "Exact and prefix-content overlap with v68 sources screened before scoring; semantic near-duplicates and indirect prior exposure cannot be ruled out.",
                    "Only English source. Does not validate Russian email or any arbitrary 10,000-message stream.",
                    "Labels supplied by folder (ham.zip/spam.zip), not independently adjudicated.",
                    "No first-use challenge scores may be used to retune policies; gather another genuinely new source."
                 ])
    (out/"result.json").write_text(json.dumps(metrics,indent=2))
    md=["# MailGuard v70 — English external first-use CSDMC2010 transfer",
        "", "**Frozen v69 model; v70 policies selected exclusively on prior English CAL.**",
        "No v70 model training or optimization on external challenge labels.",
        "",f'Eligible external email: {len(source)} ({freezer["spamCount"]} SPAM / {freezer["hamCount"]} HAM).',
        f'Known-archive overlaps excluded before scoring: {excluded}.',
        "", "| Policy | Recall | TP / SPAM | FP / HAM | FPR |",
        "|---|---:|---:|---:|---:|"]
    for label,d in [("v69 / "+k,m) for k,m in baseline.items()]+[
                    ("v70 / "+k,m) for k,m in improved.items()]:
        md.append(f'| {label} | {d["recall"]:.2%} | {d["tp"]} / {d["spam"]} | '
                  f'{d["fp"]} / {d["ham"]} | {d["fpr"]:.3%} |')
    md.extend(["", "**Do not tune on these external TEST labels, do not infer Russian production quality.**",
               "2010 repackaged public messages are not representative of today's customer stream.",
               "Zero observed false positives on this small HAM pool cannot establish 0.01% FPR.",
               "The owner repo has MIT license but original corpus rights need separate verification.",
               "Raw mail text and identifiers are not saved in GitHub artifacts.", ""])
    (out/"result.md").write_text("\n".join(md))
    print("\n".join(md),flush=True)

if __name__=="__main__":
    main()
