#!/usr/bin/env python3
"""MailGuard v69: Russian/English NBSVM text experts and CAL-only model selection.

Ablation on the previously reported v68 test identifiers. Research ONLY:
not 10,000 newly collected independent emails, not a production FPR claim.
The v68 source families were already explored; nothing here is production-certified.
"""
import argparse,hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
import joblib
import v68_ru_en_research as v68

VERSION="v69-ru-en-nbsvm-retrospective"
PROFILES=("zero-cal-fp","research-0p1pct-cal-fpr")
EXPERTS=("word","char")
CANDIDATES=("word","char","mean","char-heavy")
FEATURES=2**18
TRAIN_LIMIT={"en":45000,"ru":16000}
TRUNCATE_TEXT=1600
RATIO_SMOOTH=2.0
EXPECTED_REVISIONS={
    "ru":"af298cd351ce1d257ae789571f649d4d94d2cdd6",
    "en":"deb4cc27dc2754c14cb157aace412727f77b41dd",
}
EXPECTED_TEST_IDS={
    "ru":"5cacb9d1014c4d1bb657a5a29e72bc390fa54440aa5380c292cd2f2f3316687c",
    "en":"63cb85f297e68708bf6b53092604431764cf73cd9dce5aecbf2c9a3ded53c922",
}

def text_view(rows):
    return [r["text"][:TRUNCATE_TEXT] for r in rows]

def vectorizer(kind):
    if kind not in EXPERTS: raise ValueError("Unknown expert")
    return HashingVectorizer(
        analyzer="word" if kind=="word" else "char",
        ngram_range=(1,2) if kind=="word" else (3,5),
        n_features=FEATURES,norm=None,binary=True,
        alternate_sign=False,dtype=np.float32,lowercase=False)

def nb_logratio(X, y):
    if X.shape[0]!=len(y):raise ValueError("Mismatch")
    if set(np.unique(y))!={0,1}:raise ValueError("Both TRAIN labels necessary")
    pos=np.asarray(X[y==1].sum(axis=0)).ravel()+RATIO_SMOOTH
    neg=np.asarray(X[y==0].sum(axis=0)).ravel()+RATIO_SMOOTH
    return np.log(pos/pos.sum())-np.log(neg/neg.sum())

def fit_expert(train,kind):
    ys=np.asarray([r["y"] for r in train],dtype=np.int8)
    hv=vectorizer(kind)
    X=hv.transform(text_view(train))
    r=nb_logratio(X,ys)
    # A normalized sparse log-count-ratio view; NEVER use CAL or TEST to fit it.
    weighted=normalize(X.multiply(r).tocsr(),copy=False)
    model=LinearSVC(C=0.7,class_weight={0:1.5,1:1.0},
                    dual=True,max_iter=1600,tol=0.002,random_state=69)
    model.fit(weighted,ys)
    print("v69 trained",kind,len(train),"rows, nonzeros",X.nnz,flush=True)
    return {"ratio":r.astype(np.float32),"model":model,"kind":kind}

def infer_one(expert,rows):
    hv=vectorizer(expert["kind"])
    values=[]
    for start in range(0,len(rows),192):
        X=hv.transform(text_view(rows[start:start+192]))
        features=normalize(X.multiply(expert["ratio"]).tocsr(),copy=False)
        values.extend(expert["model"].decision_function(features))
    return np.asarray(values,dtype=np.float64)

def combine(word,char):
    return {"word":word,"char":char,"mean":.5*word+.5*char,
            "char-heavy":.25*word+.75*char}

def choose_calibration(scores, labels, max_fp):
    """Select candidate by CAL recall with explicit CAL FP maximum only.

    Ranks tie-break deterministically by candidate order. No test data used.
    """
    best=None
    for name in CANDIDATES:
        values=scores[name]
        ham=np.sort(values[np.asarray(labels)==0])[::-1]
        spam=values[np.asarray(labels)==1]
        if len(ham)<500 or len(spam)<100:raise ValueError("Too little CAL")
        if max_fp>=len(ham):raise ValueError("Unsafe CAL FP allowance")
        cutoff=float(np.nextafter(ham[max_fp],np.inf))
        fp=int(np.count_nonzero(ham>=cutoff))
        if fp>max_fp:raise ValueError("CAL cap violated")
        tp=int(np.count_nonzero(spam>=cutoff))
        key=(tp,-fp)
        if best is None or key>best["key"]:
            best=dict(candidate=name,cutoff=cutoff,key=key,
                      calibrationTP=tp,calibrationSPAM=len(spam),
                      calibrationFP=fp,calibrationHAM=len(ham))
    return {k:v for k,v in best.items() if k!="key"}

def cal_allowance(n_ham,profile):
    if profile=="zero-cal-fp":return 0
    if profile=="research-0p1pct-cal-fpr":return max(0,int(n_ham*.001))
    raise ValueError("Unexpected profile")

def score_metric(values,truth,selection):
    return v68.confusion(values[selection["candidate"]],truth,selection["cutoff"])

def accumulate(metrics):
    t=Counter()
    for m in metrics:
        for k in ("tp","fn","fp","tn"):t[k]+=m[k]
    tp,fn,fp,tn=(t[k] for k in ("tp","fn","fp","tn"))
    from audit_v65_transfer_validity import wilson_upper
    return dict(n=tp+fn+fp+tn,spam=tp+fn,ham=fp+tn,tp=tp,fn=fn,
                fp=fp,tn=tn,recall=tp/(tp+fn),fpr=fp/(fp+tn),
                fpr95upper=wilson_upper(fp,fp+tn))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="research-v69")
    args=p.parse_args()
    rows,revisions,counts=v68.acquire()
    if revisions!=EXPECTED_REVISIONS:
        raise ValueError("Publisher dataset revisions changed, won't silently swap data")
    partitions=v68.partition(rows)
    hashes=v68.sample_fingerprint(partitions)
    for language in ("en","ru"):
        if hashes[language]["test"]["identityFamilySha256"]!=EXPECTED_TEST_IDS[language]:
            raise ValueError("TEST identity drift versus previously evaluated v68")
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/"cohort.json").write_text(json.dumps(dict(
        status="RETROSPECTIVE: reuse of v68 TEST identifiers",
        sourceRevisions=revisions,partitions=hashes,
        testPreviouslyMeasuredInV68=True,notIndependentEmails=True,
        warning=v68.WARNING),indent=2))
    del rows
    experts,selection,cal_scores={}, {}, {}
    for language in ("en","ru"):
        train=partitions[language]["train"]
        train=sorted(train,key=lambda x:v68.digest("v69-train:"+x["identity"]))[:TRAIN_LIMIT[language]]
        expert={k:fit_expert(train,k) for k in EXPERTS}
        experts[language]=expert
        cal=partitions[language]["cal"]
        raw=combine(*(infer_one(expert[k],cal) for k in EXPERTS))
        y=np.asarray([r["y"] for r in cal],dtype=np.int8)
        nham=int(np.count_nonzero(y==0))
        selections={}
        for profile in PROFILES:
            pick=choose_calibration(raw,y,cal_allowance(nham,profile))
            selections[profile]=pick
        selection[language]=selections
        cal_scores[language]={p:score_metric(raw,y,choice) for p,choice in selections.items()}
        print("v69 CAL (selected only on CAL)",language,selection[language],
              cal_scores[language],flush=True)
    bundle=dict(version=VERSION,experts=experts,selection=selection,
                trainingLimits=TRAIN_LIMIT,sourceRevisions=revisions,
                testIdentityFingerprints=EXPECTED_TEST_IDS,
                researchOnly=True)
    locked=out/"model.joblib"
    joblib.dump(bundle,locked,compress=3)
    modelsha=hashlib.sha256(locked.read_bytes()).hexdigest()
    print("V69 MODEL LOCKED BEFORE ACCESSING TEST LABELS",modelsha,flush=True)
    results={}
    for language in ("ru","en"):
        test=partitions[language]["test"]
        raw=combine(*(infer_one(experts[language][k],test) for k in EXPERTS))
        y=np.asarray([r["y"] for r in test],dtype=np.int8)
        results[language]={p:score_metric(raw,y,selection[language][p])
                           for p in PROFILES}
    combined={profile:accumulate([results[lang][profile] for lang in ("ru","en")])
              for profile in PROFILES}
    if any(z["n"]!=10000 for z in combined.values()):raise ValueError("10k audit failed")
    if hashlib.sha256(locked.read_bytes()).hexdigest()!=modelsha:
        raise ValueError("TEST changed model")
    report=dict(version=69,researchOnly=True,modelSha256=modelsha,
                warning=("REUSED V68 TEST: retrospective, NOT new or independent. "
                         "RU is Telegram chat, NOT emails; EN contains familiar source families. "
                         "No claims for arbitrary emails or 0.01% production FPR. "
                         "Do not tune using these TEST results."),
                features="NBSVM word and char hashed binary ngrams, log count ratios, LinearSVC",
                cutsSelectedOnCalibrationOnly=True,testSelectedOnOutcomes=False,
                freeze=hashes,trainLimits=TRAIN_LIMIT,selections=selection,
                calibration=cal_scores,testByLanguage=results,
                testAll=combined,claims90OnAny10000Validated=False,
                productionFPGoalValidated=False)
    (out/"result.json").write_text(json.dumps(report,indent=2))
    md=["# MailGuard v69 — RU/EN NBSVM A/B research",
        "", "**REUSE of v68 test IDs. NOT NEW INDEPENDENT 10K.**",
        "",report["warning"],"","Model SHA256: "+modelsha,"",
        "| Language | Profile | Recall | TP / SPAM | FP / HAM | FPR |",
        "|---|---|---:|---:|---:|---:|"]
    for language,d in [*results.items(),("ALL",combined)]:
        for profile,m in d.items():
            md.append(f"| {language} | {profile} | {m['recall']:.2%} | "
                      f"{m['tp']} / {m['spam']} | {m['fp']} / {m['ham']} | {m['fpr']:.3%} |")
    md.extend(["","Thresholds and expert choice were frozen using CAL only. "
               "TEST outcomes must not determine the next version's thresholds.",
               "Russian Telegram is only a linguistic proxy for true mail.",
               "At least ~30k independent HAM with 0 FP is needed for a ~0.01% "
               "one-sided 95% FPR bound; current 10k is too small.",""])
    (out/"result.md").write_text("\n".join(md))
    print("\n".join(md),flush=True)

if __name__=="__main__":
    main()
