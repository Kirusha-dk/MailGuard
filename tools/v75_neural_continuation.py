#!/usr/bin/env python3
"""MailGuard v75: larger real Transformer fine-tuning with safe v69 fallback.

Warm-start v74 from SHA-verified checkpoint, fine-tune on 10k TRAIN examples,
160-token context, separated SELECT/CAL by mail-family. Blend choice is made
only on SELECT and thresholds only on disjoint CAL. All previous 10k TEST data
is retrospective, not a new independent evaluation. RU is Telegram, not mail.
"""
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np
import v68_ru_en_research as v68
import v69_ru_en_nbsvm as v69
import v74_transformer_finetune as v74

SEED=75
V74_SHA="95363312973c9a6e1544248fd87665c4c860fbad409aa579a86c11397b35fa78"
V69_SHA="1a0b228cf86b43d06a62a98e4dc1039bb1788c4eeb14ea77efaa6eb1d9ddb228"
TRAIN_PER_CLASS={"ru":1000,"en":4000}
SIZES={"ru":(650,1400),"en":(2500,6000)}
ALPHAS=(0.,.15,.35,.6,1.)
PROFILES=("zero-cal-fp","research-0p1pct-cal-fpr")

def deterministic(rows,n,prefix):
    if len(rows)<n:raise ValueError("Insufficient rows for "+prefix)
    return sorted(rows,key=lambda r:v68.digest(prefix+":"+r["identity"]))[:n]

def split_data(partitions):
    result={}
    for lang in ("ru","en"):
        partitions_lang=partitions[lang]
        train=sum([deterministic([r for r in partitions_lang["train"] if r["y"]==y],
                    TRAIN_PER_CLASS[lang],"v75:train:"+lang+":"+str(y))
                   for y in (0,1)],[])
        candidate={"select":[],"cal":[]}
        for row in partitions_lang["cal"]:
            idx=int(v68.digest("v75:family:"+lang+":"+row["family"])[:12],16)%100
            candidate["select" if idx<40 else "cal"].append(row)
        ss,cc=SIZES[lang]
        selected=deterministic(candidate["select"],ss,"v75:select:"+lang)
        calibration=deterministic(candidate["cal"],cc,"v75:cal:"+lang)
        result[lang]=dict(train=train,select=selected,cal=calibration,
                          test=partitions_lang["test"])
        used_ids={};used_family={}
        for bucket,items in result[lang].items():
            for r in items:
                old=used_ids.setdefault(r["identity"],bucket)
                fam=used_family.setdefault(r["family"],bucket)
                if old!=bucket or fam!=bucket:
                    raise ValueError("TRAIN/SELECT/CAL/TEST group leakage")
        if len(result[lang]["test"])!=5000:raise ValueError("Invalid frozen TEST")
        for name in ("train","select","cal"):
            if len(set(r["y"] for r in result[lang][name]))!=2:
                raise ValueError("Single-label split")
    return result

def source_freeze():
    rows,revisions,_=v68.acquire()
    if revisions!=v69.EXPECTED_REVISIONS:raise ValueError("Dataset revision changed")
    src=v68.partition(rows)
    known=v68.sample_fingerprint(src)
    for lang in ("ru","en"):
        if known[lang]["test"]["identityFamilySha256"]!=v69.EXPECTED_TEST_IDS[lang]:
            raise ValueError("Previously used TEST identities changed")
    split=split_data(src)
    return split,dict(revisions=revisions,original=known,
                      selected=v68.sample_fingerprint(split),
                      oldV74Weights=V74_SHA,oldV69Weights=V69_SHA,
                      testPreviouslyMeasured=True,
                      RussianTrainingFromTelegramNotMail=True)

def baseline_scores(bundle,lang,rows,profile):
    choice=bundle["selection"][lang][profile]
    experts=bundle["experts"][lang]
    w=v69.infer_one(experts["word"],rows)
    c=v69.infer_one(experts["char"],rows)
    return v69.combine(w,c)[choice["candidate"]],float(choice["cutoff"])

def calibrate(scores,truth,allowed):
    y=np.asarray(truth,dtype=int);s=np.asarray(scores,dtype=float)
    ham=np.sort(s[y==0])[::-1];spam=s[y==1]
    if len(ham)<200 or len(spam)<100 or not np.isfinite(s).all():
        raise ValueError("Insufficient or invalid development scores")
    if allowed<0 or allowed>=len(ham):raise ValueError("Unsafe FP allowance")
    cut=float(np.nextafter(ham[allowed],np.inf))
    fp=int(np.count_nonzero(ham>=cut))
    if fp>allowed:raise ValueError("FP cap violated")
    return dict(cutoff=cut,fp=fp,tp=int(np.count_nonzero(spam>=cut)),
                ham=len(ham),spam=len(spam))

def budget(truth,profile):
    n=sum(y==0 for y in truth)
    if profile=="zero-cal-fp":return 0
    if profile=="research-0p1pct-cal-fpr":return int(n*.001)
    raise ValueError("Unsupported FP profile")

def standardizer(values,truth):
    ham=np.asarray(values)[np.asarray(truth)==0]
    if len(ham)<200:raise ValueError("Too few development HAM")
    q25,q75=np.quantile(ham,[.25,.75]);spread=float(q75-q25)
    if spread<1e-8:raise ValueError("Degenerate normalization")
    return dict(center=float(np.median(ham)),scale=spread)

def blend(base,neural,stats,alpha):
    if alpha not in ALPHAS:raise ValueError("Nonpredeclared blend")
    return ((1-alpha)*(np.asarray(base)-stats["base"]["center"])/stats["base"]["scale"]
            +alpha*(np.asarray(neural)-stats["neural"]["center"])/stats["neural"]["scale"])

def select_alpha(base,neural,truth,stats,profile):
    best=None
    for alpha in ALPHAS:
        scores=blend(base,neural,stats,alpha)
        decision=calibrate(scores,truth,budget(truth,profile))
        rank=(decision["tp"]/decision["spam"],-decision["fp"],-alpha)
        if best is None or rank>best["rank"]:
            best=dict(alpha=alpha,selectMetrics=decision,rank=rank)
    return {k:v for k,v in best.items() if k!="rank"}

def evaluate_one(scores,truth,cut):
    return v68.confusion(scores,truth,cut)

def run():
    import torch,joblib
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    ap=argparse.ArgumentParser()
    ap.add_argument("--v74",default="frozen-v74/checkpoint")
    ap.add_argument("--v69",default="frozen-v69/model.joblib")
    ap.add_argument("--out",default="research-v75")
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    torch.set_num_threads(3)
    v74.SEED=SEED
    v74.EPOCHS=1
    v74.BATCH=16
    v74.MAX_TOKENS=160
    v74.TRAIN_PER_CLASS=TRAIN_PER_CLASS
    from_checkpoint=Path(args.v74)
    if v74.file_sha(from_checkpoint/"model.safetensors")!=V74_SHA:
        raise ValueError("Wrong v74 fine-tuned model SHA")
    if v74.file_sha(args.v69)!=V69_SHA:
        raise ValueError("Wrong v69 baseline model SHA")
    base=joblib.load(args.v69)
    if base.get("version")!=v69.VERSION or base["testIdentityFingerprints"]!=v69.EXPECTED_TEST_IDS:
        raise ValueError("Wrong v69 provenance")
    data,freeze=source_freeze()
    (out/"cohort-frozen.json").write_text(json.dumps(freeze,indent=2))
    print("v75 source frozen before training",flush=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=AutoModelForSequenceClassification.from_pretrained(
        str(from_checkpoint),local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(str(from_checkpoint),local_files_only=True)
    training=data["ru"]["train"]+data["en"]["train"]
    train_report=v74.fit(model,tokenizer,training,device)
    checkpoint=out/"checkpoint"
    model.save_pretrained(checkpoint,safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    weightsha=v74.file_sha(checkpoint/"model.safetensors")
    (out/"checkpoint-locked.json").write_text(json.dumps(
        dict(modelSha256=weightsha,train=train_report,
             inputModelSha256=V74_SHA,baselineSha256=V69_SHA,
             maxTokens=160),indent=2))
    print("V75 trained checkpoint SHA frozen",weightsha,flush=True)
    # Original v69 selected signal is identical across its two profile cutoffs.
    for lang in ("ru","en"):
        if (base["selection"][lang][PROFILES[0]]["candidate"]!=
            base["selection"][lang][PROFILES[1]]["candidate"]):
            raise ValueError("Different v69 feature candidates need separate blending")
    chosen={};calibrations={};scale={};calmetrics={}
    for lang in ("ru","en"):
        s=data[lang]["select"];sy=[r["y"] for r in s]
        ns=v74.predict(model,tokenizer,s,device)
        bs,_=baseline_scores(base,lang,s,PROFILES[0])
        scale[lang]=dict(base=standardizer(bs,sy),
                         neural=standardizer(ns,sy))
        chosen[lang]={p:select_alpha(bs,ns,sy,scale[lang],p) for p in PROFILES}
        c=data[lang]["cal"];cy=[r["y"] for r in c]
        nc=v74.predict(model,tokenizer,c,device)
        bc,_=baseline_scores(base,lang,c,PROFILES[0])
        calibrations[lang]={};calmetrics[lang]={}
        for p,policy in chosen[lang].items():
            val=blend(bc,nc,scale[lang],policy["alpha"])
            selection=calibrate(val,cy,budget(cy,p))
            calibrations[lang][p]=selection
            calmetrics[lang][p]=evaluate_one(val,cy,selection["cutoff"])
        print("v75 separate SELECT and CAL",lang,chosen[lang],calmetrics[lang],flush=True)
    decisions=dict(weightsSha256=weightsha,alphaChosenOnlyOnSelect=chosen,
                   scoreNormalizationOnlyOnSelect=scale,
                   thresholdOnlyOnDisjointCalibration=calibrations,
                   calibrationMetrics=calmetrics,
                   historicalTestNotUsedForSelection=True)
    (out/"policies-locked-before-test.json").write_text(json.dumps(decisions,indent=2))
    print("V75 MODEL AND ALL ENSEMBLE POLICIES FROZEN BEFORE RETROSPECTIVE TEST",flush=True)
    # 10k already measured in v68/v69/v74; retrospective comparison only.
    new={};old={}
    for lang in ("ru","en"):
        rows=data[lang]["test"];truth=[r["y"] for r in rows]
        n=v74.predict(model,tokenizer,rows,device)
        b,_=baseline_scores(base,lang,rows,PROFILES[0])
        new[lang]={};old[lang]={}
        for p,policy in chosen[lang].items():
            pred=blend(b,n,scale[lang],policy["alpha"])
            new[lang][p]=evaluate_one(pred,truth,calibrations[lang][p]["cutoff"])
            basepred,baseline_cut=baseline_scores(base,lang,rows,p)
            old[lang][p]=evaluate_one(basepred,truth,baseline_cut)
    combined={p:v69.accumulate([new[lang][p] for lang in ("ru","en")]) for p in PROFILES}
    old_combined={p:v69.accumulate([old[lang][p] for lang in ("ru","en")]) for p in PROFILES}
    if any(m["n"]!=10000 for m in combined.values()):
        raise ValueError("Wrong historical 10k cohort")
    if v74.file_sha(checkpoint/"model.safetensors")!=weightsha:
        raise ValueError("Changed checkpoint after TEST")
    result=dict(version=75,status="DEVELOPMENT ONLY; 10k TEST RETROSPECTIVE",
       source=freeze,weightsSha256=weightsha,train=train_report,
       chosenOnSelect=chosen,calibration=calmetrics,
       testByLanguage=new,baselineByLanguage=old,
       testAll=combined,baselineAll=old_combined,
       validatedOnNewIndependentMail=False,
       validatedProductionFPR001=False,
       languages=["ru","en"],russianIsTelegram=True)
    (out/"result.json").write_text(json.dumps(result,indent=2))
    md=["# MailGuard v75 neural continuation + v69 safeguard",
        "","Retrospective reused fixed 10k, NOT an independent new-source test.",
        f'Train {len(training)} messages from verified v74 checkpoint;',
        f'context 160 tokens (v74 had 96), batch 16, one additional epoch.',
        f'Checkpoint SHA256: {weightsha}.',
        "The neural mixing weight is selected on SELECT; the threshold on family-disjoint CAL.",
        "","| Language | Profile | Model | Recall | TP/SPAM | FP/HAM | FPR |",
        "|---|---|---|---:|---:|---:|---:|"]
    for lang in ("ru","en","ALL"):
        ns=new[lang] if lang!="ALL" else combined
        os=old[lang] if lang!="ALL" else old_combined
        for p in PROFILES:
            for label,metric in (("frozen v69",os[p]),("v75 gated NN",ns[p])):
                md.append(f'| {lang} | {p} | {label} | {metric["recall"]:.2%} | '
                          f'{metric["tp"]}/{metric["spam"]} | '
                          f'{metric["fp"]}/{metric["ham"]} | {metric["fpr"]:.3%} |')
    md.extend(["","No promise of 90% on arbitrary unseen mail, or 0.01% FPR.",
       "All Russian examples are Telegram chats. Test already used repeatedly.",
       "Do not promote versions by choosing the winner on the old TEST.",""])
    (out/"result.md").write_text("\n".join(md))
    print("\n".join(md),flush=True)

if __name__=="__main__":run()
