#!/usr/bin/env python3
"""v74: Genuine pre-trained multilingual Transformer finetuning (CPU feasible).

MiniLM BERT backbone and XLM-R tokenizer; final 3 transformer layers trainable.
TRAIN/CAL are family-grouped from previous v68 source partitions. TEST reuses
previously measured v68/v69 10k. It is retrospective, NOT 10k independent mail.
RU corpus is Telegram chat, not emails. EN origins have been used previously.
No customer mail is accessed, and no raw messages are committed.
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

MODEL="microsoft/Multilingual-MiniLM-L12-H384"
TOKENIZER="FacebookAI/xlm-roberta-base"
SEED=74
EPOCHS=1
MAX_TOKENS=96
BATCH=8
TRAIN_PER_CLASS={"ru":650,"en":1100}
CAL_LIMIT={"ru":1600,"en":3200}
PROFILES={"zero-cal-fp":0,"research-0p1pct-cal-fpr":0.001}
V69_SHA="1a0b228cf86b43d06a62a98e4dc1039bb1788c4eeb14ea77efaa6eb1d9ddb228"

def file_sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for blob in iter(lambda:f.read(1<<20),b""):
            h.update(blob)
    return h.hexdigest()

def select(rows,n,seed,balanced=False):
    import v68_ru_en_research as v68
    key=lambda r:v68.digest("v74:"+seed+":"+r["identity"])
    if balanced:
        groups={k:sorted((r for r in rows if r["y"]==k),key=key) for k in (0,1)}
        if any(len(v)<n for v in groups.values()):raise ValueError("Insufficient TRAIN class")
        chosen=groups[0][:n]+groups[1][:n]
    else:
        if len(rows)<n:raise ValueError("Too few CAL messages")
        chosen=sorted(rows,key=key)[:n]
    if len({r["identity"] for r in chosen})!=len(chosen):
        raise ValueError("Selected duplicate")
    return chosen

def freeze_splits(splits):
    final={}
    ids={};family={}
    for lang in ("ru","en"):
        d=splits[lang]
        final[lang]=dict(train=select(d["train"],TRAIN_PER_CLASS[lang],lang+"train",True),
                         cal=select(d["cal"],CAL_LIMIT[lang],lang+"cal"),
                         test=d["test"])
        if len(d["test"])!=5000:raise ValueError("Frozen test count changed")
        for split,rows in final[lang].items():
            for r in rows:
                i=(lang,r["identity"]);f=(lang,r["family"])
                if i in ids and ids[i]!=split:raise ValueError("Cross-split identity")
                if f in family and family[f]!=split:raise ValueError("Cross-split family")
                ids[i]=split;family[f]=split
    return final

def cal_cutoffs(scores,labels):
    import numpy as np
    scores=np.asarray(scores)
    labels=np.asarray(labels)
    ham=np.sort(scores[labels==0])[::-1]
    spam=scores[labels==1]
    if len(ham)<500 or len(spam)<100:raise ValueError("Insufficient CAL SPAM/HAM")
    out={}
    for name,ratio in PROFILES.items():
        allowed=int(len(ham)*ratio)
        cutoff=float(np.nextafter(ham[allowed],np.inf))
        fp=int((ham>=cutoff).sum())
        if fp>allowed:raise ValueError("CAL FP exceeded")
        out[name]=dict(cutoff=cutoff,calHAM=len(ham),calSPAM=len(spam),
                       calTP=int((spam>=cutoff).sum()),calFP=fp)
    return out

def unlock_trainable(model):
    for p in model.parameters():p.requires_grad=False
    layers=model.bert.encoder.layer
    if len(layers)!=12:raise ValueError("Different pretrained architecture")
    for layer in layers[-3:]:
        for p in layer.parameters():p.requires_grad=True
    for part in (model.bert.pooler,model.classifier):
        if part is not None:
            for p in part.parameters():p.requires_grad=True
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count<1000000:raise ValueError("Backbone was not fine-tuned")
    return count

def batches(rows,n):
    for i in range(0,len(rows),n):yield rows[i:i+n]

def encoding(tokenizer,rows):
    return tokenizer([r["text"] for r in rows],padding=True,
                     truncation=True,max_length=MAX_TOKENS,return_tensors="pt")

def predict(model,tokenizer,rows,device):
    import numpy as np
    import torch
    model.eval();output=[]
    with torch.inference_mode():
        for batch in batches(rows,32):
            tensor={k:v.to(device) for k,v in encoding(tokenizer,batch).items()}
            logits=model(**tensor).logits
            output.extend((logits[:,1]-logits[:,0]).cpu().tolist())
    return np.array(output,dtype=np.float64)

def fit(model,tokenizer,training,device):
    import torch
    count=unlock_trainable(model)
    model.to(device)
    slow=[];fast=[]
    for name,p in model.named_parameters():
        if p.requires_grad:
            (fast if name.startswith("classifier.") or ".pooler." in name else slow).append(p)
    if not slow or not fast:raise ValueError("No head/backbone weights")
    optim=torch.optim.AdamW([dict(params=slow,lr=2e-5),
                             dict(params=fast,lr=5e-4)],weight_decay=.01)
    rnd=random.Random(SEED)
    losses=[]
    for epoch in range(EPOCHS):
        items=list(training);rnd.shuffle(items);model.train()
        for step,batch in enumerate(batches(items,BATCH),1):
            features={k:v.to(device) for k,v in encoding(tokenizer,batch).items()}
            labels=torch.tensor([r["y"] for r in batch],device=device,dtype=torch.long)
            optim.zero_grad(set_to_none=True)
            loss=model(**features,labels=labels).loss
            if not torch.isfinite(loss):raise ValueError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.)
            optim.step()
            losses.append(float(loss.item()))
            if step%50==0 or step==(len(items)+BATCH-1)//BATCH:
                print("v74 actual finetune",epoch+1,step,"average loss",
                      round(sum(losses[-50:])/len(losses[-50:]),4),flush=True)
    return dict(trainableParameters=count,trainRows=len(training),steps=len(losses),
                lossLast50=sum(losses[-50:])/len(losses[-50:]))

def run():
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForSequenceClassification,XLMRobertaTokenizerFast
    import v68_ru_en_research as v68
    import v69_ru_en_nbsvm as v69
    from v69_ru_en_nbsvm import accumulate
    a=argparse.ArgumentParser();a.add_argument("--output",default="research-v74")
    args=a.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    torch.set_num_threads(3)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("v74 hardware",device,"torch",torch.__version__,flush=True)
    rows,revisions,counts=v68.acquire()
    if revisions!=v69.EXPECTED_REVISIONS:raise ValueError("Upstream publisher changed")
    all_splits=v68.partition(rows)
    full=v68.sample_fingerprint(all_splits)
    for lang in ("ru","en"):
        if full[lang]["test"]["identityFamilySha256"]!=v69.EXPECTED_TEST_IDS[lang]:
            raise ValueError("Frozen previous TEST drift")
    splits=freeze_splits(all_splits)
    provenance=dict(status="RETROSPECTIVE reuse of v68/v69 TEST 10k",
                    publisherRevisions=revisions,fullPartitionFingerprints=full,
                    selectedFingerprints=v68.sample_fingerprint(splits),
                    pretrainedModel=MODEL,tokenizer=TOKENIZER,epochs=EPOCHS,
                    maxTokens=MAX_TOKENS,trainPerClass=TRAIN_PER_CLASS,
                    calLimit=CAL_LIMIT,ruDataAreTelegramNotEmail=True,
                    englishDataNotNewOrigin=True,
                    noCustomerData=True)
    (out/"cohort-frozen.json").write_text(json.dumps(provenance,indent=2))
    del rows,all_splits
    api=HfApi()
    model_rev=api.model_info(MODEL).sha
    tokenizer_rev=api.model_info(TOKENIZER).sha
    if len(model_rev)!=40 or len(tokenizer_rev)!=40:
        raise ValueError("Pretrained checkpoint not pinned")
    print("v74 HuggingFace pinned",model_rev,tokenizer_rev,flush=True)
    tokenizer=XLMRobertaTokenizerFast.from_pretrained(TOKENIZER,revision=tokenizer_rev)
    model=AutoModelForSequenceClassification.from_pretrained(
        MODEL,revision=model_rev,num_labels=2)
    if model.config.vocab_size<tokenizer.vocab_size:
        raise ValueError("Tokenizer incompatibility")
    if model.config.hidden_size!=384:raise ValueError("Unexpected hidden size")
    training=splits["ru"]["train"]+splits["en"]["train"]
    fitinfo=fit(model,tokenizer,training,device)
    checkpoint=out/"checkpoint"
    model.save_pretrained(checkpoint,safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    weights=list(checkpoint.glob("*.safetensors"))
    if len(weights)!=1:raise ValueError("Expected one weight file")
    weightsha=file_sha(weights[0])
    (out/"model-locked.json").write_text(json.dumps(dict(modelSha256=weightsha,
         pretrainedModel=MODEL,pretrainedRevision=model_rev,tokenizerRevision=tokenizer_rev,
         training=fitinfo,checkpointSavedBeforeTEST=True),indent=2))
    print("v74 MODEL CHECKPOINT LOCKED BEFORE TEST",weightsha,flush=True)
    cutoffs={};calstats={}
    for lang in ("ru","en"):
        batch=splits[lang]["cal"]
        margin=predict(model,tokenizer,batch,device)
        labels=np.array([r["y"] for r in batch],dtype=np.int8)
        cutoffs[lang]=cal_cutoffs(margin,labels)
        calstats[lang]={p:v68.confusion(margin,labels,s["cutoff"])
                        for p,s in cutoffs[lang].items()}
        print("v74 CAL",lang,calstats[lang],flush=True)
    (out/"calibration-locked.json").write_text(json.dumps(
        dict(calibration=calstats,cutoffs=cutoffs,weights=weightsha),indent=2))
    # Only from this line may the *already previously examined* TEST be scored.
    bylang={}
    for lang in ("ru","en"):
        group=splits[lang]["test"]
        margins=predict(model,tokenizer,group,device)
        truth=np.array([r["y"] for r in group],dtype=np.int8)
        bylang[lang]={p:v68.confusion(margins,truth,s["cutoff"])
                      for p,s in cutoffs[lang].items()}
        print("v74 retrospective 5k",lang,bylang[lang],flush=True)
    overall={p:accumulate([bylang[k][p] for k in ("ru","en")]) for p in PROFILES}
    if any(m["n"]!=10000 for m in overall.values()):raise ValueError("Missing TEST")
    if file_sha(weights[0])!=weightsha:raise ValueError("Mutated weights")
    baseline=json.loads(Path("reports/v69-ru-en-result.json").read_text())
    if baseline["modelSha256"]!=V69_SHA:raise ValueError("Wrong v69 baseline")
    for lang in ("ru","en"):
        old=baseline["freeze"][lang]["test"]["identityFamilySha256"]
        if old!=provenance["selectedFingerprints"][lang]["test"]["identityFamilySha256"]:
            raise ValueError("Not same previous TEST")
    result=dict(version=74,model="Multilingual MiniLM (BERT) with XLM-R tokenizer",
                fullXLMRoBERTa=False,finetunedLayers=3,
                pretrainedRevision=model_rev,tokenizerRevision=tokenizer_rev,
                weightSHA256=weightsha,training=fitinfo,
                source=provenance,calibration=calstats,cutoffs=cutoffs,
                testByLanguage=bylang,testAll=overall,baselineV69=baseline["testAll"],
                evaluation="Research retrospective: v68-v69 known 10k",
                independentNewSourceValidation=False,
                validatedProduction90Recall=False,validatedProduction001FPR=False)
    (out/"result.json").write_text(json.dumps(result,indent=2))
    md=["# MailGuard v74: trained multilingual Transformer (not a sparse classifier)",
        "", "RETROSPECTIVE: reuses v68/v69 fixed 10k TEST; no independent first-use claim.",
        "","Pretrained encoder: "+MODEL+"; tokenizer: "+TOKENIZER+".",
        "Three of twelve Transformer layers + classification head really updated.",
        "Trainable parameters: "+str(fitinfo["trainableParameters"])+
            "; optimizer steps: "+str(fitinfo["steps"])+".",
        "RU samples = Telegram CHAT, not real email; EN sources already used.",
        "Checkpoint SHA256: "+weightsha,"",
        "| Language | Policy | Recall | TP/SPAM | FP/HAM | FPR |",
        "|---|---|---:|---:|---:|---:|"]
    for lang,group in [*bylang.items(),("ALL",overall)]:
        for p,m in group.items():
            md.append(f'| {lang} | {p} | {m["recall"]:.2%} | '
                      f'{m["tp"]}/{m["spam"]} | {m["fp"]}/{m["ham"]} | {m["fpr"]:.3%} |')
    md.extend(["","Frozen v69 reference on exactly the same historical TEST:"])
    for p,m in baseline["testAll"].items():
        md.append(f'- {p}: {m["recall"]:.2%} recall, {m["fp"]}/{m["ham"]} FP')
    md.extend(["","Do NOT promote to customer based on reused test. Need genuinely unseen "
               "Russian MIME mail, diverse unseen English mail and at least ~30k independent "
               "HAM with zero FP for 0.01% FPR confidence.",""])
    (out/"result.md").write_text("\n".join(md))
    print("\n".join(md),flush=True)

if __name__=="__main__":run()
