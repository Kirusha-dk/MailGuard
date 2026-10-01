#!/usr/bin/env python3
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

import benchmark_improved as b
import benchmark_v3 as v3
import benchmark_v5 as v5
import benchmark_v7_nn as n
import benchmark_v8_nn_veto as v8
import benchmark_v9_stack as v9

REPORTS = Path("reports")
SEED = 20261001

def ham_meta_weights(rows):
    yham = np.asarray([0 if r["y"] else 1 for r in rows], dtype=np.int32)
    w = compute_sample_weight(class_weight="balanced", y=yham).astype(np.float64)
    for i, r in enumerate(rows):
        if not r["y"] and "hard_ham" in r["source"]:
            w[i] *= 16.0
        if not r["y"] and r["rscore"] >= 4.0:
            w[i] *= 7.0
        if r["y"] and r["rscore"] <= 4.0:
            w[i] *= 1.2
    return w

def fit_ham_meta_ensemble(rows, x, idx):
    keep = [i for i in idx if not rows[i]["rspam"]]
    subrows = [rows[i] for i in keep]
    xtrain = x[keep]
    yham = np.asarray([0 if r["y"] else 1 for r in subrows], dtype=np.int32)
    weights = ham_meta_weights(subrows)

    models = []
    for i, c in enumerate((0.12, 0.25, 0.5, 1.0)):
        clf = LogisticRegression(
            C=c,
            penalty="l2",
            solver="lbfgs",
            max_iter=4000,
            random_state=SEED + 10000 + i,
        )
        clf.fit(xtrain, yham, sample_weight=weights)
        models.append(clf)
    return models

def avg_ham(models, x):
    return np.mean([m.predict_proba(x)[:, 1] for m in models], axis=0)

def grid(values, fixed):
    q = np.quantile(values, np.linspace(0.0, 1.0, 60))
    return np.unique(np.concatenate([np.asarray(fixed), q]))

def metrics(rows, pred):
    return v9.metrics(rows, pred)

def choose_dual_gate(rows, pspam, pham, indices, mode):
    subrows = [rows[i] for i in indices]
    s = pspam[indices]
    h = pham[indices]

    y = np.asarray([r["y"] for r in subrows], dtype=bool)
    base = np.asarray([r["rspam"] for r in subrows], dtype=bool)
    hard = np.asarray([
        (not r["y"]) and ("hard_ham" in r["source"])
        for r in subrows
    ], dtype=bool)
    base_hard_fp = int((hard & base).sum())

    spam_grid = grid(s, [
        .05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,
        .80,.85,.90,.93,.95,.97,.98,.99,.995,.999,1.000001
    ])
    ham_grid = grid(h, [
        .01,.02,.03,.05,.08,.10,.12,.15,.18,.20,.25,.30,.35,.40,.45,
        .50,.55,.60,.65,.70,.75,.80,.85,.90,.95,.99
    ])

    best = None
    for st in spam_grid:
        sm = s >= st
        for ht in ham_grid:
            pred = base | (sm & (h <= ht))
            cur = metrics(subrows, pred)
            hard_fp = int((hard & pred).sum())
            added_hard = hard_fp - base_hard_fp

            if mode == "safe":
                if cur["addedFalsePositives"] > 0 or added_hard > 0:
                    continue
                key = (cur["recall"], -cur["falsePositives"], st, -ht)
            elif mode == "precision98":
                if cur["recall"] < .98:
                    continue
                key = (-cur["falsePositives"], -added_hard, cur["recall"], st, -ht)
            elif mode == "balanced":
                if cur["addedFalsePositives"] > 2 or added_hard > 0:
                    continue
                key = (cur["recall"], -cur["falsePositives"], st, -ht)
            else:
                raise ValueError(mode)

            point = {
                **cur,
                "spamThreshold": float(st),
                "hamVetoThreshold": float(ht),
                "hardHamFalsePositives": hard_fp,
                "addedHardHamFalsePositives": added_hard,
            }

            if best is None:
                best = (key, point)
            elif key > best[0]:
                best = (key, point)

    if best is None:
        fallback = metrics(subrows, base.copy())
        return {
            **fallback,
            "spamThreshold": 1.000001,
            "hamVetoThreshold": 0.0,
            "hardHamFalsePositives": base_hard_fp,
            "addedHardHamFalsePositives": 0,
        }
    return best[1]

def evaluate(rows, pspam, pham, gate):
    base = np.asarray([r["rspam"] for r in rows], dtype=bool)
    pred = base | (
        (pspam >= gate["spamThreshold"]) &
        (pham <= gate["hamVetoThreshold"])
    )
    return metrics(rows, pred)

def review(rows, pspam, pham):
    residual = [i for i, r in enumerate(rows) if not r["rspam"]]
    budget = min(max(1, math.ceil(len(rows) * .01)), len(residual))
    risk = np.sqrt(np.clip(pspam,1e-9,1) * np.clip(1.0-pham,1e-9,1))
    top = sorted(residual, key=lambda i:risk[i], reverse=True)[:budget]
    top_spam = sum(rows[i]["y"] for i in top)

    rnd = random.Random(SEED)
    hits=[]
    for _ in range(2000):
        sample=rnd.sample(residual,budget)
        hits.append(sum(rows[i]["y"] for i in sample))
    mean=float(np.mean(hits))
    return {
        "budget":budget,
        "topRiskSpamFound":int(top_spam),
        "randomSpamFoundMean":mean,
        "lift":top_spam/mean if mean else None,
    }

def main():
    REPORTS.mkdir(exist_ok=True)
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    b.wait_rspamd()

    groups=b.prepare()
    train,val,test=v3.build_splits(groups)

    print("TRAIN",len(train),"VAL",len(val),"TEST",len(test),flush=True)
    print("v10: stacked spam score + independent meta ham veto",flush=True)

    b.reset_bayes()
    b.learn(train)
    tr=b.scan_many(train,"v10-train")
    va=b.scan_many(val,"v10-val")
    te=b.scan_many(test,"v10-test")
    base=b.base_metrics(te)

    xt,xv,xe,ctx_t,ctx_v,ctx_e=v3.matrices(tr,va,te)
    residual_idx=[i for i,r in enumerate(tr) if not r["rspam"]]
    residual_rows=[tr[i] for i in residual_idx]

    text_models=v3.fit_ensemble(xt[residual_idx],residual_rows)
    context_models=v3.fit_ensemble(ctx_t[residual_idx],residual_rows)
    ham_linear_models=v5.fit_ham_ensemble(ctx_t[residual_idx],residual_rows)

    ptext_val=v3.ensemble_predict(text_models,xv)
    pctx_val=v3.ensemble_predict(context_models,ctx_v)
    pham_lin_val=v5.avg_prob(ham_linear_models,ctx_v)

    ptext_test=v3.ensemble_predict(text_models,xe)
    pctx_test=v3.ensemble_predict(context_models,ctx_e)
    pham_lin_test=v5.avg_prob(ham_linear_models,ctx_e)

    residual_val=[r for r in va if not r["rspam"]]
    spam_train_ds=n.MailDataset(residual_rows,n.sample_weights(residual_rows))
    spam_val_ds=n.MailDataset(residual_val,n.sample_weights(residual_val))

    ham_train_rows=v8.flipped_rows(residual_rows)
    ham_val_rows=v8.flipped_rows(residual_val)
    ham_train_ds=n.MailDataset(ham_train_rows,v8.ham_weights(residual_rows))
    ham_val_ds=n.MailDataset(ham_val_rows,v8.ham_weights(residual_val))

    full_val_ds=n.MailDataset(va)
    full_test_ds=n.MailDataset(te)

    spam_nn_models=[
        n.train_one(spam_train_ds,spam_val_ds,SEED+200+i*97)
        for i in range(3)
    ]
    ham_nn_models=[
        n.train_one(ham_train_ds,ham_val_ds,SEED+5200+i*97)
        for i in range(3)
    ]

    pspam_nn_val=v8.predict(spam_nn_models,full_val_ds)
    pham_nn_val=v8.predict(ham_nn_models,full_val_ds)
    pspam_nn_test=v8.predict(spam_nn_models,full_test_ds)
    pham_nn_test=v8.predict(ham_nn_models,full_test_ds)

    xmeta_val=v9.meta_features(
        va,ptext_val,pctx_val,pham_lin_val,pspam_nn_val,pham_nn_val
    )
    xmeta_test=v9.meta_features(
        te,ptext_test,pctx_test,pham_lin_test,pspam_nn_test,pham_nn_test
    )

    meta_idx,gate_idx=v9.split_meta(va)

    spam_meta=v9.train_meta(va,xmeta_val,meta_idx)
    ham_meta_models=fit_ham_meta_ensemble(va,xmeta_val,meta_idx)

    pspam_val=spam_meta.predict_proba(xmeta_val)[:,1]
    pspam_test=spam_meta.predict_proba(xmeta_test)[:,1]
    pham_val=avg_ham(ham_meta_models,xmeta_val)
    pham_test=avg_ham(ham_meta_models,xmeta_test)

    gates={
        "safe":choose_dual_gate(va,pspam_val,pham_val,gate_idx,"safe"),
        "balanced":choose_dual_gate(va,pspam_val,pham_val,gate_idx,"balanced"),
        "target98":choose_dual_gate(va,pspam_val,pham_val,gate_idx,"precision98"),
    }
    tests={name:evaluate(te,pspam_test,pham_test,gate) for name,gate in gates.items()}

    result={
        "version":"v10-dual-meta-veto",
        "dataset":{
            "train":len(train),
            "validation":len(val),
            "metaValidation":len(meta_idx),
            "gateValidation":len(gate_idx),
            "test":len(test),
            "testSpam":base["spamTotal"],
            "testHam":base["hamTotal"],
        },
        "architecture":{
            "stackedSpamMeta":"LogisticRegression",
            "hamMetaEnsemble":4,
            "twoDimensionalGate":True,
            "testUntouched":True,
        },
        "rspamdBayes":base,
        "validationGates":gates,
        "test":tests,
        "review1pctResidual":review(te,pspam_test,pham_test),
    }

    (REPORTS/"v10-benchmark.json").write_text(json.dumps(result,indent=2))

    md=[
        "# MailGuard v10 dual-meta veto benchmark","",
        f"Final untouched test: {base['spamTotal']} spam + {base['hamTotal']} ham.","",
        "| Mode | Spam recall | Spam rescued over base | FP total | Added FP |",
        "|---|---:|---:|---:|---:|",
        f"| Rspamd + Bayes | {base['recall']:.2%} | 0 | {base['falsePositives']}/{base['hamTotal']} | 0 |",
    ]
    labels={
        "safe":"v10 safe",
        "balanced":"v10 balanced",
        "target98":"v10 validation target 98%",
    }
    for name in ("safe","balanced","target98"):
        item=tests[name]
        md.append(
            f"| {labels[name]} | {item['recall']:.2%} | {item['rescuedSpam']} | "
            f"{item['falsePositives']}/{item['hamTotal']} | {item['addedFalsePositives']} |"
        )

    md += [
        "","## 1% residual review","",
        f"Random mean {result['review1pctResidual']['randomSpamFoundMean']:.2f}, "
        f"top-risk {result['review1pctResidual']['topRiskSpamFound']}, "
        f"lift {result['review1pctResidual']['lift']}.",
        "","## Validation gates","",
        f"Safe: {gates['safe']}",
        f"Balanced: {gates['balanced']}",
        f"Target98: {gates['target98']}",
    ]
    text="\n".join(md)+"\n"
    (REPORTS/"v10-benchmark.md").write_text(text)
    print(text,flush=True)

if __name__=="__main__":
    main()
