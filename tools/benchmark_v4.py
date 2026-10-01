#!/usr/bin/env python3
import json, math, random
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_sample_weight

import benchmark_improved as b
import benchmark_v3 as v3

REPORTS = Path('reports')
SEED = 20261001
EXTRA_FP_BUDGET = 0.005

def spam_weights(rows):
    y = np.asarray([r['y'] for r in rows], dtype=np.int32)
    w = compute_sample_weight(class_weight='balanced', y=y).astype(np.float64)
    for i, r in enumerate(rows):
        if not r['y'] and 'hard_ham' in r['source']:
            w[i] *= 8.0
        if not r['y'] and r['rscore'] >= 4.0:
            w[i] *= 4.0
        if r['y'] and r['rscore'] <= 4.0:
            w[i] *= 2.0
    return w

def ham_weights(rows):
    # y=1 means "looks like legitimate mail". This model acts as a veto.
    labels = np.asarray([0 if r['y'] else 1 for r in rows], dtype=np.int32)
    w = compute_sample_weight(class_weight='balanced', y=labels).astype(np.float64)
    for i, r in enumerate(rows):
        if not r['y'] and 'hard_ham' in r['source']:
            w[i] *= 10.0
        if not r['y'] and r['rscore'] >= 4.0:
            w[i] *= 5.0
        if r['y'] and r['rscore'] <= 4.0:
            w[i] *= 1.5
    return labels, w

def fit_models(xtrain, rows):
    yspam = np.asarray([r['y'] for r in rows], dtype=np.int32)
    sw = spam_weights(rows)
    yham, hw = ham_weights(rows)

    spam_models = []
    ham_models = []
    for i, alpha in enumerate((1e-4, 3e-4, 1e-3)):
        sm = SGDClassifier(
            loss='log_loss', penalty='l2', alpha=alpha,
            max_iter=250, tol=1e-5, random_state=SEED+i,
            average=True, fit_intercept=True,
        )
        hm = SGDClassifier(
            loss='log_loss', penalty='l2', alpha=alpha,
            max_iter=250, tol=1e-5, random_state=SEED+100+i,
            average=True, fit_intercept=True,
        )
        sm.fit(xtrain, yspam, sample_weight=sw)
        hm.fit(xtrain, yham, sample_weight=hw)
        spam_models.append(sm)
        ham_models.append(hm)
    return spam_models, ham_models

def avg_prob(models, x):
    return np.mean([m.predict_proba(x)[:,1] for m in models], axis=0)

def thresholds(values):
    fixed = np.asarray([0.0,.05,.10,.20,.30,.40,.50,.60,.70,.80,.85,.90,.93,.95,.97,.98,.99,.995,.999,1.000001])
    qs = np.quantile(values, np.linspace(.40, 1.0, 80))
    return np.unique(np.concatenate([fixed, qs]))

def choose_gate(rows, pspam, pham):
    y = np.asarray([r['y'] for r in rows], dtype=bool)
    rspam = np.asarray([r['rspam'] for r in rows], dtype=bool)
    spam_total = int(y.sum())
    ham_total = int((~y).sum())
    base_fp = int(((~y) & rspam).sum())
    max_fp = base_fp + max(1, int(math.floor(ham_total * EXTRA_FP_BUDGET)))

    source_names = sorted(set(r['source'] for r in rows if not r['y']))
    source_masks = {
        s: np.asarray([(r['source'] == s and not r['y']) for r in rows], dtype=bool)
        for s in source_names
    }
    source_base = {s: int((source_masks[s] & rspam).sum()) for s in source_names}
    source_max = {
        s: source_base[s] + max(1, int(math.floor(int(source_masks[s].sum()) * EXTRA_FP_BUDGET)))
        for s in source_names
    }

    best = None
    best90 = None
    for ts in thresholds(pspam):
        spam_ok = pspam >= ts
        for th in thresholds(pham):
            # High spam confidence AND low legitimate-mail confidence.
            pred = rspam | (spam_ok & (pham <= th))
            tp = int((y & pred).sum())
            fp = int(((~y) & pred).sum())
            if fp > max_fp:
                continue

            group_ok = True
            group_fp = {}
            for s, mask in source_masks.items():
                sfp = int((mask & pred).sum())
                group_fp[s] = sfp
                if sfp > source_max[s]:
                    group_ok = False
                    break
            if not group_ok:
                continue

            rec = tp / max(1, spam_total)
            fpr = fp / max(1, ham_total)
            point = {
                'spamThreshold': float(ts),
                'hamVetoThreshold': float(th),
                'spamDetected': tp,
                'falsePositives': fp,
                'recall': rec,
                'fpr': fpr,
                'baseFalsePositives': base_fp,
                'maxFalsePositives': max_fp,
                'groupFalsePositives': group_fp,
                'groupMaxFalsePositives': source_max,
            }
            if best is None or rec > best['recall'] or (rec == best['recall'] and fp < best['falsePositives']):
                best = point
            if rec >= .90:
                if best90 is None or fp < best90['falsePositives'] or (fp == best90['falsePositives'] and rec > best90['recall']):
                    best90 = point

    if best is None:
        best = {
            'spamThreshold': 1.000001,
            'hamVetoThreshold': 0.0,
            'spamDetected': int((y & rspam).sum()),
            'falsePositives': base_fp,
            'recall': int((y & rspam).sum()) / max(1, spam_total),
            'fpr': base_fp / max(1, ham_total),
            'baseFalsePositives': base_fp,
            'maxFalsePositives': max_fp,
            'groupFalsePositives': source_base,
            'groupMaxFalsePositives': source_max,
        }
    return best, best90

def evaluate(rows, pspam, pham, gate):
    y = np.asarray([r['y'] for r in rows], dtype=bool)
    rspam = np.asarray([r['rspam'] for r in rows], dtype=bool)
    pred = rspam | ((pspam >= gate['spamThreshold']) & (pham <= gate['hamVetoThreshold']))
    spam_total = int(y.sum())
    ham_total = int((~y).sum())
    tp = int((y & pred).sum())
    fp = int(((~y) & pred).sum())
    return {
        'spamTotal': spam_total,
        'hamTotal': ham_total,
        'spamDetected': tp,
        'falsePositives': fp,
        'recall': tp / max(1, spam_total),
        'fpr': fp / max(1, ham_total),
        'spamThreshold': gate['spamThreshold'],
        'hamVetoThreshold': gate['hamVetoThreshold'],
    }, pred

def review(rows, pspam, pham):
    residual = [i for i,r in enumerate(rows) if not r['rspam']]
    k = min(max(1, math.ceil(len(rows)*.01)), len(residual))
    risk = np.clip(pspam,0,1) * (1.0 - np.clip(pham,0,1))
    top = sorted(residual, key=lambda i:risk[i], reverse=True)[:k]
    top_spam = sum(rows[i]['y'] for i in top)

    rnd = random.Random(SEED)
    vals = []
    for _ in range(2000):
        sample = rnd.sample(residual, k)
        vals.append(sum(rows[i]['y'] for i in sample))
    mean = float(np.mean(vals))
    return {
        'budget': k,
        'topRiskSpamFound': int(top_spam),
        'randomSpamFoundMean': mean,
        'lift': top_spam/mean if mean else None,
    }

def run(name, train, val, test, with_bayes):
    print('\n===',name,'===',flush=True)
    b.reset_bayes()
    if with_bayes:
        b.learn(train)

    tr = b.scan_many(train,name+'-train')
    va = b.scan_many(val,name+'-val')
    te = b.scan_many(test,name+'-test')
    base = b.base_metrics(te)

    xt,xv,xe,ctx_t,ctx_v,ctx_e = v3.matrices(tr,va,te)

    residual_idx = [i for i,r in enumerate(tr) if not r['rspam']]
    residual_rows = [tr[i] for i in residual_idx]
    # Context representation includes Rspamd metadata; the ham-veto deliberately
    # gets the same view so it can learn when Rspamd score is misleading.
    spam_models, ham_models = fit_models(ctx_t[residual_idx], residual_rows)

    pspam_val = avg_prob(spam_models,ctx_v)
    pham_val = avg_prob(ham_models,ctx_v)
    pspam_test = avg_prob(spam_models,ctx_e)
    pham_test = avg_prob(ham_models,ctx_e)

    safe_gate, gate90 = choose_gate(va,pspam_val,pham_val)
    safe,safe_pred = evaluate(te,pspam_test,pham_test,safe_gate)

    target90 = None
    if gate90 is not None:
        target90,_ = evaluate(te,pspam_test,pham_test,gate90)

    return {
        'base': base,
        'safeGateValidation': safe_gate,
        'safeTest': safe,
        'target90GateValidation': gate90,
        'target90Test': target90,
        'review1pct': review(te,pspam_test,pham_test),
    }

def main():
    REPORTS.mkdir(exist_ok=True)
    b.wait_rspamd()
    groups = b.prepare()
    train,val,test = v3.build_splits(groups)
    print('TRAIN',len(train),'VAL',len(val),'TEST',len(test),flush=True)

    plain = run('plain-v4',train,val,test,False)
    bayes = run('bayes-v4',train,val,test,True)

    result = {
        'version':'v4-ham-veto',
        'dataset':{
            'train':len(train),'validation':len(val),'test':len(test),
            'testSpam':bayes['base']['spamTotal'],'testHam':bayes['base']['hamTotal'],
        },
        'plain':plain,
        'bayes':bayes,
    }
    (REPORTS/'v4-benchmark.json').write_text(json.dumps(result,indent=2))

    rows = [
        ('Rspamd',plain['base']),
        ('Rspamd + MailGuard v4 safe',plain['safeTest']),
        ('Rspamd + Bayes',bayes['base']),
        ('Rspamd + Bayes + MailGuard v4 safe',bayes['safeTest']),
    ]
    md = [
        '# MailGuard v4 ham-veto benchmark','',
        f"Final untouched test: {result['dataset']['testSpam']} spam + {result['dataset']['testHam']} ham.",'',
        '| Mode | Spam detected | Spam recall | False positives | FP rate |',
        '|---|---:|---:|---:|---:|',
    ]
    for label,r in rows:
        md.append(f"| {label} | {r['spamDetected']}/{r['spamTotal']} | {r['recall']:.2%} | {r['falsePositives']}/{r['hamTotal']} | {r['fpr']:.3%} |")

    md += [
        '',
        '## Validation-selected 90% point',
        f"Plain: {plain['target90Test']}",
        f"Bayes: {bayes['target90Test']}",
        '',
        '## 1% residual review',
        f"Plain: random {plain['review1pct']['randomSpamFoundMean']:.2f}, top-risk {plain['review1pct']['topRiskSpamFound']}, lift {plain['review1pct']['lift']}.",
        f"Bayes: random {bayes['review1pct']['randomSpamFoundMean']:.2f}, top-risk {bayes['review1pct']['topRiskSpamFound']}, lift {bayes['review1pct']['lift']}.",
        '',
        '## Selected gates',
        f"Plain: {plain['safeGateValidation']}",
        f"Bayes: {bayes['safeGateValidation']}",
    ]
    text='\n'.join(md)+'\n'
    (REPORTS/'v4-benchmark.md').write_text(text)
    print(text,flush=True)

if __name__=='__main__':
    main()
