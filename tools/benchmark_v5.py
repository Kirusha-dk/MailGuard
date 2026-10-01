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

def fit_ham_ensemble(xtrain, rows):
    yham = np.asarray([0 if r['y'] else 1 for r in rows], dtype=np.int32)
    weights = compute_sample_weight(class_weight='balanced', y=yham).astype(np.float64)
    for i, r in enumerate(rows):
        if not r['y'] and 'hard_ham' in r['source']:
            weights[i] *= 12.0
        if not r['y'] and r['rscore'] >= 4.0:
            weights[i] *= 5.0
        if r['y'] and r['rscore'] <= 4.0:
            weights[i] *= 1.5

    models = []
    for i, alpha in enumerate((1e-4, 3e-4, 1e-3)):
        clf = SGDClassifier(
            loss='log_loss', penalty='l2', alpha=alpha,
            max_iter=250, tol=1e-5, random_state=SEED + i,
            average=True, fit_intercept=True,
        )
        clf.fit(xtrain, yham, sample_weight=weights)
        models.append(clf)
    return models

def avg_prob(models, x):
    return np.mean([m.predict_proba(x)[:, 1] for m in models], axis=0)

def grid(values, low=0.5):
    fixed = np.asarray([low, .55, .60, .65, .70, .75, .80, .85, .90,
                        .93, .95, .97, .98, .99, .995, .999, 1.000001])
    q = np.quantile(values, np.linspace(.55, 1.0, 28))
    return np.unique(np.concatenate([fixed, q[q >= low]]))

def ham_grid(values):
    fixed = np.asarray([0.02,.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.60,.70])
    q = np.quantile(values, np.linspace(0.0, .75, 20))
    return np.unique(np.concatenate([fixed, q]))

def predict_gate(rows, ptext, pctx, pham, tt, tc, hv):
    rspam = np.asarray([r['rspam'] for r in rows], dtype=bool)
    # Normal path: all three signals must agree.
    add = (ptext >= tt) & (pctx >= tc) & (pham <= hv)
    # Very-high-confidence bypass remains conservative: both spam models
    # extremely high and ham model very low.
    ultra = (ptext >= .995) & (pctx >= .95) & (pham <= .10)
    return rspam | add | ultra

def stats(rows, pred):
    y = np.asarray([r['y'] for r in rows], dtype=bool)
    spam = int(y.sum())
    ham = int((~y).sum())
    tp = int((y & pred).sum())
    fp = int(((~y) & pred).sum())
    return {
        'spamTotal': spam, 'hamTotal': ham,
        'spamDetected': tp, 'falsePositives': fp,
        'recall': tp / max(1, spam),
        'fpr': fp / max(1, ham),
    }

def choose_gate(rows, ptext, pctx, pham):
    y = np.asarray([r['y'] for r in rows], dtype=bool)
    rspam = np.asarray([r['rspam'] for r in rows], dtype=bool)

    ham_mask = ~y
    hard = np.asarray([('hard_ham' in r['source']) and (not r['y']) for r in rows], dtype=bool)
    easy = np.asarray([('easy_ham' in r['source']) and (not r['y']) for r in rows], dtype=bool)

    base_fp = int((ham_mask & rspam).sum())
    base_hard_fp = int((hard & rspam).sum())
    base_easy_fp = int((easy & rspam).sum())

    # We want test FP near 0.5-0.8%, so validation is stricter:
    # at most one added ham overall, and none on hard ham.
    max_fp = base_fp + 1
    max_hard_fp = base_hard_fp
    max_easy_fp = base_easy_fp + 1

    best = None
    best90 = None
    for tt in grid(ptext, .50):
        for tc in grid(pctx, .50):
            for hv in ham_grid(pham):
                pred = predict_gate(rows, ptext, pctx, pham, tt, tc, hv)
                s = stats(rows, pred)
                hard_fp = int((hard & pred).sum())
                easy_fp = int((easy & pred).sum())

                point = {
                    **s,
                    'textThreshold': float(tt),
                    'contextThreshold': float(tc),
                    'hamVetoThreshold': float(hv),
                    'hardHamFalsePositives': hard_fp,
                    'easyHamFalsePositives': easy_fp,
                    'baseFalsePositives': base_fp,
                    'maxFalsePositives': max_fp,
                }

                if s['recall'] >= .90:
                    if (best90 is None or s['falsePositives'] < best90['falsePositives'] or
                        (s['falsePositives'] == best90['falsePositives'] and s['recall'] > best90['recall'])):
                        best90 = point

                if s['falsePositives'] > max_fp:
                    continue
                if hard_fp > max_hard_fp:
                    continue
                if easy_fp > max_easy_fp:
                    continue

                if (best is None or s['recall'] > best['recall'] or
                    (s['recall'] == best['recall'] and s['falsePositives'] < best['falsePositives'])):
                    best = point

    if best is None:
        pred = rspam
        s = stats(rows, pred)
        best = {
            **s,
            'textThreshold': 1.000001,
            'contextThreshold': 1.000001,
            'hamVetoThreshold': 0.0,
            'hardHamFalsePositives': base_hard_fp,
            'easyHamFalsePositives': base_easy_fp,
            'baseFalsePositives': base_fp,
            'maxFalsePositives': max_fp,
        }
    return best, best90

def review(rows, ptext, pctx, pham):
    residual = [i for i, r in enumerate(rows) if not r['rspam']]
    k = min(max(1, math.ceil(len(rows) * .01)), len(residual))
    risk = np.cbrt(np.clip(ptext, 1e-9, 1) *
                   np.clip(pctx, 1e-9, 1) *
                   np.clip(1.0 - pham, 1e-9, 1))
    top = sorted(residual, key=lambda i: risk[i], reverse=True)[:k]
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
        'lift': top_spam / mean if mean else None,
    }

def main():
    REPORTS.mkdir(exist_ok=True)
    b.wait_rspamd()
    groups = b.prepare()
    train, val, test = v3.build_splits(groups)

    print('TRAIN', len(train), 'VAL', len(val), 'TEST', len(test), flush=True)
    print('Bayes-only optimization run', flush=True)

    b.reset_bayes()
    b.learn(train)
    tr = b.scan_many(train, 'v5-train')
    va = b.scan_many(val, 'v5-val')
    te = b.scan_many(test, 'v5-test')

    base = b.base_metrics(te)
    xt, xv, xe, ctx_t, ctx_v, ctx_e = v3.matrices(tr, va, te)

    residual_idx = [i for i, r in enumerate(tr) if not r['rspam']]
    residual_rows = [tr[i] for i in residual_idx]

    text_models = v3.fit_ensemble(xt[residual_idx], residual_rows)
    context_models = v3.fit_ensemble(ctx_t[residual_idx], residual_rows)
    ham_models = fit_ham_ensemble(ctx_t[residual_idx], residual_rows)

    ptext_val = v3.ensemble_predict(text_models, xv)
    pctx_val = v3.ensemble_predict(context_models, ctx_v)
    pham_val = avg_prob(ham_models, ctx_v)

    ptext_test = v3.ensemble_predict(text_models, xe)
    pctx_test = v3.ensemble_predict(context_models, ctx_e)
    pham_test = avg_prob(ham_models, ctx_e)

    safe_gate, gate90 = choose_gate(va, ptext_val, pctx_val, pham_val)
    safe_pred = predict_gate(
        te, ptext_test, pctx_test, pham_test,
        safe_gate['textThreshold'],
        safe_gate['contextThreshold'],
        safe_gate['hamVetoThreshold'])
    safe_test = stats(te, safe_pred)

    target90_test = None
    if gate90 is not None:
        pred90 = predict_gate(
            te, ptext_test, pctx_test, pham_test,
            gate90['textThreshold'],
            gate90['contextThreshold'],
            gate90['hamVetoThreshold'])
        target90_test = stats(te, pred90)

    hard_mask = np.asarray([('hard_ham' in r['source']) and (not r['y']) for r in te], dtype=bool)
    easy_mask = np.asarray([('easy_ham' in r['source']) and (not r['y']) for r in te], dtype=bool)

    result = {
        'version': 'v5-precision-three-key',
        'dataset': {
            'train': len(train), 'validation': len(val), 'test': len(test),
            'testSpam': base['spamTotal'], 'testHam': base['hamTotal'],
        },
        'rspamdBayes': base,
        'safeValidationGate': safe_gate,
        'safeTest': safe_test,
        'safeTestHardHamFalsePositives': int((hard_mask & safe_pred).sum()),
        'safeTestEasyHamFalsePositives': int((easy_mask & safe_pred).sum()),
        'target90ValidationGate': gate90,
        'target90Test': target90_test,
        'review1pctResidual': review(te, ptext_test, pctx_test, pham_test),
    }

    (REPORTS / 'v5-benchmark.json').write_text(json.dumps(result, indent=2))
    md = [
        '# MailGuard v5 precision gate benchmark', '',
        f"Final untouched test: {base['spamTotal']} spam + {base['hamTotal']} ham.", '',
        '| Mode | Spam detected | Spam recall | False positives | FP rate |',
        '|---|---:|---:|---:|---:|',
        f"| Rspamd + Bayes | {base['spamDetected']}/{base['spamTotal']} | "
        f"{base['recall']:.2%} | {base['falsePositives']}/{base['hamTotal']} | {base['fpr']:.3%} |",
        f"| Rspamd + Bayes + MailGuard v5 safe | "
        f"{safe_test['spamDetected']}/{safe_test['spamTotal']} | {safe_test['recall']:.2%} | "
        f"{safe_test['falsePositives']}/{safe_test['hamTotal']} | {safe_test['fpr']:.3%} |",
        '',
        f"Hard-ham FP on final test: {result['safeTestHardHamFalsePositives']}.",
        f"Easy-ham FP on final test: {result['safeTestEasyHamFalsePositives']}.",
        '',
        '## Validation-selected 90% alternative', '',
        str(target90_test),
        '',
        '## 1% residual review', '',
        f"Random mean {result['review1pctResidual']['randomSpamFoundMean']:.2f}, "
        f"top-risk {result['review1pctResidual']['topRiskSpamFound']}, "
        f"lift {result['review1pctResidual']['lift']}.",
        '',
        '## Selected validation gate', '',
        str(safe_gate),
    ]
    text = '\n'.join(md) + '\n'
    (REPORTS / 'v5-benchmark.md').write_text(text)
    print(text, flush=True)

if __name__ == '__main__':
    main()
