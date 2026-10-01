#!/usr/bin/env python3
import csv, json, math, random, re
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_sample_weight

import benchmark_improved as b

REPORTS = Path('reports')
SEED = 1337
EXTRA_FP_BUDGET = 0.005

def split_group(files, train_fraction, val_fraction, seed):
    files = list(files)
    rnd = random.Random(seed)
    rnd.shuffle(files)
    nt = int(round(len(files) * train_fraction))
    nv = int(round(len(files) * val_fraction))
    return files[:nt], files[nt:nt+nv], files[nt+nv:]

def build_splits(groups):
    # More adaptation data from the genuinely difficult distributions.
    # The final test is still untouched and is never used to choose thresholds.
    easy_tr, easy_va, easy_te = split_group(groups['new_easy_ham'], .20, .20, SEED + 1)
    hard_tr, hard_va, hard_te = split_group(groups['new_hard_ham'], .50, .20, SEED + 2)
    spam_tr, spam_va, spam_te = split_group(groups['new_spam'], .20, .20, SEED + 3)

    train = ([{'path': p, 'y': 0, 'source': 'base_ham'} for p in groups['base_ham']] +
             [{'path': p, 'y': 1, 'source': 'base_spam'} for p in groups['base_spam']] +
             [{'path': p, 'y': 0, 'source': 'adapt_easy_ham'} for p in easy_tr] +
             [{'path': p, 'y': 0, 'source': 'adapt_hard_ham'} for p in hard_tr] +
             [{'path': p, 'y': 1, 'source': 'adapt_spam'} for p in spam_tr])

    val = ([{'path': p, 'y': 0, 'source': 'val_easy_ham'} for p in easy_va] +
           [{'path': p, 'y': 0, 'source': 'val_hard_ham'} for p in hard_va] +
           [{'path': p, 'y': 1, 'source': 'val_spam'} for p in spam_va])

    test = ([{'path': p, 'y': 0, 'source': 'test_easy_ham'} for p in easy_te] +
            [{'path': p, 'y': 0, 'source': 'test_hard_ham'} for p in hard_te] +
            [{'path': p, 'y': 1, 'source': 'test_spam'} for p in spam_te])
    return train, val, test

def pure_doc(row):
    return b.extract_document(row['raw'])

def context_doc(row):
    text = pure_doc(row)
    pseudo = ['__action_' + re.sub(r'\W+', '_', row['action'].lower())]
    for name, score in row['symbols'][:120]:
        safe = re.sub(r'\W+', '_', name.lower())
        pseudo.append('__sym_' + safe)
        if score >= 2:
            pseudo.append('__sympos_' + safe)
        elif score <= -2:
            pseudo.append('__symneg_' + safe)
    return text + '\n' + ' '.join(pseudo)

def numeric(rows):
    values = []
    for r in rows:
        req = r['required']
        ratio = r['rscore'] / req if req else 0.0
        action = r['action'].lower()
        values.append([
            max(-3.0, min(3.0, r['rscore'] / 15.0)),
            max(-3.0, min(3.0, ratio)),
            math.log1p(len(r['raw'])) / 12.0,
            1.0 if action == 'soft reject' else 0.0,
            1.0 if action == 'no action' else 0.0,
        ])
    return csr_matrix(np.asarray(values, dtype=np.float64))

def matrices(train, val, test):
    pure_tr = [pure_doc(r) for r in train]
    pure_va = [pure_doc(r) for r in val]
    pure_te = [pure_doc(r) for r in test]
    ctx_tr = [context_doc(r) for r in train]
    ctx_va = [context_doc(r) for r in val]
    ctx_te = [context_doc(r) for r in test]

    text_tr = hstack([b.WORD.transform(pure_tr), b.CHAR.transform(pure_tr)], format='csr')
    text_va = hstack([b.WORD.transform(pure_va), b.CHAR.transform(pure_va)], format='csr')
    text_te = hstack([b.WORD.transform(pure_te), b.CHAR.transform(pure_te)], format='csr')

    ctx_tr = hstack([b.WORD.transform(ctx_tr), b.CHAR.transform(ctx_tr), numeric(train)], format='csr')
    ctx_va = hstack([b.WORD.transform(ctx_va), b.CHAR.transform(ctx_va), numeric(val)], format='csr')
    ctx_te = hstack([b.WORD.transform(ctx_te), b.CHAR.transform(ctx_te), numeric(test)], format='csr')
    return text_tr, text_va, text_te, ctx_tr, ctx_va, ctx_te

def residual_weights(rows):
    y = np.asarray([r['y'] for r in rows], dtype=np.int32)
    weights = compute_sample_weight(class_weight='balanced', y=y).astype(np.float64)
    for i, r in enumerate(rows):
        if not r['y'] and 'hard_ham' in r['source']:
            weights[i] *= 6.0
        if not r['y'] and r['rscore'] >= 4.0:
            weights[i] *= 3.0
        if r['y'] and r['rscore'] <= 4.0:
            weights[i] *= 2.0
    return weights

def fit_ensemble(xtrain, rows, alphas=(1e-4, 3e-4, 1e-3)):
    y = np.asarray([r['y'] for r in rows], dtype=np.int32)
    sw = residual_weights(rows)
    models = []
    for i, alpha in enumerate(alphas):
        clf = SGDClassifier(
            loss='log_loss', penalty='l2', alpha=alpha,
            max_iter=200, tol=1e-5, random_state=SEED + i,
            average=True, fit_intercept=True,
        )
        clf.fit(xtrain, y, sample_weight=sw)
        models.append(clf)
    return models

def ensemble_predict(models, x):
    return np.mean([m.predict_proba(x)[:, 1] for m in models], axis=0)

def candidate_thresholds(p):
    base = [0.0, 0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95,
            0.97, 0.98, 0.99, 0.995, 0.999, 1.000001]
    qs = np.quantile(p, np.linspace(.50, 1.0, 70))
    return np.unique(np.concatenate([np.asarray(base), qs]))

def choose_gate(rows, ptext, pctx):
    y = np.asarray([r['y'] for r in rows], dtype=bool)
    rspam = np.asarray([r['rspam'] for r in rows], dtype=bool)
    spam_total = int(y.sum())
    ham_total = int((~y).sum())
    base_fp = int(((~y) & rspam).sum())
    max_fp = base_fp + max(1, int(math.floor(ham_total * EXTRA_FP_BUDGET)))

    ttext = candidate_thresholds(ptext)
    tctx = candidate_thresholds(pctx)
    safe_best = None
    target90 = None

    for tt in ttext:
        mt = ptext >= tt
        for tc in tctx:
            pred = rspam | (mt & (pctx >= tc))
            tp = int((y & pred).sum())
            fp = int(((~y) & pred).sum())
            recall = tp / max(1, spam_total)
            fpr = fp / max(1, ham_total)
            point = {
                'textThreshold': float(tt),
                'contextThreshold': float(tc),
                'spamDetected': tp,
                'falsePositives': fp,
                'recall': recall,
                'fpr': fpr,
                'baseFalsePositives': base_fp,
                'maxFalsePositives': max_fp,
            }
            if fp <= max_fp:
                if (safe_best is None or recall > safe_best['recall'] or
                    (recall == safe_best['recall'] and fp < safe_best['falsePositives'])):
                    safe_best = point
            if recall >= .90:
                if (target90 is None or fp < target90['falsePositives'] or
                    (fp == target90['falsePositives'] and recall > target90['recall'])):
                    target90 = point

    if safe_best is None:
        safe_best = {
            'textThreshold': 1.000001, 'contextThreshold': 1.000001,
            'spamDetected': int((y & rspam).sum()),
            'falsePositives': base_fp,
            'recall': int((y & rspam).sum()) / max(1, spam_total),
            'fpr': base_fp / max(1, ham_total),
            'baseFalsePositives': base_fp, 'maxFalsePositives': max_fp,
        }
    return safe_best, target90

def evaluate(rows, ptext, pctx, gate):
    y = np.asarray([r['y'] for r in rows], dtype=bool)
    rspam = np.asarray([r['rspam'] for r in rows], dtype=bool)
    add = ((ptext >= gate['textThreshold']) &
           (pctx >= gate['contextThreshold']))
    pred = rspam | add
    spam_total = int(y.sum())
    ham_total = int((~y).sum())
    tp = int((y & pred).sum())
    fp = int(((~y) & pred).sum())
    out = {
        'spamTotal': spam_total, 'hamTotal': ham_total,
        'spamDetected': tp, 'falsePositives': fp,
        'recall': tp / max(1, spam_total),
        'fpr': fp / max(1, ham_total),
        'textThreshold': gate['textThreshold'],
        'contextThreshold': gate['contextThreshold'],
    }
    return out, pred

def subset_metrics(rows, pred):
    result = {}
    for source in sorted(set(r['source'] for r in rows)):
        idx = np.asarray([r['source'] == source for r in rows], dtype=bool)
        y = np.asarray([r['y'] for r in rows], dtype=bool)[idx]
        pp = pred[idx]
        result[source] = {
            'count': int(idx.sum()),
            'spam': int(y.sum()),
            'ham': int((~y).sum()),
            'spamDetected': int((y & pp).sum()),
            'falsePositives': int(((~y) & pp).sum()),
        }
    return result

def review(rows, ptext, pctx):
    residual_idx = [i for i, r in enumerate(rows) if not r['rspam']]
    budget = max(1, math.ceil(len(rows) * .01))
    budget = min(budget, len(residual_idx))
    # Auto-spam needs both keys; review ranking can use either source of evidence.
    risk = np.sqrt(np.clip(ptext, 1e-9, 1) * np.clip(pctx, 1e-9, 1))
    ordered = sorted(residual_idx, key=lambda i: risk[i], reverse=True)[:budget]
    top = sum(rows[i]['y'] for i in ordered)

    rnd = random.Random(SEED)
    vals = []
    for _ in range(2000):
        sample = rnd.sample(residual_idx, budget)
        vals.append(sum(rows[i]['y'] for i in sample))
    mean = float(np.mean(vals))
    return {
        'budget': budget, 'residualCandidates': len(residual_idx),
        'topRiskSpamFound': int(top),
        'randomSpamFoundMean': mean,
        'lift': top / mean if mean else None,
    }

def run_context(name, train, val, test, with_bayes):
    print('\n===', name, '===', flush=True)
    b.reset_bayes()
    if with_bayes:
        b.learn(train)

    tr = b.scan_many(train, name + '-train')
    va = b.scan_many(val, name + '-val')
    te = b.scan_many(test, name + '-test')

    base = b.base_metrics(te)

    # The second-stage model is trained only where Rspamd did NOT already
    # make a protected spam decision. This focuses capacity on actual misses.
    residual_train_idx = [i for i, r in enumerate(tr) if not r['rspam']]
    residual_train = [tr[i] for i in residual_train_idx]

    xt, xv, xe, xc_t, xc_v, xc_e = matrices(tr, va, te)
    xt_res = xt[residual_train_idx]
    xc_t_res = xc_t[residual_train_idx]

    text_models = fit_ensemble(xt_res, residual_train)
    ctx_models = fit_ensemble(xc_t_res, residual_train)

    ptext_val = ensemble_predict(text_models, xv)
    pctx_val = ensemble_predict(ctx_models, xc_v)
    ptext_test = ensemble_predict(text_models, xe)
    pctx_test = ensemble_predict(ctx_models, xc_e)

    safe_gate, gate90 = choose_gate(va, ptext_val, pctx_val)
    safe, safe_pred = evaluate(te, ptext_test, pctx_test, safe_gate)

    target90_test = None
    target90_pred = None
    if gate90 is not None:
        target90_test, target90_pred = evaluate(te, ptext_test, pctx_test, gate90)

    return {
        'base': base,
        'safeValidationGate': safe_gate,
        'safe': safe,
        'safeSubsets': subset_metrics(te, safe_pred),
        'target90ValidationGate': gate90,
        'target90Test': target90_test,
        'target90Subsets': subset_metrics(te, target90_pred) if target90_pred is not None else None,
        'review': review(te, ptext_test, pctx_test),
    }

def main():
    REPORTS.mkdir(exist_ok=True)
    b.wait_rspamd()
    groups = b.prepare()
    train, val, test = build_splits(groups)
    print('TRAIN', len(train), 'VAL', len(val), 'TEST', len(test), flush=True)

    plain = run_context('plain-v3', train, val, test, False)
    bayes = run_context('bayes-v3', train, val, test, True)

    result = {
        'version': 'v3-residual-two-key',
        'dataset': {
            'train': len(train), 'validation': len(val), 'test': len(test),
            'testSpam': bayes['base']['spamTotal'], 'testHam': bayes['base']['hamTotal'],
        },
        'policy': {
            'residualOnlyTraining': True,
            'twoKeyAutoSpam': True,
            'hardHamWeight': 6.0,
            'highRspamdHamWeight': 3.0,
            'lowRspamdSpamWeight': 2.0,
            'incrementalValidationFpBudget': EXTRA_FP_BUDGET,
            'hardHamAdaptationFraction': 0.50,
        },
        'plain': plain,
        'bayes': bayes,
    }
    (REPORTS / 'v3-benchmark.json').write_text(json.dumps(result, indent=2))

    rows = [
        ('Rspamd', plain['base']),
        ('Rspamd + MailGuard v3 safe', plain['safe']),
        ('Rspamd + Bayes', bayes['base']),
        ('Rspamd + Bayes + MailGuard v3 safe', bayes['safe']),
    ]
    md = [
        '# MailGuard v3 residual two-key benchmark', '',
        f"Final untouched test: {result['dataset']['testSpam']} spam + "
        f"{result['dataset']['testHam']} ham.", '',
        '| Mode | Spam detected | Spam recall | False positives | FP rate |',
        '|---|---:|---:|---:|---:|',
    ]
    for label, r in rows:
        md.append(
            f"| {label} | {r['spamDetected']}/{r['spamTotal']} | "
            f"{r['recall']:.2%} | {r['falsePositives']}/{r['hamTotal']} | "
            f"{r['fpr']:.3%} |"
        )

    md += ['', '## Validation-selected 90% tradeoff', '']
    for label, item in [('Plain', plain), ('Bayes', bayes)]:
        if item['target90Test'] is None:
            md.append(f"{label}: validation could not reach 90% recall with the searched gate.")
        else:
            r = item['target90Test']
            md.append(
                f"{label}: final test {r['spamDetected']}/{r['spamTotal']} = "
                f"{r['recall']:.2%} recall, {r['falsePositives']}/{r['hamTotal']} = "
                f"{r['fpr']:.3%} FP."
            )

    md += [
        '', '## Residual 1% review', '',
        f"Plain: random {plain['review']['randomSpamFoundMean']:.2f}, "
        f"top-risk {plain['review']['topRiskSpamFound']}, lift {plain['review']['lift']}.",
        f"Bayes: random {bayes['review']['randomSpamFoundMean']:.2f}, "
        f"top-risk {bayes['review']['topRiskSpamFound']}, lift {bayes['review']['lift']}.",
        '', '## Hard-ham diagnostics', '',
        f"Bayes safe hard-ham: {bayes['safeSubsets'].get('test_hard_ham')}",
        f"Bayes 90-target hard-ham: "
        f"{bayes['target90Subsets'].get('test_hard_ham') if bayes['target90Subsets'] else None}",
    ]
    text = '\n'.join(md) + '\n'
    (REPORTS / 'v3-benchmark.md').write_text(text)
    print(text, flush=True)

if __name__ == '__main__':
    main()
