#!/usr/bin/env python3
import hashlib, json, math, random, re, statistics, subprocess, tarfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from email import policy
from email.parser import BytesParser
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_sample_weight

BASE = 'https://spamassassin.apache.org/old/publiccorpus/'
CACHE = Path('.cache/spamassassin-improved')
REPORTS = Path('reports')
RSPAMD = 'http://127.0.0.1:11333/'
CTRL = 'http://127.0.0.1:11334/'
SEED = 1337
EXTRA_FP_BUDGET = 0.005
SPAM_ACTIONS = {'reject', 'add header', 'rewrite subject', 'quarantine', 'discard'}
ARCHIVES = {
    'base_ham': '20030228_easy_ham.tar.bz2',
    'base_spam': '20030228_spam.tar.bz2',
    'new_easy_ham': '20030228_easy_ham_2.tar.bz2',
    'new_hard_ham': '20030228_hard_ham.tar.bz2',
    'new_spam': '20050311_spam_2.tar.bz2',
}

def get(url, timeout=90):
    req = Request(url, headers={'User-Agent': 'MailGuard-improved-benchmark/1.0'})
    return urlopen(req, timeout=timeout).read()

def post(url, data, timeout=30):
    req = Request(
        url, data=data, method='POST',
        headers={'Content-Type': 'message/rfc822',
                 'User-Agent': 'MailGuard-improved-benchmark/1.0'})
    return urlopen(req, timeout=timeout).read()

def wait_rspamd():
    for _ in range(90):
        try:
            get(RSPAMD + 'ping', 5)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError('Rspamd did not become ready')

def reset_bayes():
    subprocess.run(
        ['docker', 'compose', 'exec', '-T', 'redis', 'redis-cli', 'FLUSHALL'],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    subprocess.run(
        ['docker', 'compose', 'restart', 'rspamd'],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wait_rspamd()

def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    groups = {}
    for key, name in ARCHIVES.items():
        arc = CACHE / name
        if not arc.exists():
            print('download', name, flush=True)
            arc.write_bytes(get(BASE + name, 120))
        dest = CACHE / (name + '.dir')
        if not dest.exists():
            dest.mkdir(parents=True)
            with tarfile.open(arc, 'r:bz2') as tf:
                tf.extractall(dest)
        files = [
            p for p in dest.rglob('*')
            if p.is_file() and p.name != 'cmds' and not p.name.startswith('.')
        ]
        files.sort()
        groups[key] = files
        print(key, len(files), flush=True)
    return groups

def stable_split(files, adapt_fraction=0.10, val_fraction=0.10):
    files = list(files)
    rnd = random.Random(SEED + len(files))
    rnd.shuffle(files)
    na = int(round(len(files) * adapt_fraction))
    nv = int(round(len(files) * val_fraction))
    return files[:na], files[na:na+nv], files[na+nv:]

def dedupe(parts):
    seen = set()
    out = []
    dropped = 0
    for item in parts:
        raw = item['path'].read_bytes()
        sig = hashlib.sha256(raw).digest()
        if sig in seen:
            dropped += 1
            continue
        seen.add(sig)
        out.append(item)
    return out, dropped

def build_splits(groups):
    easy_a, easy_v, easy_t = stable_split(groups['new_easy_ham'])
    hard_a, hard_v, hard_t = stable_split(groups['new_hard_ham'])
    spam_a, spam_v, spam_t = stable_split(groups['new_spam'])

    base = ([{'path': p, 'y': 0, 'source': 'base_ham'} for p in groups['base_ham']] +
            [{'path': p, 'y': 1, 'source': 'base_spam'} for p in groups['base_spam']])

    adapt = ([{'path': p, 'y': 0, 'source': 'adapt_easy_ham'} for p in easy_a] +
             [{'path': p, 'y': 0, 'source': 'adapt_hard_ham'} for p in hard_a] +
             [{'path': p, 'y': 1, 'source': 'adapt_spam'} for p in spam_a])

    val = ([{'path': p, 'y': 0, 'source': 'val_easy_ham'} for p in easy_v] +
           [{'path': p, 'y': 0, 'source': 'val_hard_ham'} for p in hard_v] +
           [{'path': p, 'y': 1, 'source': 'val_spam'} for p in spam_v])

    test = ([{'path': p, 'y': 0, 'source': 'test_easy_ham'} for p in easy_t] +
            [{'path': p, 'y': 0, 'source': 'test_hard_ham'} for p in hard_t] +
            [{'path': p, 'y': 1, 'source': 'test_spam'} for p in spam_t])

    train, d1 = dedupe(base + adapt)
    val, d2 = dedupe(train + val)
    val = val[len(train):]
    test_all, d3 = dedupe(train + val + test)
    test = test_all[len(train) + len(val):]
    return train, val, test, d1 + d2 + d3

def learn(items):
    for label, value, endpoint in [('ham', 0, 'learnham'), ('spam', 1, 'learnspam')]:
        files = [x['path'] for x in items if x['y'] == value]
        for i, p in enumerate(files, 1):
            post(CTRL + endpoint, p.read_bytes(), 30)
            if i % 250 == 0 or i == len(files):
                print('learn', label, i, '/', len(files), flush=True)

def scan_one(item):
    raw = item['path'].read_bytes()
    j = json.loads(post(RSPAMD + 'checkv2', raw, 30))
    symbols = []
    for name, value in (j.get('symbols') or {}).items():
        if isinstance(value, dict):
            symbols.append((name, float(value.get('score', 0) or 0)))
    action = str(j.get('action', '')).lower()
    return {
        **item,
        'raw': raw,
        'action': action,
        'rscore': float(j.get('score', 0) or 0),
        'required': float(j.get('required_score', 0) or 0),
        'symbols': symbols,
        'rspam': action in SPAM_ACTIONS,
    }

def scan_many(items, label, workers=8):
    print('scan', label, len(items), flush=True)
    out = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(scan_one, item): i for i, item in enumerate(items)}
        done = 0
        for future in as_completed(futures):
            out[futures[future]] = future.result()
            done += 1
            if done % 250 == 0 or done == len(items):
                print(label, done, '/', len(items), flush=True)
    return out

def extract_document(raw):
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        chunks = [
            'subject ' + str(msg.get('subject', '')),
            'from ' + str(msg.get('from', '')),
            'to ' + str(msg.get('to', '')),
        ]
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_maintype() == 'text':
                    try:
                        chunks.append(str(part.get_content()))
                    except Exception:
                        pass
        else:
            try:
                chunks.append(str(msg.get_content()))
            except Exception:
                pass
        text = '\n'.join(chunks)
        if len(text) > 250000:
            text = text[:250000]
        return text
    except Exception:
        return raw.decode('utf-8', 'ignore')[:250000]

def model_document(row):
    text = extract_document(row['raw'])
    pseudo = [
        '__rspamd_action_' + re.sub(r'\W+', '_', row['action']),
    ]
    for name, score in row['symbols'][:120]:
        safe = re.sub(r'\W+', '_', name.lower())
        pseudo.append('__rspamd_symbol_' + safe)
        if score >= 2:
            pseudo.append('__rspamd_strong_' + safe)
    return text + '\n' + ' '.join(pseudo)

WORD = HashingVectorizer(
    n_features=1 << 17,
    alternate_sign=False,
    norm='l2',
    lowercase=True,
    ngram_range=(1, 2),
    token_pattern=r'(?u)\b[\w@.\-]{2,}\b',
)
CHAR = HashingVectorizer(
    n_features=1 << 17,
    alternate_sign=False,
    norm='l2',
    lowercase=True,
    analyzer='char_wb',
    ngram_range=(3, 5),
)

def vectorize(train, val, test):
    docs_train = [model_document(r) for r in train]
    docs_val = [model_document(r) for r in val]
    docs_test = [model_document(r) for r in test]

    def numeric(rows):
        data = []
        for r in rows:
            req = r['required']
            ratio = r['rscore'] / req if req else 0.0
            data.append([
                max(-3.0, min(3.0, r['rscore'] / 15.0)),
                max(-3.0, min(3.0, ratio)),
                math.log1p(len(r['raw'])) / 12.0,
                1.0 if '<html' in extract_document(r['raw']).lower() else 0.0,
            ])
        return csr_matrix(np.asarray(data, dtype=np.float64))

    xtr = hstack([WORD.transform(docs_train), CHAR.transform(docs_train), numeric(train)],
                 format='csr')
    xva = hstack([WORD.transform(docs_val), CHAR.transform(docs_val), numeric(val)],
                 format='csr')
    xte = hstack([WORD.transform(docs_test), CHAR.transform(docs_test), numeric(test)],
                 format='csr')
    return xtr, xva, xte

def base_metrics(rows):
    spam = sum(r['y'] for r in rows)
    ham = len(rows) - spam
    tp = sum(1 for r in rows if r['y'] and r['rspam'])
    fp = sum(1 for r in rows if not r['y'] and r['rspam'])
    return {
        'spamTotal': spam, 'hamTotal': ham,
        'spamDetected': tp, 'falsePositives': fp,
        'recall': tp / max(1, spam), 'fpr': fp / max(1, ham),
    }

def choose_threshold(rows, probs, extra_fp_budget=EXTRA_FP_BUDGET):
    spam_total = sum(r['y'] for r in rows)
    ham_total = len(rows) - spam_total
    base_fp = sum(1 for r in rows if not r['y'] and r['rspam'])
    max_fp = base_fp + max(1, int(math.floor(ham_total * extra_fp_budget)))

    candidates = np.unique(np.concatenate(([1.000001], probs)))
    candidates.sort()
    best = None
    ninety = None

    for t in candidates[::-1]:
        tp = fp = 0
        for row, p in zip(rows, probs):
            pred = row['rspam'] or p >= t
            if row['y'] and pred:
                tp += 1
            elif not row['y'] and pred:
                fp += 1
        recall = tp / max(1, spam_total)
        fpr = fp / max(1, ham_total)
        point = {
            'threshold': float(t), 'spamDetected': tp, 'falsePositives': fp,
            'recall': recall, 'fpr': fpr, 'baseFalsePositives': base_fp,
            'maxFalsePositives': max_fp,
        }
        if fp <= max_fp:
            if (best is None or recall > best['recall'] or
                (recall == best['recall'] and fp < best['falsePositives'])):
                best = point
        if recall >= 0.90:
            if ninety is None or fp < ninety['falsePositives']:
                ninety = point
    return best, ninety

def fit_and_select(train, val, test):
    xtr, xva, xte = vectorize(train, val, test)
    ytr = np.asarray([r['y'] for r in train], dtype=np.int32)
    sample_weight = compute_sample_weight(class_weight='balanced', y=ytr)

    results = []
    for alpha in [1e-4, 3e-4, 1e-3, 3e-3]:
        clf = SGDClassifier(
            loss='log_loss', penalty='l2', alpha=alpha,
            max_iter=150, tol=1e-4, random_state=SEED,
            average=True, fit_intercept=True,
        )
        clf.fit(xtr, ytr, sample_weight=sample_weight)
        pva = clf.predict_proba(xva)[:, 1]
        best, ninety = choose_threshold(val, pva)
        results.append((alpha, clf, best, ninety))
        print('alpha', alpha, 'best', best, '90point', ninety, flush=True)

    feasible = [x for x in results if x[2] is not None]
    chosen = max(
        feasible,
        key=lambda x: (x[2]['recall'], -x[2]['falsePositives'], -x[0])
    )
    alpha, clf, chosen_point, ninety_point = chosen
    pte = clf.predict_proba(xte)[:, 1]
    return alpha, chosen_point, ninety_point, pte

def hybrid_metrics(rows, probs, threshold):
    spam = sum(r['y'] for r in rows)
    ham = len(rows) - spam
    tp = fp = 0
    prediction_rows = []
    for row, p in zip(rows, probs):
        pred = row['rspam'] or p >= threshold
        if row['y'] and pred:
            tp += 1
        elif not row['y'] and pred:
            fp += 1
        prediction_rows.append({
            'source': row['source'],
            'label': 'spam' if row['y'] else 'ham',
            'rspamdSpam': row['rspam'],
            'probability': float(p),
            'hybridSpam': bool(pred),
            'rspamdScore': row['rscore'],
            'action': row['action'],
            'path': str(row['path']),
        })
    return {
        'spamTotal': spam, 'hamTotal': ham,
        'spamDetected': tp, 'falsePositives': fp,
        'recall': tp / max(1, spam), 'fpr': fp / max(1, ham),
        'threshold': float(threshold),
    }, prediction_rows

def review_benchmark(predictions, percent=1.0, runs=2000):
    residual = [r for r in predictions if not r['rspamdSpam']]
    total = len(predictions)
    k = max(1, math.ceil(total * percent / 100.0))
    k = min(k, len(residual))
    top = sorted(residual, key=lambda r: r['probability'], reverse=True)[:k]
    top_spam = sum(r['label'] == 'spam' for r in top)

    rnd = random.Random(SEED)
    counts = []
    for _ in range(runs):
        sample = rnd.sample(residual, k)
        counts.append(sum(r['label'] == 'spam' for r in sample))
    mean = statistics.mean(counts)
    return {
        'residualCandidates': len(residual),
        'budgetFromWholeStream': k,
        'topRiskSpamFound': top_spam,
        'randomSpamFoundMean': mean,
        'lift': top_spam / mean if mean else None,
    }

def run_context(name, train, val, test, with_bayes):
    print('\n===', name, '===', flush=True)
    reset_bayes()
    if with_bayes:
        learn(train)

    tr = scan_many(train, name + '-train')
    va = scan_many(val, name + '-val')
    te = scan_many(test, name + '-test')

    base = base_metrics(te)
    alpha, selected, ninety, ptest = fit_and_select(tr, va, te)
    hybrid, predictions = hybrid_metrics(te, ptest, selected['threshold'])
    review = review_benchmark(predictions)

    return {
        'base': base,
        'hybrid': hybrid,
        'alpha': alpha,
        'validationSelected': selected,
        'validation90Point': ninety,
        'review1pctResidual': review,
        'predictions': predictions,
    }

def write_predictions(path, predictions):
    import csv
    fields = ['source','label','rspamdSpam','probability','hybridSpam',
              'rspamdScore','action','path']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)

def main():
    REPORTS.mkdir(exist_ok=True)
    wait_rspamd()
    groups = prepare()
    train, val, test, duplicate_count = build_splits(groups)
    print('train', len(train), 'validation', len(val), 'final test', len(test),
          'duplicates dropped', duplicate_count, flush=True)

    plain = run_context('plain-rspamd', train, val, test, with_bayes=False)
    bayes = run_context('rspamd-bayes', train, val, test, with_bayes=True)

    result = {
        'policy': {
            'baseTrainingCorpus': '20030228_easy_ham + 20030228_spam',
            'adaptationShareOfNewCorpora': 0.10,
            'validationShareOfNewCorpora': 0.10,
            'finalTestShareOfNewCorpora': 0.80,
            'extraAutoSpamFpBudget': EXTRA_FP_BUDGET,
            'exactDuplicatesDropped': duplicate_count,
        },
        'dataset': {
            'train': len(train), 'validation': len(val), 'test': len(test),
            'testSpam': plain['base']['spamTotal'],
            'testHam': plain['base']['hamTotal'],
        },
        'rspamd': plain['base'],
        'rspamdPlusMailGuard': plain['hybrid'],
        'rspamdBayes': bayes['base'],
        'rspamdBayesPlusMailGuard': bayes['hybrid'],
        'plainModel': {
            'alpha': plain['alpha'],
            'validationSelected': plain['validationSelected'],
            'validation90Point': plain['validation90Point'],
            'review1pctResidual': plain['review1pctResidual'],
        },
        'bayesModel': {
            'alpha': bayes['alpha'],
            'validationSelected': bayes['validationSelected'],
            'validation90Point': bayes['validation90Point'],
            'review1pctResidual': bayes['review1pctResidual'],
        },
    }

    (REPORTS / 'improved-benchmark.json').write_text(json.dumps(result, indent=2))
    write_predictions(REPORTS / 'improved-plain-predictions.csv', plain['predictions'])
    write_predictions(REPORTS / 'improved-bayes-predictions.csv', bayes['predictions'])

    rows = [
        ('Rspamd', result['rspamd']),
        ('Rspamd + MailGuard v2', result['rspamdPlusMailGuard']),
        ('Rspamd + Bayes', result['rspamdBayes']),
        ('Rspamd + Bayes + MailGuard v2', result['rspamdBayesPlusMailGuard']),
    ]
    md = [
        '# MailGuard conservative v2 benchmark',
        '',
        f"Final untouched test: {result['dataset']['testSpam']} spam + "
        f"{result['dataset']['testHam']} ham.",
        '',
        'MailGuard is allowed to add at most 0.5 percentage points of false '
        'positives on validation above the protected Rspamd baseline.',
        '',
        '| Mode | Spam detected | Spam recall | False positives | FP rate |',
        '|---|---:|---:|---:|---:|',
    ]
    for label, r in rows:
        md.append(
            f"| {label} | {r['spamDetected']}/{r['spamTotal']} | "
            f"{r['recall']:.2%} | {r['falsePositives']}/{r['hamTotal']} | "
            f"{r['fpr']:.3%} |"
        )
    md += [
        '',
        '## Residual 1% human-review ranking',
        '',
        f"Plain Rspamd residual: random mean "
        f"{plain['review1pctResidual']['randomSpamFoundMean']:.2f}, "
        f"top-risk {plain['review1pctResidual']['topRiskSpamFound']}, "
        f"lift {plain['review1pctResidual']['lift']}.",
        f"Bayes residual: random mean "
        f"{bayes['review1pctResidual']['randomSpamFoundMean']:.2f}, "
        f"top-risk {bayes['review1pctResidual']['topRiskSpamFound']}, "
        f"lift {bayes['review1pctResidual']['lift']}.",
        '',
        '## Validation diagnostics',
        '',
        f"Plain selected: {plain['validationSelected']}",
        f"Plain minimum-FP point reaching 90% recall: {plain['validation90Point']}",
        f"Bayes selected: {bayes['validationSelected']}",
        f"Bayes minimum-FP point reaching 90% recall: {bayes['validation90Point']}",
    ]
    text = '\n'.join(md) + '\n'
    (REPORTS / 'improved-benchmark.md').write_text(text)
    print(text, flush=True)

if __name__ == '__main__':
    main()
