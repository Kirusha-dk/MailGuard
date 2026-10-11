#!/usr/bin/env python3
"""Streaming multilingual learning with campaign-group split isolation.

The 500k corpus has already been inspected in v54; tests here are engineering
holdouts for v55, not a new independent benchmark or a production guarantee.
"""
import argparse
import csv
import gc
import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize

from benchmark_v48_calibrated import threshold_for_budget
from rspamd_benchmark_audit import load_frozen_v40
from train_v47_phishing_aware import from_eml, metrics_at

URL = re.compile(r'https?://[^\s<>"\'\u3000-\u30ff\u3400-\u9fff]+', re.I)
EMAIL = re.compile(r'[\w.+-]+@[\w.-]+\.[\w-]+', re.U)
CJK = r'\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af'
CJK_NOISE = re.compile(r'(?<=[' + CJK + r'])[\s\W_]+(?=[' + CJK + r'])')
CSS = re.compile(r'/\*.*?\*/|(?:[.#][\w.:[\]# >,+-]{1,80}|\b(?:html|body|p|a|td|table|span|div|h[1-6]))\s*\{[^{}]{0,4000}\}', re.S)
FEATURES = 2**18 + 2**18 + 2**17
CORPUS_SHA = '5c0993c0258b854ce9cf2d5a31172d56692aaa351a533ca57c1ecc6e97c81f82'
WARNING = ('Previously inspected public corpora: engineering holdouts, not an independent production benchmark. '
           'No test rows fit the estimator, IDF, NB ratio, model selection or threshold. '
           'Campaign-family fingerprints isolate splits, but semantic relatives may remain. '
           'Legitimate multilingual mail is underrepresented; low FP on these ham sources does not establish multilingual safety.')


def sha(text):
    return hashlib.sha256(text.encode('utf-8', 'replace')).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fields(text):
    headers, sep, body = text.partition('\nBODY: ')
    if not sep:
        raise ValueError('Canonical body marker missing')
    subject = headers.split('\n', 1)[0].removeprefix('SUBJECT: ')
    match = re.search(r'^URL_HOSTS: (.*)$', headers, re.M)
    return subject, body, match.group(1) if match else ''


def clean(value):
    value = unicodedata.normalize('NFKC', value).casefold()
    value = re.sub('[\u200b-\u200f\ufeff]', '', value)
    value = CSS.sub(' ', value)
    value = EMAIL.sub(' emailtoken ', value)
    value = re.sub(r'\d+', ' numbertoken ', value)
    # Repeated substitutions also remove runs of punctuation separating CJK letters.
    value = CJK_NOISE.sub('', value)
    return re.sub(r'\s+', ' ', value).strip()


def campaign_key(text):
    _, body, _ = fields(text)
    # Ignore tracking hosts, addresses, numbers and obfuscating punctuation.
    # Prefix grouping is deliberately conservative: common boilerplate can merge.
    body = URL.sub(' ', unicodedata.normalize('NFKC', body[:6000]).casefold())
    body = EMAIL.sub(' ', body)
    body = CSS.sub(' ', body)
    body = re.sub(r'[\W_\d]+', '', body, flags=re.U)
    if len(body) < 20:
        # Tiny messages still get stable grouping without collapsing all URL-only mail.
        subject, _, _ = fields(text)
        body += re.sub(r'[\W_\d]+', '', unicodedata.normalize('NFKC', subject).casefold())
    return sha('v55-family:' + body[:300])


def archive_split(key):
    bucket = int(key[:12], 16) % 100
    return 'train' if bucket < 20 else 'select' if bucket < 25 else 'cal' if bucket < 30 else 'test'


def views(text):
    subject, body, hosts = fields(text)
    # Detect links even if the URL immediately follows a Japanese/CJK character.
    links = URL.findall(body[:12000])
    discovered = []
    for url in links:
        try:
            host = urlsplit(url).hostname
            if host:
                discovered.append(host.lower())
        except ValueError:
            continue
    hosts = ' '.join(sorted(set(hosts.split() + discovered))[:24])
    # Discard variable URL paths and recipients; never use source/year or envelope domain.
    body = URL.sub(' urltoken ', body[:6000]) + ' ' + URL.sub(' urltoken ', body[-1200:])
    subject = clean(URL.sub(' urltoken ', subject))[:500]
    body = clean(body)
    focus = f'subject {subject} hosts {hosts} start {body[:1000]}'
    context = f'subject {subject} body {body[:2400]} tail {body[-600:]}'
    return focus, context


def features(rows):
    focus = [r['focus'] for r in rows]
    context = [r['context'] for r in rows]
    def char(texts, bounds):
        return HashingVectorizer(analyzer='char', ngram_range=bounds, n_features=2**18,
                                 alternate_sign=False, norm=None, dtype=np.float32).transform(texts)
    return hstack([char(focus, (2, 5)), char(context, (3, 5)),
                   HashingVectorizer(ngram_range=(1, 2), n_features=2**17,
                                     alternate_sign=False, norm=None, dtype=np.float32).transform(context)], format='csr')


def transformed(raw, idf, ratio=None):
    x = raw.copy()
    x.data = np.log(x.data) + 1.
    x = x.multiply(idf).tocsr()
    if ratio is not None:
        x = x.multiply(ratio).tocsr()
    return normalize(x, norm='l2', copy=False)


def compact(row, family):
    focus, context = views(row['ngram_text'])
    return dict(identity=row['identity'], family=family, focus=focus, context=context,
                y=int(row['y']), source=row['source'])


def corpus_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def prepare(args, out):
    corpus = Path(args.corpus)
    if file_sha(corpus) != CORPUS_SHA:
        raise ValueError('Frozen v54 corpus hash mismatch')
    frozen, audit = load_frozen_v40('.cache/v40-fresh50k')
    legacy, claimed = {k: [] for k in ('train', 'select', 'cal', 'test')}, {}
    legacy_labels, legacy_conflicts = {}, set()
    excluded = Counter()
    # Reserve legacy test first, then validation, so TRAIN cannot absorb a related holdout.
    for split in ('test', 'val', 'train'):
        for old in frozen[split]:
            row = dict(from_eml(old['path'].read_bytes()), y=int(old['y']), source=old['source'])
            family = campaign_key(row['ngram_text'])
            if family in claimed:
                if legacy_labels[family] != row['y']:
                    legacy_conflicts.add(family)
                excluded['legacyFamilyDuplicate'] += 1
                continue
            target = 'test' if split == 'test' else 'train' if split == 'train' else 'select' if int(family[:8], 16) % 2 == 0 else 'cal'
            claimed[family] = target
            legacy_labels[family] = row['y']
            legacy[target].append(compact(row, family))
    excluded['conflictingLegacyFamilies'] = len(legacy_conflicts)
    for split in legacy:
        legacy[split] = [r for r in legacy[split] if r['family'] not in legacy_conflicts]
    # Conflict discovery precedes selection. Do not turn label-conflicting families into training examples.
    families, conflicts = {}, set()
    for row in corpus_rows(corpus):
        family = campaign_key(row['ngram_text'])
        if family in claimed:
            excluded['archiveRelatedToLegacy'] += 1
            continue
        y = int(row['y'])
        if family in families and families[family] != y:
            conflicts.add(family)
        families[family] = y
    excluded['conflictingArchiveFamilies'] = len(conflicts)
    legacy_families = set(claimed)
    selected = {k: list(v) for k, v in legacy.items() if k != 'test'}
    # Smoke uses a fixed content order and caps each split, retaining both classes if available.
    if args.limit:
        for split in selected:
            selected[split].sort(key=lambda r: r['identity'])
            selected[split] = selected[split][:args.limit // 2]
        legacy['test'].sort(key=lambda r: r['identity'])
        legacy['test'] = legacy['test'][:args.limit // 2]
    seen, test_count = set(), 0
    new_counts = Counter()
    new_limit = args.limit - args.limit // 2
    test_path = out / 'test.jsonl.gz'
    with gzip.open(test_path, 'wt', encoding='utf-8') as test, (out / 'split-manifest.csv').open('w', newline='') as mf:
        writer = csv.writer(mf)
        writer.writerow(['identity', 'family', 'split', 'label', 'source'])
        for split, rows in selected.items():
            for r in rows:
                writer.writerow([r['identity'], r['family'], split, r['y'], r['source']])
        for r in legacy['test']:
            test.write(json.dumps(r, ensure_ascii=False) + '\n')
            writer.writerow([r['identity'], r['family'], 'test', r['y'], r['source']])
        for row in corpus_rows(corpus):
            family = campaign_key(row['ngram_text'])
            if family in legacy_families or family in conflicts:
                continue
            split = archive_split(family)
            # One representative per new family in training/validation; all members stay in its test split.
            if split != 'test' and family in seen:
                excluded['archiveTrainingFamilyDuplicate'] += 1
                continue
            if split != 'test':
                seen.add(family)
                if args.limit and new_counts[split] >= new_limit:
                    continue
            elif args.limit and test_count >= new_limit:
                continue
            r = compact(row, family)
            if split == 'test':
                test.write(json.dumps(r, ensure_ascii=False) + '\n')
                test_count += 1
            else:
                selected[split].append(r)
                new_counts[split] += 1
            writer.writerow([r['identity'], family, split, r['y'], r['source']])
    train_keys = {r['family'] for r in selected['train']}
    select_keys = {r['family'] for r in selected['select']}
    cal_keys = {r['family'] for r in selected['cal']}
    test_keys = {r['family'] for r in corpus_rows(test_path)}
    all_keys = [train_keys, select_keys, cal_keys, test_keys]
    if any(all_keys[i] & all_keys[j] for i in range(4) for j in range(i + 1, 4)):
        raise AssertionError('Campaign-family split leakage')
    # Fixed shuffle breaks any label/source order without changing split membership.
    rng = np.random.default_rng(55)
    for split, rows in selected.items():
        if {r['y'] for r in rows} != {0, 1}:
            raise ValueError(f'{split} requires both classes')
        rng.shuffle(rows)
    counts = {k: dict(Counter(r['y'] for r in rows)) for k, rows in selected.items()}
    counts['test'] = dict(Counter(r['y'] for r in corpus_rows(test_path)))
    split_report = dict(counts=counts, legacyAudit=audit, exclusions=dict(excluded),
                        familyOverlap=0, corpusSha256=CORPUS_SHA, archiveSplit='family hash: 20% TRAIN, 5% selection, 5% calibration, 70% test',
                        warning=WARNING, smokeLimitPerSourceSplit=args.limit)
    (out / 'split-report.json').write_text(json.dumps(split_report, indent=2))
    print('Split inventory:', json.dumps(counts), flush=True)
    del frozen, legacy, families, claimed, seen, test_keys
    gc.collect()
    return selected, test_path, split_report


def predict(model, rows, idf, ratio):
    return model.predict_proba(transformed(features(rows), idf, ratio))[:, 1]


def predict_texts(artifact, texts):
    if artifact.get('version') != 'v55-multilingual-stream':
        raise ValueError('Expected a v55 artifact')
    rows = []
    for text in texts:
        focus, context = views(text)
        rows.append({'focus': focus, 'context': context})
    return predict(artifact['model'], rows, artifact['idf'], artifact['ratio'])


def fp_budget(labels, rate):
    # Correct denominator: labelled HAM, not all messages or a spam-heavy corpus size.
    return int(np.count_nonzero(np.asarray(labels) == 0) * rate)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=1000)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--corpus', default='.cache/v54-corpus/selected.jsonl.gz')
    p.add_argument('--output', default='reports/v55-smoke')
    p.add_argument('--dev-fpr', type=float, default=.0002)
    p.add_argument('--baseline-predictions', default='.cache/v54-evidence/predictions.csv')
    args = p.parse_args()
    if args.limit < 0 or args.epochs < 1 or not 0 <= args.dev_fpr < 1:
        p.error('Invalid limits or ham FP rate')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    selected, test_path, split_report = prepare(args, out)
    train, selection, cal = (selected[k] for k in ('train', 'select', 'cal'))
    # Fit IDF and NB ratio exclusively on TRAIN, in bounded sparse batches.
    df = np.zeros(FEATURES, dtype=np.float64)
    class_totals = [np.zeros(FEATURES, dtype=np.float64) for _ in range(2)]
    for start in range(0, len(train), 256):
        batch = train[start:start + 256]
        raw = features(batch)
        df += np.bincount(raw.indices, minlength=FEATURES)
        y = np.asarray([r['y'] for r in batch])
        for label in (0, 1):
            class_totals[label] += np.asarray(raw[y == label].sum(axis=0)).ravel()
    idf = (np.log((1 + len(train)) / (1 + df)) + 1).astype(np.float32)
    positive, negative = class_totals[1] + 1, class_totals[0] + 1
    nb = np.log((positive / positive.sum()) / (negative / negative.sum())).astype(np.float32)
    del df, class_totals, positive, negative
    candidates = []
    for rep in ('tfidf', 'nb-tfidf'):
        for alpha in (1e-5, 3e-6):
            model = SGDClassifier(loss='log_loss', alpha=alpha, average=True, random_state=55)
            candidates.append(dict(name=f'{rep}-alpha{alpha}', model=model, ratio=nb if rep == 'nb-tfidf' else None))
    rng = np.random.default_rng(55)
    for epoch in range(args.epochs):
        order = rng.permutation(len(train))
        for start in range(0, len(train), 256):
            batch = [train[i] for i in order[start:start + 256]]
            raw = features(batch)
            y = np.asarray([r['y'] for r in batch])
            for rep in ('tfidf', 'nb-tfidf'):
                x = transformed(raw, idf, nb if rep == 'nb-tfidf' else None)
                for c in candidates:
                    if c['name'].startswith(rep + '-'):
                        c['model'].partial_fit(x, y, classes=np.asarray([0, 1]), sample_weight=np.where(y == 0, 4., 1.))
        print(f'v55 TRAIN epoch {epoch + 1}/{args.epochs}: {len(train)} rows', flush=True)
    ys = np.asarray([r['y'] for r in selection]); sources = [r['source'] for r in selection]
    results = []
    for c in candidates:
        scores = np.concatenate([predict(c['model'], selection[i:i + 256], idf, c['ratio']) for i in range(0, len(selection), 256)])
        threshold = threshold_for_budget(ys, scores, fp_budget(ys, args.dev_fpr))
        metric = metrics_at(ys, scores, threshold, sources)
        new_mask = np.asarray([s.startswith(('SpamArchive-', 'CSDMC')) for s in sources])
        new_metric = metrics_at(ys[new_mask], scores[new_mask], threshold, list(np.asarray(sources)[new_mask]))
        old_mask = ~new_mask
        old_metric = metrics_at(ys[old_mask], scores[old_mask], threshold, list(np.asarray(sources)[old_mask]))
        # Give new and legacy sources equal influence, independent of their size.
        key = ((new_metric['recall'] + old_metric['recall']) / 2., -metric['fp'])
        c.update(selection=metric, newSelection=new_metric, oldSelection=old_metric, key=key)
        results.append({k: c[k] for k in ('name', 'selection', 'newSelection', 'oldSelection')})
        print('v55 selection', c['name'], json.dumps({'recall':metric['recall'],'newRecall':new_metric['recall'],'oldRecall':old_metric['recall'],'fp':metric['fp']}), flush=True)
    chosen = max(candidates, key=lambda c: c['key'])
    yc = np.asarray([r['y'] for r in cal])
    pc = np.concatenate([predict(chosen['model'], cal[i:i + 256], idf, chosen['ratio']) for i in range(0, len(cal), 256)])
    threshold = threshold_for_budget(yc, pc, fp_budget(yc, args.dev_fpr))
    artifact = dict(version='v55-multilingual-stream', model=chosen['model'], idf=idf, ratio=chosen['ratio'], threshold=threshold,
                    representation='v55 subject/head + compact body; char 2-5/3-5 and word 1-2', devHamFpr=args.dev_fpr,
                    campaignFamilySplit=True, corpusSha256=CORPUS_SHA)
    joblib.dump(artifact, out / 'model.joblib', compress=3)
    model_hash = file_sha(out / 'model.joblib')
    del candidates, train, selection, cal, selected
    gc.collect()
    with Path(args.baseline_predictions).open() as f:
        baseline_index = {r['identity']: (int(r['label']), int(r['predictedSpam'])) for r in csv.DictReader(f)}
    baseline_y, baseline_p, baseline_sources = [], [], []
    ys_test, ps_test, test_sources = [], [], []
    with (out / 'predictions.csv').open('w', newline='') as pf:
        writer = csv.writer(pf); writer.writerow(['identity', 'family', 'source', 'label', 'score', 'predictedSpam'])
        def consume(batch):
            scores = predict(artifact['model'], batch, artifact['idf'], artifact['ratio'])
            for r, score in zip(batch, scores):
                writer.writerow([r['identity'],r['family'],r['source'],r['y'],float(score),int(score >= threshold)])
                ys_test.append(r['y']); ps_test.append(float(score)); test_sources.append(r['source'])
                if r['identity'] in baseline_index:
                    label, b = baseline_index[r['identity']]
                    if label != r['y']:
                        raise ValueError('Baseline label mismatch')
                    baseline_y.append(label); baseline_p.append(b); baseline_sources.append(r['source'])
            if len(ys_test) % 10000 == 0:
                print('v55 holdout scored', len(ys_test), flush=True)
        batch = []
        for r in corpus_rows(test_path):
            batch.append(r)
            if len(batch) == 256:
                consume(batch); batch = []
        if batch: consume(batch)
    if model_hash != file_sha(out / 'model.joblib'):
        raise AssertionError('Estimator changed after calibration')
    y, scores = np.asarray(ys_test), np.asarray(ps_test)
    mask = np.asarray([s.startswith(('SpamArchive-', 'CSDMC')) for s in test_sources])
    new_metric = metrics_at(y[mask], scores[mask], threshold, list(np.asarray(test_sources)[mask]))
    old_metric = metrics_at(y[~mask], scores[~mask], threshold, list(np.asarray(test_sources)[~mask]))
    metric = metrics_at(y, scores, threshold, test_sources)
    baseline_metric = metrics_at(np.asarray(baseline_y), np.asarray(baseline_p), .5, baseline_sources)
    if len(baseline_y) != int(mask.sum()):
        raise AssertionError('Baseline and v55 new holdout differ')
    report = dict(version=55, selected=chosen['name'], candidates=results, calibration=metrics_at(yc, pc, threshold, ['calibration'] * len(yc)),
                  test=metric, newArchiveHoldout=new_metric, legacyHoldout=old_metric, testCount=len(y),
                  modelSha256=model_hash, modelRefitAfterCalibration=False, testLabelsUsedForEstimatorOrThresholdFit=False, testCorpusUsedForPriorErrorAnalysis=True,
                  v54OnSameNewHoldout=baseline_metric, developmentCorpusPreviouslyInspected=True, productionGoalValidated=False, splits=split_report, warning=WARNING)
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    def metric_row(name, m):
        return f'| {name} | {m["tp"]+m["fn"]+m["fp"]+m["tn"]} | {m["fp"]+m["tn"]} | {m["recall"]:.2%} | {m["fp"]} | {m["fn"]} | {m["fpr"]:.4%} |\n'
    md = ('# MailGuard v55 multilingual engineering holdouts\n\n' + WARNING + '\n\n'
          '| Split | N | Ham | Spam recall | FP | FN | FP / ham |\n|---|---:|---:|---:|---:|---:|---:|\n'
          + metric_row('v54: same new holdout', baseline_metric) + metric_row('v55: new archive holdout',new_metric) + metric_row('Legacy holdout',old_metric)
          + '\nTraining rows are excluded from these results. Threshold calibrated on separate HAM; no refit.\n')
    (out / 'report.md').write_text(md)
    print(md, flush=True)


if __name__ == '__main__':
    main()
