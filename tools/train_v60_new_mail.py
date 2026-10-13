#!/usr/bin/env python3
"""Train/select/calibrate v60 without May labels; test only after model SHA is locked."""
import copy
import csv
import gc
import json
import os
import shutil
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn
from scipy.sparse import save_npz, load_npz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from train_v55_multilingual import FEATURES, corpus_rows, features, file_sha, predict, transformed, views, compact
from train_v59_new_mail import PROFILES, metrics, threshold, score_batches
from v59_runtime import load_bundle as load59, predict_texts as predict59
from v60_neural import UnicodeCNN, encode, predict as predict_neural
from v60_runtime import stack_features, candidate_score


def group(source):
    return 'chinese' if source == 'TREC06-Chinese' else 'recent' if source.startswith('SpamArchive-2026-') else 'historical-archive' if source.startswith('SpamArchive-') else 'legacy'


def key(rows, scores, cutoff):
    names = np.array([group(r['source']) for r in rows])
    values = []
    for name in sorted(set(names)):
        ix = np.flatnonzero(names == name)
        batch = [rows[i] for i in ix]
        if any(r['y'] for r in batch):
            values.append(metrics(batch, scores[ix], cutoff)['recall'])
    return min(values), float(np.mean(values))


def cutoff(rows, scores, fpr):
    cuts = [threshold(rows, scores, fpr)]
    # HAM sources have separate allowances. Pooled Chinese HAM must not conceal
    # false positives on English mail, or vice versa.
    for name in ('chinese', 'legacy'):
        ix = np.array([i for i, r in enumerate(rows) if group(r['source']) == name])
        if len(ix) and any(rows[i]['y'] == 0 for i in ix):
            cuts.append(threshold([rows[i] for i in ix], scores[ix], fpr))
    return max(cuts)


def fit_sparse(train):
    counts = Counter((r['source'], r['y']) for r in train)
    weights = np.asarray([min(1., 6000 / counts[r['source'], r['y']]) * (4. if r['y'] == 0 else 1.)
                          * (.25 if r.get('weakLabel') else 1.) for r in train], dtype=np.float64)
    rng = np.random.default_rng(60)
    order = rng.permutation(len(train))
    train = [train[i] for i in order]
    weights = weights[order]
    y_train = np.asarray([r['y'] for r in train])
    cache = Path('.cache/v60-feature-cache'); cache.mkdir(parents=True, exist_ok=False)
    df = np.zeros(FEATURES, dtype=np.float64)
    totals = [np.zeros(FEATURES, dtype=np.float64) for _ in (0, 1)]
    for start in range(0, len(train), 256):
        raw = features(train[start:start+256]); y = np.asarray([r['y'] for r in train[start:start+256]])
        save_npz(cache / f'{start:08d}.npz', raw, compressed=True)
        df += np.bincount(raw.indices, minlength=FEATURES)
        for label in (0, 1):
            # Frequency ratios also respect the source caps, without the HAM cost multiplier.
            w = weights[start:start+len(y)][y == label] / (4. if label == 0 else 1.)
            totals[label] += np.asarray(raw[y == label].multiply(w[:, None]).sum(axis=0)).ravel()
        if start % 8192 == 0:
            print('TRAIN feature cache', start, '/', len(train), flush=True)
    idf = (np.log((1 + len(train)) / (1 + df)) + 1).astype(np.float32)
    pos, neg = totals[1] + 1., totals[0] + 1.
    nb = np.log((pos / pos.sum()) / (neg / neg.sum())).astype(np.float32)
    del df, totals, pos, neg
    candidates = [dict(name=f'{rep}-alpha{alpha}', model=SGDClassifier(loss='log_loss', alpha=alpha, average=True, random_state=60),
                       idf=idf, ratio=nb if rep == 'nb' else None)
                  for rep in ('tfidf', 'nb') for alpha in (3e-6, 1e-6)]
    for epoch in range(4):
        # Initial global shuffle mixes sources/classes; each epoch also shuffles
        # blocks and rows. Only TRAIN features are cached, never TEST features.
        blocks = rng.permutation(np.arange(0, len(train), 256))
        for start in blocks:
            raw = load_npz(cache / f'{start:08d}.npz')
            within = rng.permutation(raw.shape[0])
            raw = raw[within]
            ix = start + within; y = y_train[ix]
            for rep in (None, nb):
                x = transformed(raw, idf, rep)
                for c in candidates:
                    if c['ratio'] is rep:
                        c['model'].partial_fit(x, y, classes=np.asarray([0, 1]), sample_weight=weights[ix])
        print('v60 sparse epoch', epoch+1, '/', 4, 'rows', len(train), flush=True)
    shutil.rmtree(cache)
    return candidates


def fit_neural(train, selection):
    torch.manual_seed(60)
    model = UnicodeCNN()
    counts = Counter((group(r['source']), r['y']) for r in train)
    weight = torch.tensor([min(1., 6000 / counts[group(r['source']), r['y']])
                           * (3. if r['y'] == 0 else 1.) * (.25 if r.get('weakLabel') else 1.) for r in train], dtype=torch.float32)
    x = torch.from_numpy(encode(train)); y = torch.tensor([r['y'] for r in train], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    rng = np.random.default_rng(60)
    history, best, state, best_epoch = [], None, None, None
    for epoch in range(4):
        model.train(); order = rng.permutation(len(train)); loss_sum = 0.
        for start in range(0, len(train), 96):
            ix = order[start:start+96]
            optimizer.zero_grad(set_to_none=True)
            loss = (loss_fn(model(x[ix]), y[ix]) * weight[ix]).mean()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.); optimizer.step()
            loss_sum += float(loss.detach()) * len(ix)
            if start % 12288 == 0:
                print('Unicode CNN epoch', epoch+1, 'rows', start, '/', len(train), flush=True)
        scores = predict_neural(model, selection)
        cut = cutoff(selection, scores, PROFILES['high-recall'])
        chosen_key = key(selection, scores, cut)
        history.append(dict(epoch=epoch+1, loss=loss_sum/len(train), selection=metrics(selection, scores, cut), macroKey=chosen_key))
        print('CNN selection', json.dumps(history[-1]), flush=True)
        if best is None or chosen_key > best:
            best, state, best_epoch = chosen_key, copy.deepcopy(model.state_dict()), epoch+1
    model.load_state_dict(state)
    return model, dict(state=state, selectedEpoch=best_epoch, history=history)


def restore_validation(rows):
    from rspamd_benchmark_audit import load_frozen_v40
    from train_v47_phishing_aware import from_eml
    wanted = {r['identity']: r for r in rows if 'text' not in r}
    frozen, _ = load_frozen_v40('.cache/v40-fresh50k')
    for entry in frozen['val']:
        parsed = from_eml(entry['path'].read_bytes())
        if parsed['identity'] in wanted:
            wanted[parsed['identity']]['text'] = parsed['ngram_text']
    if any('text' not in r for r in rows):
        raise AssertionError('Canonical validation text missing')


def previous_scores(bundle, rows, cache=None):
    result = {p: [] for p in PROFILES}
    cache = cache or {}
    # v59 has the same gate for both profiles, so its continuous score needs one
    # inference pass. Keep separate code paths if a future bundle differs.
    name = bundle[0]['profiles']['conservative']['selected']
    shared = name != 'cjk-specialist' and name == bundle[0]['profiles']['high-recall']['selected']
    for start in range(0, len(rows), 128):
        batch = rows[start:start+128]
        positions = [i for i,r in enumerate(batch) if r['identity'] not in cache]
        texts = [batch[i]['text'] for i in positions]
        originals = [batch[i].get('old_text',batch[i]['text']) for i in positions]
        scores = predict59(bundle, texts, 'conservative', originals) if positions else np.empty(0)
        scores_high = scores if shared or not positions else predict59(bundle,texts,'high-recall',originals)
        for profile, values in (('conservative',scores),('high-recall',scores_high)):
            column = 'conservativeScore' if profile == 'conservative' else 'highRecallScore'
            complete = np.array([float(cache[r['identity']][column]) if r['identity'] in cache else np.nan for r in batch])
            complete[positions] = values
            result[profile].append(complete)
        if start % 4096 == 0:
            print('Frozen v59 inference', start, '/', len(rows), flush=True)
    return {p: np.concatenate(v) for p, v in result.items()}


def legacy_rows():
    from rspamd_benchmark_audit import load_frozen_v40
    from train_v47_phishing_aware import from_eml
    with Path('.cache/v55/v55-full/split-manifest.csv').open() as f:
        wanted = {r['identity']: r for r in csv.DictReader(f) if r['split'] == 'test' and not r['source'].startswith('SpamArchive-')}
    restored = {}
    def retain(parsed, y, source):
        entry = wanted.get(parsed['identity'])
        if entry and (int(entry['label']), entry['source']) == (y, source) and parsed['identity'] not in restored:
            r = compact(dict(parsed, y=y, source=source), entry['family'])
            r['text'] = parsed['ngram_text']; restored[r['identity']] = r
    frozen, _ = load_frozen_v40('.cache/v40-fresh50k')
    for entry in frozen['test']:
        retain(from_eml(entry['path'].read_bytes()), int(entry['y']), entry['source'])
    for r in corpus_rows('.cache/v54-corpus/selected.jsonl.gz'):
        if r['identity'] in wanted:
            retain(r, int(r['y']), r['source'])
    if set(restored) != set(wanted):
        raise AssertionError('Legacy holdout reconstruction incomplete')
    return list(restored.values())


def main():
    torch.set_num_threads(4)
    data = Path('.cache/v60-data'); out = Path('reports/v60-full'); out.mkdir(parents=True, exist_ok=False)
    provenance = json.loads((data / 'provenance.json').read_text())
    for split, digest in provenance['splitHashes'].items():
        if file_sha(data / f'{split}.jsonl.gz') != digest:
            raise ValueError('Frozen v60 split hash mismatch')
    train, selection, cal = [list(corpus_rows(data / f'{s}.jsonl.gz')) for s in ('train', 'select', 'cal')]
    if any(r.get('weakLabel') for r in selection + cal):
        raise AssertionError('Weak HAM outside TRAIN')
    restore_validation(selection + cal)
    fit_ix = np.array([i for i,r in enumerate(selection) if int(r['family'][-12:],16)%10 < 6])
    choose_ix = np.array([i for i,r in enumerate(selection) if int(r['family'][-12:],16)%10 >= 6])
    choose = [selection[i] for i in choose_ix]
    candidates = fit_sparse(train)
    previous_bundle = load59('.cache/v59-model')
    candidates.append(dict(previous_bundle[0]['core'], name='v59-frozen-core'))
    core_results = []
    for core in candidates:
        scores = score_batches(core, choose); cut = cutoff(choose, scores, PROFILES['high-recall'])
        core['key'] = key(choose, scores, cut)
        core_results.append(dict(name=core['name'], selection=metrics(choose, scores, cut), macroKey=core['key']))
        print('v60 core selection', json.dumps(core_results[-1]), flush=True)
    core = max(candidates, key=lambda c: c['key'])
    model, neural_checkpoint = fit_neural(train, choose)
    del train, candidates; gc.collect()
    sparse = score_batches(core, selection); neural = predict_neural(model, selection)
    previous = previous_scores(previous_bundle, selection)
    sparse_cal = score_batches(core, cal); neural_cal = predict_neural(model, cal)
    previous_cal = previous_scores(previous_bundle, cal)
    labels = np.array([r['y'] for r in selection]); counts = Counter((group(selection[i]['source']), selection[i]['y']) for i in fit_ix)
    gates, profiles, gate_results = {}, {}, []
    for profile, fpr in PROFILES.items():
        x = stack_features(selection, sparse, neural, previous[profile]); xc = stack_features(cal, sparse_cal, neural_cal, previous_cal[profile])
        gates[profile] = {}
        for ham_cost in (4., 12.):
            w = np.array([min(1., 1000 / counts[group(selection[i]['source']), selection[i]['y']]) * (ham_cost if labels[i] == 0 else 1.) for i in fit_ix])
            gate = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=7, min_samples_leaf=50, l2_regularization=12., random_state=60)
            gate.fit(x[fit_ix], labels[fit_ix], sample_weight=w)
            gates[profile][f'gate-ham{int(ham_cost)}'] = gate
            linear = make_pipeline(StandardScaler(), LogisticRegression(C=.2, max_iter=1000, random_state=60))
            linear.fit(x[fit_ix], labels[fit_ix], logisticregression__sample_weight=w)
            gates[profile][f'linear-ham{int(ham_cost)}'] = linear
        options = []
        for name in ('v59-frozen','sparse-only','neural-only','blend-0.15','blend-0.35',*gates[profile]):
            scores = candidate_score(name,gates[profile],x[choose_ix],sparse[choose_ix],neural[choose_ix],previous[profile][choose_ix])
            cut = cutoff(choose, scores, fpr); rank = key(choose, scores, cut)
            options.append((rank,name,cut))
            gate_results.append(dict(profile=profile,candidate=name,macroKey=rank,selection=metrics(choose,scores,cut)))
        _, name, select_cut = max(options)
        scores = candidate_score(name,gates[profile],xc,sparse_cal,neural_cal,previous_cal[profile])
        cut = max(select_cut,cutoff(cal,scores,fpr))
        profiles[profile] = dict(selected=name,threshold=cut,targetHamFpr=fpr,calibration=metrics(cal,scores,cut))
    shutil.copytree('.cache/v59-model',out/'v59')
    torch.save(neural_checkpoint,out/'unicode.pt')
    artifact = dict(version='v60-unicode-ensemble',core=core,gates=gates,profiles=profiles,
                    selectionObjective='minimum recall across Chinese, recent 2026 spam, historical archive, legacy; then average',
                    calibration='Separate Chinese/legacy HAM budgets; never relax selection threshold',
                    codeCommit=os.environ.get('GITHUB_SHA'),githubRunId=os.environ.get('GITHUB_RUN_ID'),
                    components={str(p.relative_to(out)):file_sha(p) for p in out.rglob('*') if p.is_file()},
                    splitManifestSha256=provenance['manifestSha256'])
    joblib.dump(artifact,out/'model.joblib',compress=3)
    locked = file_sha(out/'model.joblib'); neural_sha = file_sha(out/'unicode.pt')
    print('MODEL AND THRESHOLDS LOCKED',locked,json.dumps(profiles),flush=True)
    from v60_runtime import load_bundle, predict_texts
    bundle = load_bundle(out)
    for profile in PROFILES:
        setting = profiles[profile]
        x = stack_features(cal[:32],sparse_cal[:32],neural_cal[:32],previous_cal[profile][:32])
        expected = candidate_score(setting['selected'],gates[profile],x,sparse_cal[:32],neural_cal[:32],previous_cal[profile][:32])
        observed = predict_texts(bundle,[r['text'] for r in cal[:32]],profile,[r.get('old_text',r['text']) for r in cal[:32]])
        if not np.allclose(expected,observed,rtol=1e-6,atol=1e-7):
            raise AssertionError('Saved runtime differs from training inference')
    with Path('.cache/v59-model/predictions.csv').open() as f:
        previous_cache = {r['identity']:r for r in csv.DictReader(f)}
    def evaluate(rows, title):
        print('EVALUATING',title,len(rows),flush=True)
        sparse = score_batches(core,rows); neural = predict_neural(model,rows); old = previous_scores(previous_bundle,rows,previous_cache)
        comparisons = {}
        with (out/(title+'-predictions.csv')).open('w',newline='') as f:
            writer = csv.writer(f); writer.writerow(['identity','family','source','label','v60Conservative','v60HighRecall','v59Conservative','v59HighRecall'])
            all_scores = {}
            for profile in PROFILES:
                x = stack_features(rows,sparse,neural,old[profile]); setting = profiles[profile]
                scores = candidate_score(setting['selected'],gates[profile],x,sparse,neural,old[profile]); all_scores[profile]=scores
                comparisons[profile] = dict(v60=metrics(rows,scores,setting['threshold']),v59=metrics(rows,old[profile],previous_bundle[0]['profiles'][profile]['threshold']))
            for i,r in enumerate(rows):
                writer.writerow([r['identity'],r['family'],r['source'],r['y'],all_scores['conservative'][i],all_scores['high-recall'][i],old['conservative'][i],old['high-recall'][i]])
        if locked != file_sha(out/'model.joblib') or neural_sha != file_sha(out/'unicode.pt'):
            raise AssertionError('Model changed after TEST opened')
        return dict(count=len(rows),ham=sum(r['y']==0 for r in rows),profiles=comparisons)
    # The new TEST has not been opened, scored or used for fitting before here.
    test = list(corpus_rows(data/'test.jsonl.gz'))
    if any(r.get('weakLabel') for r in test):
        raise AssertionError('Weak HAM in new TEST')
    fresh = evaluate(test,'fresh')
    old59 = evaluate(list(corpus_rows('.cache/v59-data/test.jsonl.gz')),'v59-regression')
    legacy = evaluate(legacy_rows(),'legacy-regression')
    result = dict(version=60,codeCommit=os.environ.get('GITHUB_SHA'),githubRunId=os.environ.get('GITHUB_RUN_ID'),
                  profiles=profiles,freshTest=fresh,v59Regression=dict(old59,reused=True),legacyRegression=dict(legacy,reused=True),
                  modelSha256=locked,neuralSha256=neural_sha,coreCandidates=core_results,gateSelection=gate_results,
                  selectedCore=core['name'],neuralHistory=neural_checkpoint['history'],selectedNeuralEpoch=neural_checkpoint['selectedEpoch'],
                  estimatorAndThresholdLockedBeforeTest=True,testLabelsUsedForTraining=False,v59TestAddedToTraining=False,
                  productionGoalValidated=False,provenance=provenance)
    (out/'report.json').write_text(json.dumps(result,indent=2)); shutil.copyfile(data/'manifest.csv',out/'split-manifest.csv')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    md = '# MailGuard v60 — Unicode neural ensemble\n\nModel and thresholds frozen before new May TEST. Prior v59 TEST never enters fitting.\n\n'
    md += '| Dataset | Model/profile | Mail | HAM | Spam recall | FP | FN |\n|---|---|---:|---:|---:|---:|---:|\n'
    for title,dataset in (('New TEST',fresh),('Reused v59 TEST',old59),('Reused English/CSDMC',legacy)):
        for profile,comparison in dataset['profiles'].items():
            for version,m in comparison.items():
                md += f'| {title} | {version} {profile} | {dataset["count"]} | {dataset["ham"]} | {m["recall"]:.2%} | {m["fp"]} | {m["fn"]} |\n'
    md += '\n'+provenance['caveat']+'\n'
    (out/'report.md').write_text(md); print(md,flush=True)


if __name__ == '__main__':
    main()
