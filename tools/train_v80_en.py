"""Add phishing/semantic experts; lock before fresh 2024 and old 100k regression."""
import argparse
import gzip
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import v77_mail as core
import v80_mail as runtime
from train_v78_en import fit_expert, source_weights
from train_v79_en import group_cutoff
from v79_data import fingerprint
from v80_encoder import Encoder, file_sha
from v80_metrics import enrich, upper_rate as upper_fpr


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def is_phish(row):
    return row.get('kind') == 'phish' or row['source'].startswith('Nazario-')


def measure(scores, rows, cutoff):
    return enrich(core.measure(scores, [r['y'] for r in rows], cutoff))


def recall_cutoff(scores, rows, recall=.98):
    values = np.asarray(scores)
    truth = np.asarray([r['y'] for r in rows])
    if values.shape != truth.shape or not np.isfinite(values).all() or not 0 < recall <= 1:
        raise ValueError('Invalid recall calibration input')
    spam = np.sort(values[truth == 1])
    if len(spam) < 100:
        raise ValueError('Too few calibration spam emails')
    allowed_misses = int(math.floor((1 - recall) * len(spam) + 1e-10))
    return float(spam[allowed_misses])


def source_recalls(values, rows, threshold):
    recalls = []
    def group(row):
        return 'curated-phishing' if row['source'].startswith('Nazario-') else row['source']
    for source in sorted({group(r) for r in rows}):
        indices = [i for i, r in enumerate(rows) if group(r) == source and r['y'] == 1]
        if len(indices) >= 25:
            recalls.append(float(np.mean(values[indices] >= threshold)))
    return recalls


def choose(candidates, rows, budget):
    best = None
    for name, values in candidates.items():
        cut = recall_cutoff(values, rows) if budget is None else group_cutoff(values, rows, budget)
        metric = measure(values, rows, cut)
        recalls = source_recalls(values, rows, cut)
        key = (-metric['fp'], min(recalls)) if budget is None else (min(recalls), metric['recall'], -metric['fp'])
        if best is None or key > best[0]:
            best = (key, dict(candidate=name, select_cutoff=cut, selection=metric,
                              calibration_fpr_budget=budget, worst_select_source_recall=min(recalls)))
    return best[1]


def cache_embeddings(encoder, rows, path, mask=None):
    ids = fingerprint(rows)
    text_hash = hashlib.sha256()
    for row in rows:
        raw = row['text'].encode('utf-8')
        text_hash.update(len(raw).to_bytes(8, 'big')); text_hash.update(raw)
    expected = dict(fingerprint=ids, n=len(rows), text_sha256=text_hash.hexdigest(),
                    encoder_manifest_sha256=encoder.manifest_sha,
                    mask_hash=core.digest(str(None if mask is None else mask.tolist())))
    if path.is_file():
        meta = json.loads(path.with_suffix('.json').read_text())
        if meta != expected:
            raise ValueError('Neural cache mismatch')
        return np.load(path)
    result = np.zeros((len(rows), 384), dtype=np.float32)
    indices = np.arange(len(rows)) if mask is None else np.flatnonzero(mask)
    result[indices] = encoder.encode([rows[i]['text'] for i in indices], progress=True)
    np.save(path, result)
    path.with_suffix('.json').write_text(json.dumps(expected))
    return result


def semantic_pool(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['source'], row['y']), []).append(row)
    result = []
    for (source, label), values in sorted(groups.items()):
        cap = 2500 if source == 'Merged-public' else 1000
        result += sorted(values, key=lambda r: core.digest('v80-neural:' + r['identity']))[:cap]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('.cache/v79/frozen'))
    p.add_argument('--extra', type=Path, default=Path('.cache/v80/extra'))
    p.add_argument('--base', type=Path, default=Path('.cache/v80/base/model.joblib'))
    p.add_argument('--encoder', type=Path, default=Path('.cache/v80/encoder-s7'))
    p.add_argument('--out', type=Path, default=Path('reports/v80'))
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    cache = Path('.cache/v80/features'); cache.mkdir(parents=True, exist_ok=True)
    original_manifest = json.loads((a.data / 'frozen.json').read_text())
    extra_manifest = json.loads((a.extra / 'frozen.json').read_text())
    parts = {}
    for name in ('train', 'select', 'cal'):
        original = read_rows(a.data / (name + '.jsonl.gz'))
        added = read_rows(a.extra / (name + '.jsonl.gz'))
        if fingerprint(original) != original_manifest['development_fingerprints'][name] or fingerprint(added) != extra_manifest['fingerprints'][name]:
            raise ValueError('Frozen development changed')
        parts[name] = original + added
    (a.out / 'base').mkdir(); shutil.copyfile(a.base, a.out / 'base/model.joblib')
    shutil.copytree(a.encoder, a.out / 'encoder')
    bundle = dict(version=runtime.VERSION, experts={}, gates={}, policies={},
                  original_manifest=original_manifest, extra_manifest=extra_manifest)
    runtime.attach(bundle, SimpleNamespace(model=a.out / 'model.joblib'))
    print('Development sizes', {k: len(v) for k, v in parts.items()}, flush=True)
    focused = [r for r in parts['train'] if r['y'] == 0 or is_phish(r)]
    for kind in ('word', 'char'):
        bundle['experts']['phish-' + kind] = fit_expert(focused, kind)
        joblib.dump(bundle['experts']['phish-' + kind], cache / ('phish-' + kind + '.joblib'), compress=3)
    neural_rows = semantic_pool(parts['train'])
    embedding = cache_embeddings(bundle['_encoder'], neural_rows, cache / 'train.npy')
    labels = np.asarray([r['y'] for r in neural_rows]); weights = source_weights(neural_rows)
    sem = LinearSVC(C=4., dual=False, max_iter=5000, random_state=80)
    sem.fit(embedding, labels, sample_weight=weights)
    bundle['experts']['semantic'] = sem
    indices = np.array([i for i, row in enumerate(neural_rows) if row['y'] == 0 or is_phish(row)])
    focused_rows = [neural_rows[i] for i in indices]
    semantic_phish = make_pipeline(Nystroem(gamma=6., n_components=256, random_state=80),
                                  LinearSVC(C=4., dual=False, max_iter=5000, random_state=80))
    semantic_phish.fit(embedding[indices], labels[indices], linearsvc__sample_weight=source_weights(focused_rows))
    bundle['experts']['semantic-phish'] = semantic_phish
    print('Neural heads fitted', len(neural_rows), len(focused_rows), flush=True)
    features = {}
    for name in ('select', 'cal'):
        texts = [r['text'] for r in parts[name]]
        fast = runtime.fast_features(bundle, texts)
        mask = runtime.needs_neural(fast)
        print('Neural gate', name, int(mask.sum()), '/', len(mask), flush=True)
        values = cache_embeddings(bundle['_encoder'], parts[name], cache / (name + '.npy'), mask)
        features[name] = runtime.feature_matrix(bundle, texts, fast, values)
    fit = np.array([i for i, row in enumerate(parts['select']) if int(core.digest('v80-meta:' + row['family'])[:10], 16) % 2 == 0])
    fit_set = set(fit.tolist())
    select = np.array([i for i in range(len(parts['select'])) if i not in fit_set])
    select_rows = [parts['select'][i] for i in select]
    fit_rows = [parts['select'][i] for i in fit]
    labels = np.asarray([r['y'] for r in fit_rows]); weights = source_weights(fit_rows)
    for ham_cost in (1., 4.):
        w = weights * np.where(labels == 0, ham_cost, 1.)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000, random_state=80))
        model.fit(features['select'][fit], labels, logisticregression__sample_weight=w)
        bundle['gates']['linear-ham' + str(int(ham_cost))] = model
        tree = HistGradientBoostingClassifier(max_leaf_nodes=7, max_iter=150, min_samples_leaf=60,
                                             l2_regularization=15., random_state=80)
        tree.fit(features['select'][fit], labels, sample_weight=w)
        bundle['gates']['tree-ham' + str(int(ham_cost))] = tree
    selected = runtime.candidate_scores(bundle, features['select'][select])
    cal = runtime.candidate_scores(bundle, features['cal'])
    for profile, budget in runtime.PROFILES.items():
        cfg = choose(selected, select_rows, budget)
        values = cal[cfg['candidate']]
        cal_cut = recall_cutoff(values, parts['cal']) if budget is None else group_cutoff(values, parts['cal'], budget)
        cfg['cutoff'] = min(cfg['select_cutoff'], cal_cut) if budget is None else max(cfg['select_cutoff'], cal_cut)
        cfg['calibration'] = measure(values, parts['cal'], cfg['cutoff'])
        bundle['policies'][profile] = cfg
    serializable = {k: v for k, v in bundle.items() if not k.startswith('_')}
    model_path = a.out / 'model.joblib'; joblib.dump(serializable, model_path, compress=3)
    locked = file_sha(model_path)
    (a.out / 'policies-locked.json').write_text(json.dumps(dict(model_sha256=locked, policies=bundle['policies']), indent=2))
    print('LOCKED BEFORE FRESH 2024 TEST', locked, json.dumps(bundle['policies']), flush=True)
    del parts, features, embedding
    from evaluate_v80_en import evaluate
    evaluate(a.data, a.extra, a.out)


if __name__ == '__main__':
    main()
