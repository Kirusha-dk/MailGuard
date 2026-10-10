#!/usr/bin/env python3
"""Keep v48 decisions; learn a precision gate for residual neural detections."""
import argparse
import copy
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from benchmark_v48_calibrated import arrays, features, prepare, threshold_for_budget
from train_v47_phishing_aware import metrics_at
from rspamd_benchmark_audit import load_frozen_v40
from train_v50_neural import ByteCNN, predict_neural


def gate_features(sparse, neural):
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    s, n = logit(sparse), logit(neural)
    return np.column_stack([s, n, n - s, sparse * neural, np.minimum(sparse, neural)])


def residual_threshold(y, base, scores, max_fp):
    """Reserve FP allowance for base; calibrate only remaining ham decisions."""
    y, base = np.asarray(y), np.asarray(base, dtype=bool)
    inherited = int(((y == 0) & base).sum())
    residual = ~base
    if inherited > max_fp or not np.any((y == 0) & residual):
        return 2., inherited
    return threshold_for_budget(y[residual], scores[residual], max_fp - inherited), inherited


def scored_union(base, rescue_scores, threshold):
    # Sentinel 2 exceeds all rescue probabilities and preserves base decisions.
    # Threshold 2 disables every residual rescue while keeping the base.
    scores = np.asarray(rescue_scores).copy()
    scores[np.asarray(base, dtype=bool)] = 2.
    return scores


def predict_scores(artifact, sparse, neural):
    gate = artifact.get('residualGate')
    rescue = neural if gate is None else gate.predict_proba(gate_features(sparse, neural))[:, 1]
    base = sparse >= artifact['baseThreshold']
    return scored_union(base, rescue, artifact['threshold'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--output', default='reports/v51-smoke')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    v48_files = list(Path('.cache/v48-artifact').rglob('v48-full/model.joblib'))
    v50_files = list(Path('.cache/v50-artifact').rglob('v50-full/neural.pt'))
    if len(v48_files) != 1 or len(v50_files) != 1:
        raise RuntimeError('Expected exact frozen v48 and v50 artifacts')
    baseline = joblib.load(v48_files[0])
    neural = ByteCNN()
    ckpt = torch.load(v50_files[0], map_location='cpu', weights_only=True)
    neural.load_state_dict(ckpt['state'])
    frozen, audit = load_frozen_v40('.cache/v40-fresh50k')
    data, dedup = prepare(frozen, args.limit)
    val = data['val']
    fit = [r for r in val if int(r['identity'][:8], 16) % 4 == 0]
    select = [r for r in val if int(r['identity'][:8], 16) % 4 == 2]
    cal = [r for r in val if int(r['identity'][:8], 16) % 2 == 1]
    def component(rows):
        texts = [r['ngram_text'] for r in rows]
        s = baseline['model'].predict_proba(baseline['tfidf'].transform(features(texts)))[:, 1]
        return s, predict_neural(neural, texts)
    sf, nf = component(fit)
    ss, ns = component(select)
    sc, nc = component(cal)
    yf, _ = arrays(fit)
    ys, sources = arrays(select)
    yc, csources = arrays(cal)
    models = [('neural-only-rescue', None)]
    for c in (.1, 1.):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000, random_state=51))
        model.fit(gate_features(sf, nf), yf)
        models.append((f'logistic-{c}', model))
    tree = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=7,
                                          min_samples_leaf=50, l2_regularization=5., random_state=51)
    tree.fit(gate_features(sf, nf), yf)
    models.append(('small-tree-gate', tree))
    base = ss >= baseline['threshold']
    candidates = []
    # Original v48 is an explicit fallback; no forced rescue if it loses validation.
    bm = metrics_at(ys, scored_union(base, np.zeros(len(base)), 2.), 2., sources)
    candidates.append((bm, 'v48-no-rescue', None, True))
    for name, gate in models:
        p = ns if gate is None else gate.predict_proba(gate_features(ss, ns))[:, 1]
        t, inherited = residual_threshold(ys, base, p, int(len(select) * .0012))
        m = metrics_at(ys, scored_union(base, p, t), t, sources)
        m['inheritedFP'] = inherited
        candidates.append((m, name, gate, False))
    chosen, name, gate, disabled = max(candidates, key=lambda c: (c[0]['recall'], -c[0]['fp']))
    basec = sc >= baseline['threshold']
    pc = nc if gate is None else gate.predict_proba(gate_features(sc, nc))[:, 1]
    threshold, inherited = residual_threshold(yc, basec, pc, int(len(cal) * .0012))
    if disabled:
        threshold = 2.
    artifact = copy.deepcopy(baseline)
    artifact.update(version='v51-residual-neural', baseThreshold=baseline['threshold'],
                    residualGate=gate, threshold=threshold, neuralWeight=1., neuralCheckpoint='neural.pt')
    shutil.copyfile(v50_files[0], out / 'neural.pt')
    joblib.dump(artifact, out / 'model.joblib', compress=3)
    test = data['test']
    st, nt = component(test)
    yt, tsources = arrays(test)
    scores = predict_scores(artifact, st, nt)
    result = metrics_at(yt, scores, threshold, tsources)
    report = dict(version=51, selected=name, count=len(test), test=result,
                  calibration=metrics_at(yc, scored_union(basec, pc, threshold), threshold, csources),
                  calibrationInheritedFP=inherited, candidates=[dict(name=n, metric=m) for m,n,_,_ in candidates],
                  audit=audit, dedup=dedup, gateFit=len(fit), selection=len(select), calibrationSize=len(cal),
                  neuralRetrained=False, baselineDecisionsPreserved=True,
                  warning='Development and engineering comparison; reused data, not independent 98% evidence.')
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    text = (f"# MailGuard v51 residual rescue\n\n{report['warning']}\n\nSelected: {name}\n\n"
            '| N | Recall | FP | FN |\n|---:|---:|---:|---:|\n'
            f"| {len(test)} | {result['recall']:.2%} | {result['fp']} | {result['fn']} |\n")
    (out / 'report.md').write_text(text)
    print(text, flush=True)


if __name__ == '__main__':
    main()
