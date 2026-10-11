#!/usr/bin/env python3
"""Fixed OR of frozen v54 and frozen v56 decisions. No fitting or retuning."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import joblib
import numpy as np

from train_v55_multilingual import file_sha
from train_v56_consensus import is_new, WARNING
from v57_runtime import protect_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='.cache/v56-artifact/v56-full')
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--output', default='reports/v57-smoke')
    args = parser.parse_args()
    if args.limit < 0 or args.limit == 1:
        parser.error('Limit must be zero or at least two')
    source, out = Path(args.source), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    prior = json.loads((source / 'report.json').read_text())
    if prior['limit'] != 0 or prior['testCount'] != 395276:
        raise ValueError('Expected the completed full v56 report')
    if file_sha(source / 'model.joblib') != prior['modelSha256']:
        raise ValueError('Frozen v56 model changed')
    artifact = joblib.load(source / 'model.joblib')
    if artifact['version'] != 'v56-consensus':
        raise ValueError('Expected v56')
    for name, digest in artifact['components'].items():
        if file_sha(source / name) != digest:
            raise ValueError(f'Component hash mismatch: {name}')
        target = out / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    artifact['version'] = 'v57-preserved-consensus'
    artifact['protectedV54Decisions'] = True
    artifact['sourceV56Run'] = 38171539756
    artifact['sourceV56ModelSha256'] = prior['modelSha256']
    for profile in artifact['profiles'].values():
        profile['sourceV56Calibration'] = profile.pop('calibration')
        profile['sourceV56DevelopmentHamFpr'] = profile.pop('developmentHamFpr')
        profile['sourceV56HamAllowance'] = profile.pop('hamAllowance')
        profile['unionCalibrationEvaluated'] = False
        if not 0 <= profile['threshold'] <= 1.:
            raise ValueError('The protected-vote sentinel must exceed the frozen threshold')
    joblib.dump(artifact, out / 'model.joblib', compress=3)
    counts = {profile: {group: dict(tp=0, fn=0, fp=0, tn=0) for group in ('new', 'legacy', 'all')}
              for profile in artifact['profiles']}
    selected_groups = {False: 0, True: 0}
    seen = set()
    with (source / 'predictions.csv').open() as f, (out / 'predictions.csv').open('w', newline='') as pf:
        reader = csv.DictReader(f); writer = csv.writer(pf)
        writer.writerow(['identity', 'family', 'source', 'label', 'v54Spam'] + [f'{profile}Spam' for profile in counts])
        for row in reader:
            new = is_new(row['source'])
            if args.limit and selected_groups[new] >= args.limit // 2:
                continue
            if row['identity'] in seen:
                raise AssertionError('Duplicate test identity')
            seen.add(row['identity']); selected_groups[new] += 1
            y, base = int(row['label']), bool(int(row['v54Spam']))
            decisions = []
            for profile in counts:
                source_vote = bool(int(row[f'{profile}Spam']))
                pred = base or source_vote
                # Exercise the same sentinel rule used by live inference.
                t = artifact['profiles'][profile]['threshold']
                mock_score = t if source_vote else -1.
                actual = bool(protect_scores([base], [mock_score])[0] >= t)
                if actual != pred or (base and not pred):
                    raise AssertionError('Protected vote lost')
                decisions.append(int(pred))
                key = ('tp' if pred else 'fn') if y else ('fp' if pred else 'tn')
                counts[profile]['new' if new else 'legacy'][key] += 1
                counts[profile]['all'][key] += 1
            writer.writerow([row['identity'], row['family'], row['source'], y, int(base)] + decisions)
    results = {}
    for profile, groups in counts.items():
        results[profile] = {}
        for group, c in groups.items():
            results[profile][group] = dict(c, recall=c['tp']/max(1,c['tp']+c['fn']),
                                         fpr=c['fp']/max(1,c['fp']+c['tn']))
    n = len(seen)
    if not args.limit and n != prior['testCount']:
        raise AssertionError('Matched holdout count changed')
    report = dict(version=57, limit=args.limit, testCount=n, results=results,
                  sourceV56Run=38171539756, sourceV56ModelSha256=prior['modelSha256'],
                  sourcePredictionsSha256=file_sha(source / 'predictions.csv'), modelSha256=file_sha(out / 'model.joblib'),
                  profiles=artifact['profiles'], protectedV54Decisions=True,
                  thresholdsChanged=False, estimatorRefit=False,
                  testLabelsUsedForEstimatorOrThresholdFit=False, testPreviouslyInspected=True,
                  independentBenchmark=False, productionGoalValidated=False,
                  warning=WARNING + ' v57 is a fixed OR; source-v56 calibration budgets do not constrain inherited v54 FP.')
    report['engineeringGoals'] = {profile: dict(recall95AndFp50=(r['all']['recall'] >= .95 and r['all']['fp'] <= 50 and not args.limit),
                                               bothSourceGroupsRecall95=(r['new']['recall'] >= .95 and r['legacy']['recall'] >= .95 and not args.limit))
                                  for profile, r in results.items()}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    md = '# MailGuard v57 preserved consensus\n\n' + report['warning'] + '\n\n| Profile | Group | N | Ham | Spam recall | FP | FN |\n|---|---|---:|---:|---:|---:|---:|\n'
    for profile, groups in results.items():
        for group, m in groups.items():
            total = sum(m[k] for k in ('tp','fn','fp','tn'))
            md += f"| {profile} | {group} | {total} | {m['fp']+m['tn']} | {m['recall']:.2%} | {m['fp']} | {m['fn']} |\n"
    (out / 'report.md').write_text(md)
    print(md, flush=True)


if __name__ == '__main__':
    main()
