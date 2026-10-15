#!/usr/bin/env python3
"""Audit v65 claims without fitting, choosing cutoffs, or inspecting email text.

Separate family-disjoint regression, unseen training-source checks and actual
first-use sources. Never call a previously benchmarked corpus "independent".
"""
import argparse
import ast
import json
import math
from pathlib import Path

TARGET_FPR = 1e-4


def wilson_upper(fp, ham, z=1.959963984540054):
    if not isinstance(fp,int) or not isinstance(ham,int) or not (0 <= fp <= ham):
        raise ValueError('invalid FP/HAM counters')
    if not ham:
        return None
    p=fp/ham
    zz=z*z
    return (p+zz/(2*ham)+z*math.sqrt((p*(1-p)+zz/(4*ham))/ham))/(1+zz/ham)


def min_zero_fp_ham(fpr=TARGET_FPR, confidence=.95):
    """Exact one-sided binomial upper target if zero false positives are seen."""
    return math.ceil(math.log(1-confidence)/math.log1p(-fpr))


def source_names_from_split(counts):
    names=set()
    for entries in counts.values():
        for key in entries:
            if ':' not in key:
                raise ValueError('Malformed provenance source key')
            source,label=key.rsplit(':',1)
            if label not in ('0','1'):
                raise ValueError('Malformed provenance class')
            names.add(source)
    return names


def sources_from_v65_training(report):
    names=set()
    for key,count in report['splitCounts'].items():
        parts=ast.literal_eval(key)
        if not isinstance(parts,tuple) or len(parts)!=3:
            raise ValueError('Malformed v65 split key')
        split,source,_=parts
        if split in ('train','cal') and count:
            names.add(source)
    return names


def validate_metrics(metrics):
    if not all(k in metrics for k in ('n','spam','ham','tp','fn','fp','tn','recall','fpr')):
        raise ValueError('Missing v65 metric field')
    if metrics['n']!=metrics['spam']+metrics['ham']:
        raise ValueError('Incorrect sample count')
    if metrics['spam']!=metrics['tp']+metrics['fn']:
        raise ValueError('Incorrect spam count')
    if metrics['ham']!=metrics['fp']+metrics['tn']:
        raise ValueError('Incorrect ham count')
    for label,num in (('recall',metrics['spam']),('fpr',metrics['ham'])):
        observed=metrics[label]
        correct=(metrics['tp']/num if label=='recall' else metrics['fp']/num) if num else None
        if observed is None and correct is None:continue
        if observed is None or correct is None or abs(observed-correct)>1e-10:
            raise ValueError('Incorrect '+label)


def audit(v65,v64,provenance):
    if v65.get('version')!=65 or v64.get('version')!=64 or provenance.get('version')!=60:
        raise ValueError('Unexpected versions')
    if not v65.get('modelLockedBeforeHoldout') or v65.get('testLabelsUsedForTraining'):
        raise ValueError('Cannot attest v65 holdout protocol')
    previously_scored=set(v64['audit']['bySource'])
    old_dev=source_names_from_split(provenance['counts'])
    adapt_sources=sources_from_v65_training(v65)
    if not adapt_sources<=previously_scored:
        raise ValueError('Unexpected v65 TRAIN/CAL provenance')
    all_holdout=v65['results']['all-holdout']
    for metrics in all_holdout.values():validate_metrics(metrics)
    for group,table in v65['results'].items():
        for metrics in table.values():validate_metrics(metrics)
    source_groups={key.removeprefix('source/'):v for key,v in v65['results'].items()
                   if key.startswith('source/')}
    if not source_groups:
        raise ValueError('No source-group result')
    audited={}
    for source,versions in sorted(source_groups.items()):
        current=versions['v65-conservative-rescue']
        status=('trained_or_calibrated_by_v65' if source in adapt_sources else
                'used_in_older_development' if source in old_dev else
                'first_use' if source not in previously_scored else
                'previously_scored_source')
        audited[source]=dict(status=status,count=current['n'],spam=current['spam'],
                             ham=current['ham'],recall=current['recall'],fp=current['fp'],
                             fpr=current['fpr'],fprWilson95Upper=wilson_upper(current['fp'],current['ham']),
                             previouslyScored=source in previously_scored,
                             usedByV65Adaptation=source in adapt_sources)
    for version,metrics in all_holdout.items():
        weighted_sum=sum(v[version]['n'] for v in source_groups.values())
        if weighted_sum!=metrics['n']:
            raise ValueError('Source row totals do not match overall '+version)
    if all_holdout['v65-conservative-rescue']['n']!=v65['holdoutCount']:
        raise ValueError('Holdout count mismatch')
    selection=[key for key in v65['results'] if key.startswith('fixed-100k/')]
    for name in selection:
        m=v65['results'][name]['v65-conservative-rescue']
        if m['n']!=100000:
            raise ValueError('Incorrect 100k snapshot size')
    first_use=[name for name,x in audited.items() if x['status']=='first_use']
    unseen_adaptation=[name for name,x in audited.items() if x['status'] in ('previously_scored_source','first_use')]
    held=all_holdout['v65-conservative-rescue']
    upper=wilson_upper(held['fp'],held['ham'])
    return dict(audit='v65-transfer-validity',firstUseSources=first_use,
                sourceDisjointFromV65Development=unseen_adaptation,
                previouslyScoredSourceNames=sorted(previously_scored),
                oldDevelopmentSources=sorted(old_dev),
                v65AdaptationSources=sorted(adapt_sources),
                bySource=audited,holdout=dict(held,fprWilson95Upper=upper),
                minHAMForZeroFPtoBoundFPRbelowTarget95=min_zero_fp_ham(),
                targetFpr=TARGET_FPR,
                sourceIndependentPerformanceValidated=bool(first_use),
                fpRequirementDemonstrated=upper is not None and upper<=TARGET_FPR,
                generalizationToAny100kDemonstrated=False,
                fixed100kHaveHAM=all(v65['results'][name]['v65-conservative-rescue']['ham']>0 for name in selection),
                sourceCountChecksPassed=True,
                warning='All v65 evaluated sources were previously scored by v64; true first-use source count is zero. GenSpam and Published-ASCII-mail are reused during v65 fitting or calibration, Nazario-2025 is not fitted but was benchmarked by v64. Historical 100k archive snapshots contain no HAM. Low FP risk in production is not established.')


def report_md(a):
    rows=[
        '# MailGuard v65 — verified provenance and FPR audit','',
        '**This is an audit, NOT a new benchmark or trained model.**','',
        'v65 source-disjoint adaptation is not the same as independent first-use evidence.',
        'All checked v65 sources already appeared in v64 first-use experiments.',
        '',
        '| Source | Status | Messages | Spam | Recall | FP/HAM |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for name,s in sorted(a['bySource'].items()):
        recall='n/a' if s['recall'] is None else f"{100*s['recall']:.2f}%"
        rows.append(f"| {name} | {s['status']} | {s['count']} | {s['spam']} | {recall} | {s['fp']}/{s['ham']} |")
    h=a['holdout']
    rows+=['','## False positives',
           f"v65: {h['fp']}/{h['ham']} HAM; FPR {h['fpr']:.4%}; 95% Wilson upper bound {h['fprWilson95Upper']:.4%}.",
           f"Target FPR: {a['targetFpr']:.4%}. With zero FP, at least {a['minHAMForZeroFPtoBoundFPRbelowTarget95']:,} independent HAM are needed for a one-sided 95% binomial bound.",
           'Two 100000-email archive subsets are all SPAM; their FP is undefined, not zero-risk.',
           '',
           '## Acceptance',
           f"- Previously unused sources scored for v65: **{len(a['firstUseSources'])}**.",
           f"- Demonstrated target FPR: **{a['fpRequirementDemonstrated']}**.",
           '- Generalization to arbitrary new 100000-email mixes: **not demonstrated**.',
           '',
           'Next: collect an independent original-source corpus with its native HAM/SPAM labels; freeze provenance and exclusions before scoring. Maintain separate first-use report and do not re-label already-inspected data as independent.',
           '']
    return '\n'.join(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--v65',required=True);p.add_argument('--v64',required=True)
    p.add_argument('--provenance',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    records=[json.loads(Path(name).read_text()) for name in (args.v65,args.v64,args.provenance)]
    a=audit(*records)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    (root/'audit.json').write_text(json.dumps(a,indent=2,sort_keys=True))
    (root/'audit.md').write_text(report_md(a))
    print(report_md(a),flush=True)

if __name__=='__main__':
    main()
