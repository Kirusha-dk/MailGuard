"""Verify text-free paired score caches against counts and the published v80 baseline.

This performs arithmetic on existing scores only: no model, inference or fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=Path('.cache/v82-v80-comparison'))
    parser.add_argument('--report',type=Path,help='Optional full comparison report from Git')
    parser.add_argument('--request',type=Path,default=Path('research/v82/v80-comparison.json'))
    parser.add_argument('--v80-report',type=Path,default=Path('reports/v80-result.json'))
    args=parser.parse_args()
    if not __debug__:
        raise RuntimeError('Verification requires Python assertions enabled')
    root=args.cache
    report_path=args.report or root/'comparison.json'
    request=json.loads(args.request.read_text())
    published=json.loads(args.v80_report.read_text())
    count_keys=('n','spam','ham','tp','fp','tn','fn')
    transition_keys=('baseline_spam_detections_lost','spam_detections_gained','false_positives_removed','false_positives_added')
    verified=[]
    for path in sorted(root.glob('*.npz')):
     name=path.stem
     report=json.loads((root/(name+'.json')).read_text())
     with np.load(path,allow_pickle=False) as cache:
      assert set(cache.files)=={'identity','y','v80','v82'}
      ids,y,a,b=(cache[k] for k in ('identity','y','v80','v82'))
      assert len(ids)==len(set(ids.tolist()))==len(y)==len(a)==len(b)
      assert all(len(i)==64 and all(c in '0123456789abcdef' for c in i) for i in ids.tolist())
      assert set(y.tolist())<={0,1} and np.isfinite(a).all() and np.isfinite(b).all()
      assert ((b>=0)&(b<=1)).all()
      old,new=a>=request['v80_threshold'],b>=request['v82_threshold']
      spam,ham=y==1,y==0
      for key,detected in [('baseline',old),('current',new)]:
       counts=dict(n=len(y),spam=int(spam.sum()),ham=int(ham.sum()),tp=int((spam&detected).sum()),
                   fp=int((ham&detected).sum()),tn=int((ham&~detected).sum()),fn=int((spam&~detected).sum()))
       assert all(report[key][k]==counts[k] for k in count_keys),(name,key,counts)
      counts=dict(baseline_spam_detections_lost=int((spam&old&~new).sum()),
                  spam_detections_gained=int((spam&~old&new).sum()),
                  false_positives_removed=int((ham&old&~new).sum()),
                  false_positives_added=int((ham&~old&new).sum()))
      assert all(report[k]==v for k,v in counts.items())
      assert report['first_lost_spam_identities']==ids[spam&old&~new].tolist()[:30]
      assert report['score_cache_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
      if name.startswith('block-'):
       original=published['regression_blocks'][int(name[-2:])-1]
       expected=original['metrics']['recall-98']
       assert all(report['baseline'][k]==expected[k] for k in count_keys),(name,'published baseline')
      else:
       expected=published['fresh_2024']['v80']['recall-98']
       assert all(report['baseline'][k]==expected[k] for k in count_keys)
      verified.append(name)
    if report_path.exists():
     full=json.loads(report_path.read_text())
     assert full['request']==request
     assert len(verified)==11
     for version in ('baseline','current'):
      for key in count_keys:
       assert full['frozen_100000'][version][key]==sum(full['cohorts'][name][version][key] for name in verified if name.startswith('block-'))
     for key in transition_keys:
      assert full['frozen_100000'][key]==sum(full['cohorts'][name][key] for name in verified if name.startswith('block-'))
     for source,r in full['frozen_100000']['by_source'].items():
      expected=published['regression_by_source'][source]['recall-98']
      assert all(r['baseline'][k]==expected[k] for k in count_keys)
     assert full['checkpoint_and_model_files_unchanged'] and full['optimizer_steps_executed']==0
     result=dict(verification_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),verified_cohorts=verified,numpy_cache_counts_and_individual_transitions_reproduced=True,
                 published_v80_per_block_and_aggregate_source_counts_reproduced=True,score_cache_has_only_hash_identities_labels_and_scores=True,
                 all_11_cohorts_complete=True,comparison_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest())
     (root/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(verified_cohorts=verified,complete=report_path.exists())))


if __name__ == '__main__':
    main()
