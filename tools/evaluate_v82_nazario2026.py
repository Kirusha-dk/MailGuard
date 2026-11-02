"""Read-only 2026 phishing-period evaluation; locked v82 step7000, no HAM claims."""
import gzip, hashlib, json, mailbox, urllib.request
from pathlib import Path
from collections import Counter
import fasttext
from audit_v82_errors import metric, sha
from compare_v82_v80 import initialize_scorer, score_chunk
from v77_mail import parse_eml
from v79_data import normalize, LID_SHA, english_prediction
from v78_data import identifiers
from v82_neural import fingerprint
LOCK='a188ef8ba238993e4be795836e502f53003089995f8eeec4d55992599f7b531d'
CUTOFF=0.03412362188100815
def main():
    out=Path('external-output');out.mkdir(exist_ok=False)
    state=Path('external-state/checkpoint.pt')
    if sha(state)!=LOCK: raise ValueError('Wrong locked checkpoint')
    report=json.loads(Path('external-state/report.json').read_text())
    if report['global_optimizer_steps']!=7000 or report['development']['threshold']!=CUTOFF:
        raise ValueError('Wrong source report')
    bb=Path('.cache/v82/backbone')
    manifest=json.loads((bb/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if sha(bb/name)!=expected: raise ValueError('Backbone drift')
    import torch
    cp=torch.load(state,map_location='cpu',weights_only=True)
    train=[]; select=[]; prior=set(); priorids=set()
    for root in (Path('.cache/v79/frozen'),Path('.cache/v80/extra')):
        for part in ('train','select','cal','test','fresh-test'):
            path=root/(part+'.jsonl.gz')
            if not path.exists(): continue
            with gzip.open(path,'rt') as stream:
                for line in stream:
                    r=json.loads(line);prior.add(r['family']);priorids.add(r['identity'])
                    if part=='train':train.append(r)
                    if part=='select':select.append(r)
    if cp['metadata']['train']!=fingerprint(train) or cp['metadata']['select']!=fingerprint(select):
        raise ValueError('Original corpus mismatch')
    if cp['progress']['step']!=7000 or cp['progress']['active_session'] is not None:
        raise ValueError('Checkpoint is not completed step7000')
    raw=Path('.cache/external-2026');raw.mkdir(parents=True,exist_ok=True)
    url='https://monkey.org/~jose/phishing/phishing-2026'
    with urllib.request.urlopen(url,timeout=90) as response:
        data=response.read(50_000_001)
    if not data or len(data)>50_000_000:raise ValueError('Invalid source download size')
    source_sha=hashlib.sha256(data).hexdigest()
    (raw/'source.mbox').write_bytes(data)
    lidpath=raw/'lid.176.ftz'
    with urllib.request.urlopen('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz',timeout=90) as response:
        lidpath.write_bytes(response.read(2_000_000))
    if sha(lidpath)!=LID_SHA:raise ValueError('Language model drift')
    lid=fasttext.load_model(str(lidpath))
    counts=Counter();rows=[];seen=set()
    box=mailbox.mbox(raw/'source.mbox',create=False)
    for mail in box:
        counts['raw']+=1
        text=normalize(parse_eml(mail.as_bytes()))
        if not 30<=len(text)<=250000:counts['length_excluded']+=1;continue
        if not english_prediction(lid,text)[0]:counts['language_excluded']+=1;continue
        identity,family=identifiers(text)
        if identity in seen:counts['duplicate_excluded']+=1;continue
        seen.add(identity)
        if identity in priorids or family in prior:counts['prior_overlap_excluded']+=1;continue
        rows.append(dict(identity=identity,family=family,text=text,y=1,source='Nazario-2026'))
    box.close()
    if not rows:raise ValueError('No eligible new English messages')
    # Freeze input identity set and source bytes hash before any model inference.
    lock=dict(checkpoint_sha256=LOCK,threshold=CUTOFF,source_url=url,source_sha256=source_sha,
              source_bytes=len(data),cohort_fingerprint=fingerprint(rows),counts=dict(counts),eligible=len(rows))
    (out/'input-lock.json').write_text(json.dumps(lock,indent=2))
    initialize_scorer(bb,state,LOCK)
    scores=score_chunk([r['text'] for r in rows])
    measured=metric(rows,scores,CUTOFF)
    result=dict(input_lock=lock,metrics=measured,optimizer_steps_executed=0,
                checkpoint_unchanged=sha(state)==LOCK,is_phishing_only=True,
                false_positive_rate_evaluable=False,new_time_period=True,new_publisher=False,
                independent_publisher_test=False,threshold_refitted=False,accepted_for_production=False)
    import numpy as np
    np.savez_compressed(out/'scores.npz',identity=np.array([r['identity'] for r in rows]),score=np.array(scores))
    (out/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
