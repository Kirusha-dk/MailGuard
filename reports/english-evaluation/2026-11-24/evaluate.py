"""Read-only English mail benchmark. Inputs and thresholds frozen before scoring."""
import sys,json,time,hashlib,math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace',type=Path,required=True,help='Private local directory with manifest.json, test.jsonl.gz, v84/ and v86/ state')
parser.add_argument('--tools',type=Path,required=True,help='MailGuard tools directory')
parser.add_argument('--backbone',type=Path,required=True,help='Pinned local backbone directory')
args=parser.parse_args()
sys.path.insert(0,str(args.tools.resolve()))
from v82_neural import NeuralMail,load_trainable,text_tokenizer
from v84_replay_data import read_rows,fingerprint,sha
p=args.workspace;bb=args.backbone
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
manifest=json.loads((p/'manifest.json').read_text());rows=list(read_rows(p/'test.jsonl.gz'));assert fingerprint(rows)==manifest['fingerprint'];assert sha(p/'test.jsonl.gz')==manifest['gzip_sha256']
y=np.array([r['y'] for r in rows]);tokenizer=AutoTokenizer.from_pretrained(bb,local_files_only=True)
# Tokenize each batch once; no labels enter the model.
batches=[text_tokenizer(tokenizer,[r['text'] for r in rows[i:i+32]],192) for i in range(0,len(rows),32)]
def interval(k,n):
 if not n:return None
 z=1.959963984540054;d=1+z*z/n;c=(k/n+z*z/(2*n))/d;w=z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/d;return [c-w,c+w]
def metrics(scores,threshold,mask=None):
 mask=np.ones(len(y),dtype=bool) if mask is None else mask;truth=y[mask];pred=scores[mask]>=threshold
 tp=int(((truth==1)&pred).sum());fp=int(((truth==0)&pred).sum());fn=int(((truth==1)&~pred).sum());tn=int(((truth==0)&~pred).sum());s=tp+fn;h=tn+fp
 return dict(threshold=threshold,n=len(truth),spam=s,ham=h,tp=tp,fp=fp,fn=fn,tn=tn,spam_recall=tp/s if s else None,ham_specificity=tn/h if h else None,fpr=fp/h if h else None,precision=tp/(tp+fp) if tp+fp else None,spam_recall_wilson95=interval(tp,s),ham_specificity_wilson95=interval(tn,h))
result=dict(date='2026-11-24',cohort_rows=len(rows),cohort_fingerprint=manifest['fingerprint'],class_counts=manifest['classes'],source_counts=manifest['source_class'],optimizer_steps=0,threshold_fitting_performed=False,training_data_modified=False,raw_mail_published=False,models={})
for v,parent,run in [('v84',8500,45334794184),('v86',46000,45332932979)]:
 started=time.monotonic();cp_path=p/v/'checkpoint.pt';before=sha(cp_path);assert before==manifest['evaluation_models'][v]['checkpoint_sha256'];cp=torch.load(cp_path,map_location='cpu',weights_only=False)
 for filename,digest in cp['metadata']['backbone']['files'].items():assert sha(bb/filename)==digest,(v,filename)
 model=NeuralMail(AutoModel.from_pretrained(bb,local_files_only=True,use_safetensors=True,attn_implementation='sdpa'),cp['metadata']['config']);load_trainable(model,cp['adapter']);model.eval();model.requires_grad_(False)
 frozen_before={k:t.detach().clone() for k,t in model.state_dict().items()}
 eager=NeuralMail(AutoModel.from_pretrained(bb,local_files_only=True,use_safetensors=True,attn_implementation='eager'),cp['metadata']['config']);load_trainable(eager,cp['adapter']);eager.eval();eager.requires_grad_(False)
 with torch.inference_mode():
  reference=eager(**batches[0]).sigmoid().numpy();optimized=model(**batches[0]).sigmoid().numpy()
 max_backend_error=float(np.max(np.abs(reference-optimized)));assert max_backend_error<1e-5,max_backend_error
 print(v,'SDPA/eager maximum first-batch error',max_backend_error,flush=True)
 score_parts=[];done=0
 with torch.inference_mode():
  for inputs in batches:
   s=model(**inputs).sigmoid().numpy();assert np.isfinite(s).all();score_parts.append(s);done+=len(s)
   if done//1024>(done-len(s))//1024:print(v,done,'/',len(rows),'seconds',round(time.monotonic()-started),flush=True)
 scores=np.concatenate(score_parts)
 fitted=float(json.loads((p/v/'report.json').read_text())['development']['threshold'])
 near=np.where(np.minimum.reduce([np.abs(scores-t) for t in [0.04897100105881691,0.5,fitted]])<1e-5)[0]
 with torch.inference_mode():
  for offset in range(0,len(near),32):
   indices=near[offset:offset+32];inputs=text_tokenizer(tokenizer,[rows[int(i)]['text'] for i in indices],192);scores[indices]=eager(**inputs).sigmoid().numpy()
 np.savez_compressed(p/(v+'-scores.npz'),scores=scores,y=y)
 changed=[k for k,t in model.state_dict().items() if not torch.equal(t,frozen_before[k])];assert not changed;assert sha(cp_path)==before
 report=json.loads((p/v/'report.json').read_text());fitted=float(report['development']['threshold']);thresholds={ 'fixed_parent':0.04897100105881691,'fixed_0_5':0.5,'previous_SELECT_fitted98':fitted}
 summary=dict(checkpoint_sha256=before,run=f'https://github.com/Kirusha-dk/MailGuard/actions/runs/{run}',global_optimizer_steps=cp['progress']['step'],inherited_optimizer_steps=parent,phase_optimizer_steps=cp['progress']['step']-parent,training_stopped=cp['progress']['stopped'],model_revision=cp['metadata']['backbone']['revision'],internal_lora_matrices=sum(k.endswith(('.a','.b')) for k in cp['adapter']),attention_backend='CPU SDPA FP32; eager verification and eager rescoring within 1e-5 of every reported threshold',eager_equivalence_max_first_batch_error=max_backend_error,eager_rescored_near_threshold_rows=len(near),optimizer_steps_in_evaluation=0,changed_tensors_in_evaluation=changed,backbone_verified_against_checkpoint=True,checkpoint_unchanged=True,metrics={k:metrics(scores,t) for k,t in thresholds.items()},by_source={source:metrics(scores,thresholds['fixed_parent'],np.array([r['source']==source for r in rows])) for source in sorted({r['source'] for r in rows})},elapsed_seconds=round(time.monotonic()-started,2))
 result['models'][v]=summary;(p/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(v,'DONE',summary['metrics'],flush=True)
 del model,eager,cp,frozen_before
