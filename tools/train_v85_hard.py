"""One bounded English hard-example pilot; isolated from the ongoing v84 campaign."""
import argparse,hashlib,json,math,random
from pathlib import Path
import numpy as np,torch
from transformers import AutoModel,AutoTokenizer
from safetensors.torch import save_file
from v82_neural import NeuralMail,load_trainable,trainable_state,text_tokenizer,save_checkpoint,restore_checkpoint
from v84_replay import confusion,fp_budget_points,FIXED_THRESHOLD
from v84_replay_data import read_rows,fingerprint,sha
from train_v84_replay import backbone_hash
PARENT_STEP=42875
PARENT_SHA='3b77371c3d413a7dc90f66ae69d0744c93e1b5a4f17631e9517b51d5878e0f79'
def scores(model,tok,rows):
 model.eval();v=[]
 with torch.no_grad():
  for i in range(0,len(rows),32):v.extend(model(**text_tokenizer(tok,[r['text'] for r in rows[i:i+32]])).sigmoid().tolist())
 return np.asarray(v)
def metrics(rows,s):
 y=np.array([r['y'] for r in rows]);fixed=confusion(y,s,FIXED_THRESHOLD);cut=float(np.sort(s[y==1])[math.floor(.02*sum(y))]) if int(sum(y))>=100 else None
 return dict(partition='SELECT',development=True,independent_test=False,fixed_parent=fixed,fitted98=(confusion(y,s,cut) if cut is not None else None),fitted98_status=('development_fit' if cut is not None else 'insufficient_spam_support'),fitted98_selected_on_same_select=(cut is not None),fp_budget=fp_budget_points(y,s))
def hard_order(y,s,ids,steps):
 wrong=(s>=FIXED_THRESHOLD)!=np.asarray(y,dtype=bool)
 hard=sorted(np.flatnonzero(wrong).tolist(),key=lambda i:hashlib.sha256(('hard:'+ids[i]).encode()).hexdigest())
 easy=sorted(np.flatnonzero(~wrong).tolist(),key=lambda i:hashlib.sha256(('support:'+ids[i]).encode()).hexdigest())
 # Up to3 presentations for error rows; correct support once. No holdout scores.
 h=hard*3;e=easy;order=[];hi=ei=0
 while len(order)<steps*16 and (hi<len(h) or ei<len(e)):
  for _ in range(12):
   if hi<len(h):order.append(h[hi]);hi+=1
  for _ in range(4):
   if ei<len(e):order.append(e[ei]);ei+=1
  if hi==len(h) and ei<len(e):order.extend(e[ei:ei+max(0,steps*16-len(order))]);break
  if ei==len(e) and hi<len(h):order.extend(h[hi:hi+max(0,steps*16-len(order))]);break
 order=order[:steps*16]
 assert len(order) and max(np.bincount(order,minlength=len(ids)))<=3
 return order,wrong

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('.cache/v85/corpus'));p.add_argument('--parent',type=Path,required=True);p.add_argument('--backbone',type=Path,default=Path('.cache/v82/backbone'));p.add_argument('--state',type=Path,default=Path('hard-state/checkpoint.pt'));p.add_argument('--steps',type=int,default=100);a=p.parse_args()
 if a.steps!=100 or torch.cuda.is_available():raise ValueError('Pilot fixed to100 CPU steps')
 torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
 manifest=json.loads((a.data/'manifest.json').read_text());lock=json.loads(Path('research/v85/frozen.json').read_text())
 if manifest!=lock or not lock['training_ready']:raise ValueError('Data gate or frozen manifest failed')
 train=list(read_rows(a.data/'train.jsonl.gz'));select=list(read_rows(a.data/'select.jsonl.gz'))
 for k,rows in [('train',train),('select',select)]:
  if fingerprint(rows)!=lock['fingerprints'][k]:raise ValueError('Partition drift')
 if {r['split_group'] for r in train}&{r['split_group'] for r in select}:raise ValueError('Split leak')
 backbone=json.loads((a.backbone/'manifest.json').read_text())
 for n,h in backbone['files'].items():
  if sha(a.backbone/n)!=h:raise ValueError('Backbone drift')
 if sha(a.parent)!=PARENT_SHA:raise ValueError('Wrong pinned v84 parent')
 parent=torch.load(a.parent,weights_only=True);assert parent['progress']['step']==PARENT_STEP and parent['progress']['active_session'] is None
 model=NeuralMail(AutoModel.from_pretrained(a.backbone,local_files_only=True,attn_implementation='eager'));tok=AutoTokenizer.from_pretrained(a.backbone,local_files_only=True)
 opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad]);opt.load_state_dict(parent['optimizer'])
 for g in opt.param_groups:g.pop('initial_lr',None)
 sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:max(0.,1-s/100))
 metadata=dict(campaign='v85-en-hard-20261116',parent_sha256=PARENT_SHA,parent_global_step=PARENT_STEP,frozen=manifest,backbone=backbone,device='cpu',scheduler_explicitly_replanned_for_new100step_pilot=True,optimizer_and_rng_inherited=True)
 a.state.parent.mkdir(parents=True,exist_ok=True);out=a.state.parent;restored=a.state.exists()
 if restored:
  progress=restore_checkpoint(a.state,model,opt,sched,metadata)
 else:
  if (out/'migration.json').exists():raise ValueError('Started pilot lost its checkpoint; cold restart forbidden')
  load_trainable(model,parent['adapter']);torch.set_rng_state(parent['rng'])
  ts=scores(model,tok,train);ss=scores(model,tok,select)
  order,wrong=hard_order([r['y'] for r in train],ts,[r['identity'] for r in train],100)
  baseline=metrics(select,ss)
  np.savez_compressed(out/'baseline-select-scores.npz',scores=ss,y=[r['y'] for r in select],identity=[r['identity'] for r in select])
  mining=dict(train_rows=len(train),threshold=FIXED_THRESHOLD,errors=int(wrong.sum()),false_positive_ham=int((wrong & (np.array([r['y'] for r in train])==0)).sum()),false_negative_spam=int((wrong & (np.array([r['y'] for r in train])==1)).sum()),planned_presentations=len(order),planned_unique_rows=len(set(order)),planned_error_presentations=int(sum(wrong[i] for i in order)),planned_correct_support_presentations=int(sum(not wrong[i] for i in order)),selected_using_only_train_labels_and_scores=True,select_or_test_errors_used_for_training=False)
  (out/'mining.json').write_text(json.dumps(mining,indent=2)+'\n');(out/'baseline-development.json').write_text(json.dumps(baseline,indent=2)+'\n')
  progress=dict(step=PARENT_STEP,cursor=0,order=order,exposures=torch.zeros(len(train),dtype=torch.uint8),completed=False)
  save_checkpoint(a.state,model,opt,sched,progress,metadata)
  (out/'migration.json').write_text(json.dumps(dict(parent_sha256=PARENT_SHA,old_scheduler=parent['scheduler'],new_steps=100,scheduler_replanned=True),indent=2)+'\n')
  print(json.dumps(dict(mining=mining,baseline=baseline)),flush=True)
 initial=progress['step'];before=trainable_state(model);frozen=backbone_hash(model)
 while progress['cursor']<len(progress['order']) and progress['step']-PARENT_STEP<100:
  ix=progress['order'][progress['cursor']:progress['cursor']+16];batch=[train[i] for i in ix]
  model.train();opt.zero_grad(set_to_none=True);y=torch.tensor([r['y'] for r in batch],dtype=torch.float32);logits=model(**text_tokenizer(tok,[r['text'] for r in batch]));loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*torch.where(y==0,2.,1.)).mean()
  if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
  loss.backward();torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.,error_if_nonfinite=True);opt.step();sched.step()
  progress['cursor']+=len(ix);progress['step']+=1
  for i in ix:progress['exposures'][i]+=1
  if int(progress['exposures'].max())>3:raise ValueError('Exposure limit')
  save_checkpoint(a.state,model,opt,sched,progress,metadata)
  if (progress['step']-PARENT_STEP)%25==0:print(json.dumps(dict(global_step=progress['step'],pilot_steps=progress['step']-PARENT_STEP,presentations=progress['cursor'])),flush=True)
 ss=scores(model,tok,select);development=metrics(select,ss);after=trainable_state(model);changed=sum(not torch.equal(v,before[k]) for k,v in after.items() if k.endswith(('.a','.b')));changed_parent=sum(not torch.equal(v,parent['adapter'][k]) for k,v in after.items() if k.endswith(('.a','.b')))
 if backbone_hash(model)!=frozen or (progress['step']>initial and changed!=24):raise ValueError('LoRA/frozen backbone assertion failed')
 progress['completed']=True;save_checkpoint(a.state,model,opt,sched,progress,metadata);save_file(after,str(out/'candidate-adapter.safetensors'))
 report=dict(campaign=metadata['campaign'],status='completed',parent_step=PARENT_STEP,pilot_optimizer_steps=progress['step']-PARENT_STEP,invocation_optimizer_steps=progress['step']-initial,global_optimizer_steps=progress['step'],presentations=progress['cursor'],unique_train_rows=int((progress['exposures']>0).sum()),max_row_exposures=int(progress['exposures'].max()),resumed=restored,adam_and_rng_inherited=True,scheduler_replanned_once=True,changed_lora_matrices_invocation=changed,changed_lora_matrices_since_parent=changed_parent,completed_pilot_replayed=(progress['step']==initial),frozen_backbone_unchanged=True,development=development,baseline_development=json.loads((out/'baseline-development.json').read_text()),mining=json.loads((out/'mining.json').read_text()),test_opened=False,customer_mail_used=False,accepted_for_production=False,checkpoint_sha256=sha(a.state))
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(out/'candidate-select-scores.npz',scores=ss,y=[r['y'] for r in select],identity=[r['identity'] for r in select]);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
