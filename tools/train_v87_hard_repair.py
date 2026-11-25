"""Bounded hard-TRAIN replay with faster classifier-head adaptation; no TEST access."""
import argparse,hashlib,json,math
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from transformers import AutoModel,AutoTokenizer
from v82_neural import NeuralMail,load_trainable,trainable_state,text_tokenizer
from v84_replay_data import read_rows,fingerprint,sha
from v86_month import development,confusion
from v87_repair_objective import threshold_aware_loss, roll_forward_exposures


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 root=Path('/workspace');work=root/'scratch/v87-repair-20261124';bb=root/'mailguard-english-replay/.cache/v82/backbone'
 source_path=work/'state/checkpoint.pt';history=torch.load(source_path,map_location='cpu',weights_only=True)
 usage_path=root/'scratch/v87-repair-20261125/stage2/checkpoint.pt';usage=torch.load(usage_path,map_location='cpu',weights_only=True);assert usage['progress']['step']==49692
 source=history
 first=json.loads((work/'state/report.json').read_text());assert sha(source_path)==first['checkpoint_sha256'];assert source['progress']['step']==49564
 teacher_cp=torch.load(root/'mailguard-v86/.cache/v86/parent/checkpoint.pt',map_location='cpu',weights_only=True)
 source=teacher_cp
 third_path=root/'scratch/v87-repair-20261125/stage3/checkpoint.pt';third=torch.load(third_path,map_location='cpu',weights_only=True);assert third['progress']['step']==46128;source=third
 assert sha(root/'mailguard-v86/.cache/v86/parent/checkpoint.pt')=='2e62947281989573c47e322e2dd372a9d0690750f4490c9238f9f962ef17e685'
 for name,h in teacher_cp['metadata']['backbone']['files'].items():assert sha(bb/name)==h
 old=list(read_rows(root/'mailguard-english-replay/.cache/v84/corpus/train.jsonl.gz'))
 new=list(read_rows(root/'mailguard-v86/.cache/v86/corpus/train.jsonl.gz'))
 selected=list(read_rows(root/'mailguard-v86/.cache/v86/corpus/select.jsonl.gz'))
 v85root=root/'scratch/v85-en-20261116/corpus';v85=list(read_rows(v85root/'train.jsonl.gz'));dev85=list(read_rows(v85root/'select.jsonl.gz'))
 f85=json.loads((root/'mailguard-v85/research/v85/frozen.json').read_text());assert fingerprint(v85)==f85['fingerprints']['train'];assert fingerprint(dev85)==f85['fingerprints']['select']
 f84=json.loads((root/'mailguard-english-replay/research/v84/frozen.json').read_text());f86=json.loads((root/'mailguard-v86/research/v86/frozen.json').read_text());assert fingerprint(old)==f84['train_fingerprint'];assert fingerprint(new)==f86['train_fingerprint'];assert fingerprint(selected)==f86['select_fingerprint']
 fresh=list(read_rows(work/'fresh-train-screened/candidates.jsonl.gz'));assert fingerprint(fresh)=='088410fbbb98225bfb6cd880da8c2ff21b4a203942a16e4578197278b153ced1'
 base=history['progress']['source_original_progress'];counts={r['identity']:int(teacher_cp['progress']['exposures'][i]) for i,r in enumerate(old)}
 counts.update({r['identity']:int(base['exposures'][i]) for i,r in enumerate(new)})
 used=set()
 deltas=history['progress']['exposure_deltas']+usage['progress']['exposure_deltas']+third['progress']['exposure_deltas']
 counts=roll_forward_exposures(counts,deltas);used.update(d['identity'] for d in deltas)
 used.update(new[int(i)]['identity'] for i in base['order'][48000:56000])
 torch.set_num_threads(2)
 miner=NeuralMail(AutoModel.from_pretrained(bb,local_files_only=True,use_safetensors=True,attn_implementation='sdpa'));load_trainable(miner,source['adapter']);miner.eval();tok_m=AutoTokenizer.from_pretrained(bb,local_files_only=True)
 def mine(pool):
  values={}
  with torch.inference_mode():
   for start in range(0,len(pool),32):
    batch=pool[start:start+32];scores=miner(**text_tokenizer(tok_m,[r['text'] for r in batch],192)).sigmoid().tolist()
    values.update({r['identity']:v for r,v in zip(batch,scores)})
  return values
 mining={**mine(v85),**mine(fresh)};del miner
 groups=[('old-spam',[r for r in old if r['y']==1],384,True),('fresh-2026',fresh,128,False),('new-phishing',[r for r in v85 if r['y']==1],256,False),('old-ham',[r for r in new if r['y']==0 and r['cohort']=='ham-support'],256,True),('new-ham',[r for r in v85 if r['y']==0],512,False)]
 rows=[]
 for group,pool,n,teacher in groups:
  pool=[r for r in pool if counts.get(r['identity'],0)<3]
  if group in ('fresh-2026','new-phishing','new-ham'):
   # Source labels remain fixed. Only TRAIN scores drive hard-example order.
   pool.sort(key=lambda r:((-mining[r['identity']] if r['y']==0 else mining[r['identity']]),r['identity']))
  else:pool.sort(key=lambda r:hashlib.sha256(('v87-hard-20261125:'+group+':'+r['identity']).encode()).hexdigest())
  assert len(pool)>=n,(group,len(pool),n)
  rows.extend([(r,counts.get(r['identity'],0),teacher,group) for r in pool[:n]])
 rows.sort(key=lambda x:hashlib.sha256(('v87-domain-order:'+x[0]['identity']).encode()).hexdigest())
 assert len(rows)==1536 and len({x[0]['identity'] for x in rows})==1536 and sum(x[0]['y'] for x in rows)==768
 assert not {x[0]['family'] for x in rows}&{r['family'] for r in selected+dev85}
 manifest=dict(campaign='v87-en-repair-20261124',stage='hard-train-head-adapt-20261125',session_id='2026-11-25',source_checkpoint_sha256=sha(third_path),source_global_step=46128,usage_ledger_sha256=sha(third_path),minimum_accounted_work_steps=49820,discarded_update_work_preserved=3692,budget_steps=96,budget_rows=1536,train_fingerprint=fingerprint([x[0] for x in rows]),sources={k:n for k,_,n,_ in groups},new_v85_train_fingerprint=f85['fingerprints']['train'],fresh_train_fingerprint=fingerprint(fresh),primary_development='old SELECT22790',additional_development='v85 SELECT699',benchmark_used_for_training_or_selection=False,lora_lr=.00001,head_lr=.0005,teacher_regularization=4.,hard_mining_partition='TRAIN only',threshold_aware_new_domain_loss=True,hard_labels_unchanged=True,scheduler_explicitly_replanned_once=True,lifetime_limit=3)
 file=a.output/'manifest.json';cpfile=a.output/'checkpoint.pt'
 if file.exists() and not cpfile.exists():raise ValueError('Lost stage2 state; no cold restart')
 if file.exists():assert json.loads(file.read_text())==manifest
 file.write_text(json.dumps(manifest,indent=2)+'\n')
 torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
 def build(adapter,backend='eager'):
  m=NeuralMail(AutoModel.from_pretrained(bb,local_files_only=True,use_safetensors=True,attn_implementation=backend));load_trainable(m,adapter);return m
 student=build(source['adapter']);teacher=build(teacher_cp['adapter']);teacher.eval();teacher.requires_grad_(False);tok=AutoTokenizer.from_pretrained(bb,local_files_only=True)
 trainables=[(name,v) for name,v in student.named_parameters() if v.requires_grad]
 lora=[v for name,v in trainables if not name.startswith('head.')];head=[v for name,v in trainables if name.startswith('head.')]
 assert len(lora)==24 and len(head)==2
 old=source['optimizer'];assert len(old['param_groups'])==1
 ids=old['param_groups'][0]['params'];assert len(ids)==len(trainables)
 groups_new=[]
 for is_head,lr in [(False,.00001),(True,.0005)]:
  g=dict(old['param_groups'][0]);g['params']=[i for (name,_),i in zip(trainables,ids) if name.startswith('head.')==is_head];g['lr']=lr;g.pop('initial_lr',None);groups_new.append(g)
 opt=torch.optim.AdamW([dict(params=lora,lr=.00001),dict(params=head,lr=.0005)])
 opt.load_state_dict(dict(state=old['state'],param_groups=groups_new))
 assert all(int(x['step'])==46128 for x in opt.state_dict()['state'].values())
 scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda k:max(0.,1-k/96));torch.set_rng_state(source['rng']);cursor=0
 if cpfile.exists():
  c=torch.load(cpfile,map_location='cpu',weights_only=True);assert c['metadata']['manifest']==manifest;load_trainable(student,c['adapter']);opt.load_state_dict(c['optimizer']);scheduler.load_state_dict(c['scheduler']);torch.set_rng_state(c['rng']);cursor=c['progress']['stage_steps'];assert c['progress']['step']==46128+cursor;assert c['progress']['accounted_work_step']==49820+cursor
 initial=cursor;before=trainable_state(student);frozen={k:v.detach().clone() for k,v in student.named_parameters() if not v.requires_grad}
 def save():
  c=dict(adapter=trainable_state(student),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=torch.get_rng_state(),cuda_rng=[],metadata=dict(campaign=manifest['campaign'],manifest=manifest,backbone=teacher_cp['metadata']['backbone']),progress=dict(step=46128+cursor,accounted_work_step=49820+cursor,stage_steps=cursor,cursor=cursor*16,source_original_progress=source['progress'],usage_original_progress=third['progress'],source_original_scheduler=source['scheduler'],exposure_deltas=[dict(identity=x[0]['identity'],before=x[1],after=x[1]+1,group=x[3]) for x in rows[:cursor*16]]));pending=cpfile.with_suffix('.partial');torch.save(c,pending);pending.replace(cpfile)
 student.train();save()
 while cursor<96:
  b=rows[cursor*16:(cursor+1)*16];inp=text_tokenizer(tok,[x[0]['text'] for x in b],192);y=torch.tensor([x[0]['y'] for x in b],dtype=torch.float32);anchor=torch.tensor([x[2] for x in b],dtype=torch.bool)
  with torch.no_grad():t=teacher(**inp)
  opt.zero_grad(set_to_none=True);s=student(**inp);loss=threshold_aware_loss(s,y,anchor)
  if anchor.any():loss+=4*4*torch.nn.functional.binary_cross_entropy_with_logits(s[anchor]/2,(t[anchor]/2).sigmoid())
  assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([x for x in student.parameters() if x.requires_grad],1.,error_if_nonfinite=True);opt.step();scheduler.step();cursor+=1
  if cursor%8==0:save();print(dict(stage_steps=cursor,model_lineage_step=46128+cursor,accounted_work_step=49820+cursor,loss=float(loss.detach())),flush=True)
 save();assert all(torch.equal(v,frozen[k]) for k,v in student.named_parameters() if not v.requires_grad)
 assert all(int(x['step'])==46224 for x in opt.state_dict()['state'].values())
 evaluator=build(trainable_state(student),'sdpa');evaluator.eval();torch.set_num_threads(2)
 metrics=development(evaluator,tok,selected,text_tokenizer,score_cache=a.output/'select-scores.npz')
 teacher=build(teacher_cp['adapter'],'sdpa');teacher.eval()
 def scores(m,rs):
  values=[]
  with torch.inference_mode():
   for start in range(0,len(rs),32):values.extend(m(**text_tokenizer(tok,[r['text'] for r in rs[start:start+32]],192)).sigmoid().tolist())
  return values
 y=[r['y'] for r in dev85];ts=scores(teacher,dev85);ss=scores(evaluator,dev85);parent=json.loads((work/'state/report.json').read_text())['baseline_development']
 pnew=confusion(y,ts,.04897100105881691);snew=confusion(y,ss,.04897100105881691);pf=confusion(y,ts,parent['threshold']);sf=confusion(y,ss,metrics['threshold'])
 # A declared two-domain gate: bounded old-domain tradeoff and a substantial
 # new-domain HAM gain, without fitting thresholds on either external TEST.
 passed=(metrics['fp']<=math.ceil(parent['fp']*1.2) and metrics['fixed_parent_threshold']['tp']>=9733 and metrics['fixed_parent_threshold']['fp']<=579 and metrics['log_loss']<=.205 and snew['fp']<=math.floor(pnew['fp']*.6) and snew['tp']>=pnew['tp']-2 and sf['fp']<=math.floor(pf['fp']*.6) and sf['tp']>=pf['tp']-2)
 selected_state=trainable_state(student) if passed else teacher_cp['adapter'];save_file(selected_state,str(a.output/'selected-adapter.safetensors'));save_file(trainable_state(student),str(a.output/'repair-adapter.safetensors'))
 report=dict(status='completed',manifest=manifest,invocation_optimizer_steps=96-initial,stage_optimizer_steps=96,total_repair_optimizer_steps=416,global_optimizer_steps=46224,accounted_optimizer_work_steps=49916,discarded_optimizer_work_steps=3692,explicit_matched_weights_adam_rng_restore=True,usage_counters_rolled_forward_without_reset=True,source_checkpoint_unchanged=sha(third_path)==manifest['source_checkpoint_sha256'],frozen_backbone_unchanged=True,max_lifetime_exposures=max(x[1]+1 for x in rows),changed_lora_matrices=sum(not torch.equal(before[k],trainable_state(student)[k]) for k in before if k.endswith(('.a','.b'))),development=metrics,additional_development=dict(partition='v85 SELECT development',parent_fixed=pnew,candidate_fixed=snew,parent_at_own_old_select_threshold=pf,candidate_at_own_old_select_threshold=sf),retained_parent_development=parent,selection=dict(selected='domain-repair' if passed else 'retained-parent',repair_passed=passed,external_benchmark_used_for_selection=False,policy='old fitted98 FP <=1.2x parent; old fixed TP>=9733/FP<=579/logloss<=.205; new fixed and calibrated FP reduced40%, TP loss<=2'),selected_development=metrics if passed else parent,selected_adapter_sha256=sha(a.output/'selected-adapter.safetensors'),checkpoint_sha256=sha(cpfile),accepted_for_production=False)
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
