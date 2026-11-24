"""One bounded repair trial, preserving both original campaigns unchanged.

No held-out benchmark is read. Only pinned TRAIN rows enter gradients. The
candidate replaces the retained parent adapter only after a SELECT gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer
from v82_neural import NeuralMail, load_trainable, trainable_state, text_tokenizer
from v84_replay_data import read_rows, fingerprint, sha
from v86_month import development
from v86_quality_guard import choose_recovery

SOURCE_SHA='56e8284f958dbe59bc7c90c590dbc3458d72fdaae31a4e6b78e6d67bd88a5b47'
TEACHER_SHA='2e62947281989573c47e322e2dd372a9d0690750f4490c9238f9f962ef17e685'
CAMPAIGN='v87-en-repair-20261124'


def prepare(v84,v86,parent,current):
    f84=json.loads((v84/'manifest.json').read_text());f86=json.loads((v86/'manifest.json').read_text())
    old=list(read_rows(v84/'train.jsonl.gz'));new=list(read_rows(v86/'train.jsonl.gz'))
    select=list(read_rows(v86/'select.jsonl.gz'))
    assert fingerprint(old)==f84['train_fingerprint']
    assert fingerprint(new)==f86['train_fingerprint']
    assert fingerprint(select)==f86['select_fingerprint']==f84['select_fingerprint']
    assert len(select)==22790
    assert current['progress']['step']==49500 and current['progress']['active_session'] is None
    assert parent['progress']['step']==46000 and parent['progress']['active_session'] is None
    # Today is already complete in v86. Do not replay its final 8000 TRAIN rows.
    cp=current['progress'];assert cp['epoch']==0 and cp['cursor']==56000
    used_today={int(i) for i in cp['order'][48000:56000]}
    groups={'old-spam':[],'new-spam':[],'ham':[]}
    for i,r in enumerate(old):
        e=int(parent['progress']['exposures'][i])
        if r['y']==1 and e<3:groups['old-spam'].append((r,e,True,'parent',i))
    for i,r in enumerate(new):
        e=int(cp['exposures'][i])
        if e<3 and i not in used_today:
            key='ham' if r['y']==0 else 'new-spam'
            groups[key].append((r,e,r['cohort']=='ham-support','source',i))
    picked=[]
    for group,count in [('old-spam',256),('new-spam',256),('ham',512)]:
        eligible=sorted(groups[group],key=lambda x:hashlib.sha256((CAMPAIGN+':'+x[0]['identity']).encode()).hexdigest())
        assert len(eligible)>=count
        picked.extend(eligible[:count])
    picked.sort(key=lambda x:hashlib.sha256((CAMPAIGN+':order:'+x[0]['identity']).encode()).hexdigest())
    assert len({x[0]['identity'] for x in picked})==1024
    assert not {x[0]['family'] for x in picked}&{r['family'] for r in select}
    return picked,select


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','teacher','v84','v86','backbone','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    assert sha(a.source/'checkpoint.pt')==SOURCE_SHA
    assert sha(a.teacher/'checkpoint.pt')==TEACHER_SHA
    current=torch.load(a.source/'checkpoint.pt',map_location='cpu',weights_only=True)
    parent=torch.load(a.teacher/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert current['metadata']['torch_version']==str(torch.__version__)
    assert current['metadata']['transformers_version']==transformers.__version__
    assert current['metadata']['backbone']==parent['metadata']['backbone']
    for name,digest in current['metadata']['backbone']['files'].items():assert sha(a.backbone/name)==digest
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
    picked,select=prepare(a.v84,a.v86,parent,current)
    manifest=dict(campaign=CAMPAIGN,source_sha256=SOURCE_SHA,teacher_sha256=TEACHER_SHA,
        rows=1024,ham=512,old_spam=256,new_spam=256,optimizer_steps=64,
        row_fingerprint=fingerprint([x[0] for x in picked]),max_lifetime_exposures=3,
        original_v86_date_replayed=False,training_partition='TRAIN only',benchmark_rows_used=0,
        loss='equal-class BCE + 1.0 KL to retained parent on old replay only; T=2',
        lr_scale=.5,scheduler='new 64-step linear decay; original scheduler preserved',
        source_training_date='2026-11-24',additional_trial_authorized_by_user_repair_request=True)
    mf=a.output/'manifest.json'
    if mf.exists():assert json.loads(mf.read_text())==manifest
    mf.write_text(json.dumps(manifest,indent=2)+'\n')
    tok=AutoTokenizer.from_pretrained(a.backbone,local_files_only=True)
    def model(state):
        m=NeuralMail(AutoModel.from_pretrained(a.backbone,local_files_only=True,
                         use_safetensors=True,attn_implementation='eager'))
        load_trainable(m,state);return m
    student=model(current['adapter']);teacher=model(parent['adapter']);teacher.eval();teacher.requires_grad_(False)
    opt=torch.optim.AdamW([v for v in student.parameters() if v.requires_grad],lr=1e-5)
    opt.load_state_dict(current['optimizer']);initial_lr=opt.param_groups[0]['lr']*.5
    for g in opt.param_groups:g['lr']=initial_lr;g.pop('initial_lr',None)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda k:max(0.,1-k/64))
    cp_path=a.output/'checkpoint.pt';before=trainable_state(student);cursor=0
    original_frozen={k:v.detach().clone() for k,v in student.named_parameters() if not v.requires_grad}
    torch.set_rng_state(current['rng'])
    if cp_path.exists():
        saved=torch.load(cp_path,map_location='cpu',weights_only=True)
        assert saved['metadata']['repair']==manifest
        load_trainable(student,saved['adapter']);opt.load_state_dict(saved['optimizer'])
        sched.load_state_dict(saved['scheduler']);torch.set_rng_state(saved['rng']);cursor=saved['progress']['repair_steps']
    invocation_start=cursor
    def save():
        # Original ledgers and cursor stay intact. Explicit per-identity deltas
        # record the additional branch presentations; no inherited count resets.
        checkpoint=dict(adapter=trainable_state(student),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
            rng=torch.get_rng_state(),cuda_rng=[],metadata=dict(repair=manifest,backbone=current['metadata']['backbone'],
            torch_version=str(torch.__version__),transformers_version=transformers.__version__,campaign=CAMPAIGN),
            progress=dict(step=49500+cursor,repair_steps=cursor,cursor=cursor*16,
                complete=cursor==64,source_original_progress=current['progress'],
                teacher_original_exposures=parent['progress']['exposures'],
                source_original_scheduler=current['scheduler'],source_original_metadata=current['metadata'],
                exposure_deltas=[dict(identity=x[0]['identity'],ledger=x[3],index=x[4],before=x[1],after=x[1]+1)
                                 for x in picked[:cursor*16]]))
        pending=cp_path.with_suffix('.partial');torch.save(checkpoint,pending);pending.replace(cp_path)
    student.train();save()
    while cursor<64:
        batch=picked[cursor*16:(cursor+1)*16]
        inp=text_tokenizer(tok,[x[0]['text'] for x in batch],192)
        y=torch.tensor([x[0]['y'] for x in batch],dtype=torch.float32)
        replay=torch.tensor([x[2] for x in batch],dtype=torch.bool)
        with torch.no_grad():q=teacher(**inp).sigmoid()
        opt.zero_grad(set_to_none=True);logits=student(**inp)
        hard=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
        # Binary KL(q || p), equivalent to soft BCE up to a constant; no
        # threshold-dependent error boost or repeated HAM-only cost multiplier.
        distill=torch.nn.functional.binary_cross_entropy_with_logits(logits[replay]/2,
                     torch.sigmoid(torch.logit(q[replay].clamp(1e-6,1-1e-6))/2))*4
        loss=hard+distill
        if not torch.isfinite(loss):raise ValueError('Nonfinite repair loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([v for v in student.parameters() if v.requires_grad],1.,error_if_nonfinite=True)
        opt.step();sched.step();cursor+=1
        if cursor%8==0:save();print(dict(repair_steps=cursor,global_optimizer_steps=49500+cursor,loss=float(loss.detach())),flush=True)
    save();assert all(torch.equal(v,original_frozen[k]) for k,v in student.named_parameters() if not v.requires_grad)
    assert all(int(s['step'])==49564 for s in opt.state_dict()['state'].values())
    evaluator=NeuralMail(AutoModel.from_pretrained(a.backbone,local_files_only=True,use_safetensors=True,attn_implementation='sdpa'))
    load_trainable(evaluator,trainable_state(student));evaluator.eval();student.eval()
    sample=text_tokenizer(tok,[r['text'] for r in select[:32]],192)
    with torch.no_grad():assert torch.max(torch.abs(student(**sample).sigmoid()-evaluator(**sample).sigmoid())).item()<1e-5
    torch.set_num_threads(2)
    metrics=development(evaluator,tok,select,text_tokenizer,score_cache=a.output/'select-scores.npz')
    baseline=json.loads((a.source/'best-development.json').read_text())
    selection=choose_recovery(baseline,metrics)
    retained=parent['adapter']
    assert all(torch.equal(retained[k],v) for k,v in __import__('safetensors.torch',fromlist=['load_file']).load_file(str(a.source/'best-adapter.safetensors')).items())
    selected=trainable_state(student) if selection['repair_passed'] else retained
    save_file(selected,str(a.output/'selected-adapter.safetensors'))
    save_file(trainable_state(student),str(a.output/'repair-adapter.safetensors'))
    report=dict(campaign=CAMPAIGN,status='completed',source_global_steps=49500,
        global_optimizer_steps=49564,phase_optimizer_steps=64,invocation_optimizer_steps=64-invocation_start,
        changed_lora_matrices=sum(not torch.equal(before[k],trainable_state(student)[k]) for k in before if k.endswith(('.a','.b'))),
        frozen_backbone_unchanged=True,optimizer_adam_rng_restored=True,source_scheduler_preserved=True,
        source_original_cursor_and_exposures_preserved=True,max_lifetime_exposures_after_repair=max(x[1]+1 for x in picked),
        original_source_checkpoint_unchanged=sha(a.source/'checkpoint.pt')==SOURCE_SHA,
        original_teacher_checkpoint_unchanged=sha(a.teacher/'checkpoint.pt')==TEACHER_SHA,
        baseline_development=baseline,repair_development=metrics,selection=selection,
        selected_development=metrics if selection['repair_passed'] else baseline,
        selected_adapter_sha256=sha(a.output/'selected-adapter.safetensors'),
        checkpoint_sha256=sha(cp_path),independent_test_performed=False,accepted_for_production=False,
        raw_mail_included=False,customer_mail_used=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
