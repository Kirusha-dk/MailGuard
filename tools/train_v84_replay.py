"""One exact, resumable daily replay target. Never promote a production model."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer

from v82_neural import NeuralMail, restore_checkpoint, save_checkpoint, text_tokenizer, trainable_state
from v84_replay_data import PARENT_SHA, PARENT_STEP, fingerprint, read_rows, sha
from v84_replay import (CONFIG, begin_session, check_request, development, migrate, run_steps)


def backbone_hash(model):
    import hashlib
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        if not value.requires_grad:
            h.update(name.encode()); h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--request', type=Path, default=Path('research/v84/cycle.json'))
    p.add_argument('--data', type=Path, default=Path('.cache/v84/corpus'))
    p.add_argument('--state', type=Path, default=Path('replay-state/checkpoint.pt'))
    p.add_argument('--parent', type=Path, default=Path('.cache/v84/warmstart/checkpoint.pt'))
    p.add_argument('--backbone', type=Path, default=Path('.cache/v82/backbone'))
    p.add_argument('--allow-initial-migration', action='store_true')
    p.add_argument('--max-steps-this-invocation', type=int, default=3125)
    args=p.parse_args(); request=json.loads(args.request.read_text()); check_request(request)
    if not 1<=args.max_steps_this_invocation<=3125: raise ValueError('Unbounded invocation')
    manifest=json.loads((args.data/'manifest.json').read_text())
    if manifest != json.loads(Path('research/v84/frozen.json').read_text()):
        raise ValueError('Frozen corpus drift')
    train=list(read_rows(args.data/'train.jsonl.gz')); select=list(read_rows(args.data/'select.jsonl.gz'))
    if (fingerprint(train)!=manifest['train_fingerprint'] or fingerprint(select)!=manifest['select_fingerprint'] or
            {r['family'] for r in train}&{r['family'] for r in select}):
        raise ValueError('Content drift or family leakage')
    if torch.cuda.is_available(): raise ValueError('CPU-only campaign')
    torch.set_num_threads(3); torch.manual_seed(84); torch.use_deterministic_algorithms(True)
    backbone=json.loads((args.backbone/'manifest.json').read_text())
    for name, digest in backbone['files'].items():
        if sha(args.backbone/name)!=digest: raise ValueError('Backbone drift')
    metadata=dict(config=CONFIG, backbone=backbone, corpus=manifest, device_type='cpu',
        torch_version=str(torch.__version__), transformers_version=transformers.__version__,
        cpu_threads=3, campaign=request['campaign'], start_date=request['start_date'], end_date=request['end_date'],
        parent_checkpoint_sha256=PARENT_SHA,
        migration='Inherit8500 steps, exact LoRA/head, Adam moments and RNG. Explicit new dataset/order and linear scheduler from parent current LR. Parent checkpoint stays immutable.')
    tokenizer=AutoTokenizer.from_pretrained(args.backbone,local_files_only=True)
    model=NeuralMail(AutoModel.from_pretrained(args.backbone,local_files_only=True,
                          use_safetensors=True,attn_implementation='eager'))
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],
                               lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    total=sum(math.ceil(sum(3-r['parent_exposures']>epoch for r in train)/16) for epoch in range(3))
    restored=args.state.is_file()
    if restored:
        saved=torch.load(args.state,map_location='cpu',weights_only=True)
        # Construct the exact scheduler before restoring optimizer/scheduler state.
        for group in optimizer.param_groups:
            group['lr']=saved['progress']['parent_learning_rate']
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:max(0.,1-step/max(1,total)))
        progress=restore_checkpoint(args.state,model,optimizer,scheduler,metadata)
    else:
        if (not args.allow_initial_migration or request['session_id']!=request['start_date'] or
                request['expected_min_step']!=PARENT_STEP):
            raise ValueError('Required continuation checkpoint missing; no silent restart')
        progress=migrate(args.parent,model,optimizer,train,metadata)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:max(0.,1-step/max(1,total)))
        metadata_parent_scheduler=torch.load(args.parent,map_location='cpu',weights_only=True)['scheduler']
        # Keep original scheduler bytes/state as evidence rather than pretending
        # its old three-epoch schedule can continue across changed data.
        args.state.parent.mkdir(parents=True,exist_ok=True)
        (args.state.parent/'migration.json').write_text(json.dumps(dict(
            parent_sha256=PARENT_SHA,parent_step=PARENT_STEP,optimizer_moments_inherited=True,
            rng_inherited=True,old_scheduler=metadata_parent_scheduler,
            scheduler_explicitly_replanned=True,new_total_scheduler_steps=total,
            parent_current_learning_rate=progress['parent_learning_rate']),indent=2)+'\n')
        save_file(trainable_state(model),str(args.state.parent/'best-adapter.safetensors'))
        (args.state.parent/'best-development.json').write_text(json.dumps(
            torch.load(args.parent,map_location='cpu',weights_only=True)['progress']['last_report']['development'],
            indent=2)+'\n')
    if progress['step']<request['expected_min_step']: raise ValueError('Checkpoint rollback')
    save=lambda:save_checkpoint(args.state,model,optimizer,scheduler,progress,metadata)
    if not begin_session(progress,request):
        cached=progress.get('last_report')
        if not cached: raise ValueError('Completed/stopped state lacks durable report')
        cached=dict(cached,invocation_optimizer_steps=0,completed_session_replayed=True,
                    checkpoint_restored_in_invocation=True,checkpoint_sha256=sha(args.state))
        (args.state.parent/'report.json').write_text(json.dumps(cached,indent=2)+'\n')
        print(json.dumps(cached,indent=2));return
    save(); before=trainable_state(model); frozen_before=backbone_hash(model); initial=progress['step']
    groups=Counter((r['source'],r['y']) for r in train)
    weights=[min(8.,math.sqrt(len(train)/(len(groups)*groups[r['source'],r['y']]))) *
             (2. if r['y']==0 else 1.) for r in train]
    mean=sum(weights)/len(weights);weights=[w/mean for w in weights]
    done=run_steps(model,tokenizer,train,weights,optimizer,scheduler,progress,text_tokenizer,
                   invocation_limit=args.max_steps_this_invocation,save=save)
    save()
    if frozen_before!=backbone_hash(model):raise ValueError('Frozen backbone changed')
    after=trainable_state(model);keys=[k for k in after if k.endswith(('.a','.b'))]
    report=dict(version=84,campaign=request['campaign'],session=request['session_id'],
        parent_checkpoint_sha256=PARENT_SHA,parent_optimizer_steps=PARENT_STEP,
        checkpoint_restored_in_invocation=restored,initial_state_migrated_from_v82=not restored,
        parent_adam_and_rng_inherited=True,scheduler_explicitly_replanned_for_new_corpus=True,
        global_optimizer_steps=progress['step'],phase_optimizer_steps=progress['step']-PARENT_STEP,
        invocation_optimizer_steps=progress['step']-initial,
        session_optimizer_steps=progress['step']-progress['active_start'],
        session_rows=progress['active_rows'],session_target_rows=progress['active_budget_rows'],
        session_ham_rows=progress['active_ham_rows'],session_spam_rows=progress['active_spam_rows'],
        session_unique_rows=len(set(progress['active_seen'])),session_complete=done,
        session_errors_observed_before_gradient=progress['active_wrong_rows'],
        error_counts_are_training_observations_not_test_metrics=True,
        mean_training_loss=progress['active_loss_sum']/max(1,progress['active_rows']),
        changed_internal_adapter_tensors=sum(not torch.equal(after[k],before[k]) for k in keys),
        internal_adapter_tensor_count=len(keys),frozen_backbone_unchanged=True,
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        train_rows=len(train),select_rows=len(select),phase_epoch=progress['epoch'],
        lifetime_max_row_exposures=int(progress['exposures'].max()),
        accepted_for_production=False,customer_mail_used=False,
        retired_tests_used_as_training=True,new_independent_test_performed=False,
        development=None,stopped=False)
    if done:
        metrics=development(model,tokenizer,select,text_tokenizer,
                            score_cache=args.state.parent/'select-scores.npz');report['development']=metrics
        improved=(metrics['fp'],metrics['log_loss'])<(progress['best_fp'],progress['best_loss'])
        if improved:
            progress.update(best_fp=metrics['fp'],best_loss=metrics['log_loss'],stalled=0)
            save_file(trainable_state(model),str(args.state.parent/'best-adapter.safetensors'))
            (args.state.parent/'best-development.json').write_text(json.dumps(metrics,indent=2)+'\n')
        else:progress['stalled']+=1
        progress['completed_sessions'].append(request['session_id'])
        progress['stopped']=progress['epoch']>=3 or (progress['epoch']>=1 and progress['stalled']>=CONFIG['patience'])
        progress.update(active_session=None,active_budget_rows=None,active_budget_steps=None,
                        active_start=None,active_seen=[])
        report.update(stopped=progress['stopped'],completed_sessions=len(progress['completed_sessions']),
                      best_development_fp=progress['best_fp'],early_stop_stalled=progress['stalled'])
        progress['last_report']=report; save()
    report['checkpoint_sha256']=sha(args.state)
    if done:
        (args.state.parent/'select-scores-metadata.json').write_text(json.dumps(dict(
            checkpoint_sha256=report['checkpoint_sha256'],
            global_optimizer_steps=progress['step'],session=request['session_id'],
            select_fingerprint=manifest['select_fingerprint'],is_development=True,
            is_independent_test=False,optimizer_steps_for_evaluation=0),indent=2)+'\n')
    (args.state.parent/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
