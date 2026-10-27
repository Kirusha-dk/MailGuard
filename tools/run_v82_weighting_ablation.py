"""Bounded, separately named forks from verified checkpoint500; SELECT only."""
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

PARENT_SHA = 'a5cafb2fface8a35c672e075e41e63fe4551a786b536e4e41a23026299b63f68'
EXPERIMENT = 'v82-source-weighting-20261027'
TRAIN_SHA = '6958e55f52c5b18eb74c85b2e66f796b3a4ef826767487bf9d87066e1f4aee5b'
SELECT_SHA = '88acb130d5c5174823ccea518339fd63cceef356945a94225e184ef04110892f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_for(rows, arm):
    if arm not in ('control', 'candidate') or not rows:
        raise ValueError('Invalid arm or empty TRAIN')
    groups = Counter((r['source'], r['y']) for r in rows)
    weights = [(min(12., 5000 / groups[r['source'], r['y']]) if arm == 'control' else 1.)
               * (2. if r['y'] == 0 else 1.) for r in rows]
    mean = sum(weights) / len(weights)
    return [w / mean for w in weights]


def fork_progress(parent):
    if (parent['step'] != 500 or parent['cursor'] != 8000 or parent['epoch'] != 0 or
            parent['active_session'] is not None or parent['active_target'] is not None or
            parent['stopped']):
        raise ValueError('Requires completed parent500 with original cursor')
    progress = copy.deepcopy(parent)
    progress.update(completed_sessions=[], active_session=EXPERIMENT, active_start=500,
                    active_target=1000, last_report=None, best_fp=None, best_loss=None,
                    stalled=0, stopped=False)
    return progress


def main():
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    from audit_v82_errors import describe, paired
    from train_v82_daily import read_part
    from v79_data import fingerprint as old_fingerprint
    from v82_neural import (CONFIG, NeuralMail, fingerprint, optimizer_and_scheduler,
                            restore_checkpoint, run_steps, save_checkpoint, trainable_state)
    from v82_weighting_scores import evaluate

    out = Path('ablation-output'); out.mkdir(exist_ok=True)
    parent = Path('ablation-parent/checkpoint.pt')
    if sha(parent) != PARENT_SHA:
        raise ValueError('Parent checkpoint hash mismatch')
    request = json.loads(Path('research/v82/experiments/source-weighting-run.json').read_text())
    if (request['experiment'] != EXPERIMENT or request['parent_sha256'] != PARENT_SHA or
            request['optimizer_steps_per_arm'] != 500 or request['arms'] != ['control', 'candidate']):
        raise ValueError('Experiment request drift')
    torch.set_num_threads(3); torch.manual_seed(82); torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        raise ValueError('CPU only')
    backbone_path = Path('.cache/v82/backbone')
    manifest = json.loads((backbone_path / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if sha(backbone_path / name) != expected:
            raise ValueError('Backbone hash mismatch')
    train, select = [], []
    for directory, reference, key in (
            (Path('.cache/v79/frozen'), 'reports/v79-frozen.json', 'development_fingerprints'),
            (Path('.cache/v80/extra'), 'reports/v80-extra-frozen.json', 'fingerprints')):
        frozen = json.loads(Path(reference).read_text())
        if json.loads((directory / 'frozen.json').read_text()) != frozen:
            raise ValueError('Frozen manifest mismatch')
        for name, output in [('train', train), ('select', select)]:
            rows = read_part(directory, name)
            if old_fingerprint(rows) != frozen[key][name]:
                raise ValueError('Frozen partition fingerprint mismatch')
            output.extend(rows)
    if (len(train) != 90680 or len(select) != 22790 or fingerprint(train) != TRAIN_SHA or
            fingerprint(select) != SELECT_SHA or
            {r['family'] for r in train} & {r['family'] for r in select}):
        raise ValueError('Combined TRAIN/SELECT drift or overlap')
    expected_meta = dict(config=CONFIG, backbone=manifest, device_type='cpu',
                         torch_version=str(torch.__version__), transformers_version=transformers.__version__,
                         cpu_threads=3, campaign='v82-english-20261020', start_date='2026-10-20',
                         end_date='2026-11-18', train=TRAIN_SHA, select=SELECT_SHA, pilot_only=False)
    tokenizer = AutoTokenizer.from_pretrained(backbone_path, local_files_only=True)

    def make():
        backbone = AutoModel.from_pretrained(backbone_path, local_files_only=True,
                                             use_safetensors=True, attn_implementation='eager')
        model = NeuralMail(backbone)
        optimizer, scheduler = optimizer_and_scheduler(model, math.ceil(len(train) / 16) * 3)
        return model, optimizer, scheduler

    model, optimizer, scheduler = make()
    original = restore_checkpoint(parent, model, optimizer, scheduler, expected_meta)
    fork_progress(original)  # Validate lineage before any optimization.
    parent_metric, parent_scores = evaluate(model, tokenizer, select)
    historical = json.loads(Path('reports/v82-daily/2026-10-20.json').read_text())['development']
    differences = {k: dict(expected=historical[k], observed=parent_metric[k])
                   for k in ('tp', 'fp', 'n', 'spam', 'ham', 'threshold')
                   if parent_metric[k] != historical[k]}
    replay = dict(parent_sha256=PARENT_SHA, observed=parent_metric, historical=historical,
                  differences=differences, cpu_capability=torch.backends.cpu.get_cpu_capability(),
                  mkldnn_enabled=torch.backends.mkldnn.enabled,
                  float32_matmul_precision=torch.get_float32_matmul_precision(),
                  optimizer_steps_executed=0, parent_checkpoint_unchanged=sha(parent) == PARENT_SHA)
    (out / 'parent-replay.json').write_text(json.dumps(replay, indent=2))
    np.savez_compressed(out / 'parent-scores.npz', identity=np.array([r['identity'] for r in select]),
                        y=np.array([r['y'] for r in select]), score=np.array(parent_scores))
    print(json.dumps(dict(parent_replay_differences=differences,
                          cpu_capability=replay['cpu_capability'], optimizer_steps_executed=0)), flush=True)
    if request.get('diagnostic_only', False):
        print('Read-only parent replay complete; neither arm trained', flush=True)
        return
    if any(parent_metric[k] != historical[k] for k in ('tp', 'fp', 'n', 'spam', 'ham', 'threshold')):
        raise ValueError('Parent SELECT counts or threshold do not reproduce')
    metrics, scores = {'parent': parent_metric}, {'parent': parent_scores}
    del model, optimizer, scheduler
    frozen_hashes = {p: sha(p) for p in ['tools/v82_neural.py', 'tools/train_v82_daily.py',
                     'research/v82/cycle.json']}
    for arm in request['arms']:
        arm_dir = out / arm; arm_dir.mkdir(exist_ok=True)
        state = arm_dir / 'checkpoint.pt'
        model, optimizer, scheduler = make()
        progress = restore_checkpoint(parent, model, optimizer, scheduler, expected_meta)
        metadata = dict(expected_meta, campaign=EXPERIMENT,
                        experiment=dict(id=EXPERIMENT, arm=arm, parent_sha256=PARENT_SHA,
                                        parent_run_id=39332953339, parent_step=500,
                                        changed_objective=arm == 'candidate', budget_steps=500,
                                        source_balance=arm == 'control', ham_cost=2))
        if state.exists():
            progress = restore_checkpoint(state, model, optimizer, scheduler, metadata)
            if (not 500 <= progress['step'] <= 1000 or progress['active_target'] != 1000 or
                    progress['active_start'] != 500 or progress['active_session'] != EXPERIMENT or
                    progress['cursor'] != progress['step'] * 16 or progress['epoch'] != 0):
                raise ValueError('Fork progress or budget drift')
        else:
            progress = fork_progress(progress)
        initial_step = progress['step']; before = trainable_state(model)
        save = lambda: save_checkpoint(state, model, optimizer, scheduler, progress, metadata)
        save()
        print(json.dumps(dict(experiment=EXPERIMENT, arm=arm, training_started=True,
                              restored_step=initial_step, target=1000)), flush=True)
        run_steps(model, tokenizer, train, weights_for(train, arm), optimizer, scheduler, progress, save=save)
        save()  # Preserve the original 1000-step target even if validation is interrupted.
        if progress['step'] != 1000 or progress['cursor'] != 16000:
            raise ValueError('Fork did not complete original budget')
        metric, values = evaluate(model, tokenizer, select)
        metrics[arm], scores[arm] = metric, values
        after = trainable_state(model)
        changes = sum(not torch.equal(before[k], after[k]) for k in after if k.endswith(('.a', '.b')))
        if initial_step < 1000 and changes == 0:
            raise ValueError('No internal adapter changes')
        report = dict(experiment=EXPERIMENT, arm=arm, parent_sha256=PARENT_SHA,
                      global_optimizer_steps=progress['step'], fork_optimizer_steps=500,
                      invocation_optimizer_steps=progress['step'] - initial_step,
                      parent_optimizer_scheduler_rng_restored=True, changed_internal_adapter_tensors=changes,
                      metadata=metadata, development=metric, checkpoint_sha256=sha(state),
                      accepted_for_production=False)
        (arm_dir / 'report.json').write_text(json.dumps(report, indent=2))
        np.savez_compressed(arm_dir / 'scores.npz', identity=np.array([r['identity'] for r in select]),
                            y=np.array([r['y'] for r in select]), score=np.array(values))
        del model, optimizer, scheduler

    control_ref = json.loads(Path('reports/v82-daily/2026-10-21.json').read_text())['development']
    reproduced = all(metrics['control'][k] == control_ref[k] for k in ('n', 'spam', 'ham', 'tp', 'fp', 'threshold'))
    result = dict(experiment=EXPERIMENT, parent_sha256=PARENT_SHA,
                  optimizer_steps_per_arm=500, metrics=metrics,
                  control_historical_counts_and_threshold_reproduced=reproduced,
                  parent_historical_log_loss=historical['log_loss'],
                  parent_replayed_log_loss=parent_metric['log_loss'],
                  parent_log_loss_replay_delta=parent_metric['log_loss'] - historical['log_loss'],
                  parent_checkpoint_unchanged=sha(parent) == PARENT_SHA,
                  campaign_files_unchanged=all(sha(p) == h for p, h in frozen_hashes.items()),
                  is_independent_test=False, cal_or_test_used=False, raw_message_text_saved=False,
                  accepted_for_production=False, production_v80_unchanged=True)
    for name in scores:
        result[name] = describe(select, scores[name], metrics[name]['threshold'])
    for old, new in [('parent', 'control'), ('parent', 'candidate'), ('control', 'candidate')]:
        result[old + '_vs_' + new] = paired(select, scores[old], scores[new],
                                            metrics[old]['threshold'], metrics[new]['threshold'])
        result[old + '_vs_' + new + '_fixed_0_5'] = paired(select, scores[old], scores[new], .5, .5)
    result['development_hypothesis_supported'] = bool(
        reproduced and metrics['candidate']['fp'] < metrics['control']['fp'] and
        result['control_vs_candidate']['baseline_spam_detections_lost'] == 0)
    np.savez_compressed(out / 'paired-scores.npz',
                        identity=np.array([r['identity'] for r in select]),
                        y=np.array([r['y'] for r in select]),
                        **{name: np.array(values) for name, values in scores.items()})
    (out / 'comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(experiment_completed=True, metrics=metrics,
                          control_reproduced=reproduced,
                          paired=result['control_vs_candidate']), indent=2), flush=True)
    if not reproduced or not result['parent_checkpoint_unchanged'] or not result['campaign_files_unchanged']:
        raise ValueError('Experiment verification failed; diagnostic output preserved')


if __name__ == '__main__':
    main()
