"""One bounded, resumable English neural training session; never promote v80."""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer

from v79_data import fingerprint as old_fingerprint
from v82_neural import (CONFIG, NeuralMail, begin_session, fingerprint, new_progress,
                        optimizer_and_scheduler, restore_checkpoint, run_steps,
                        save_checkpoint, trainable_state, validate)


def read_part(root, name):
    with gzip.open(root / (name + '.jsonl.gz'), 'rt') as stream:
        return [json.loads(line) for line in stream]


def subset(rows, limit):
    if not limit:
        return rows
    result = []
    for label in (0, 1):
        group = sorted((r for r in rows if r['y'] == label), key=lambda r: r['identity'])
        result.extend(group[:limit // 2])
    return sorted(result, key=lambda r: r['identity'])


def write_report(state, report):
    report = dict(report, checkpoint_sha256=hashlib.sha256(state.read_bytes()).hexdigest())
    target = state.parent / 'report.json'; pending = target.with_suffix('.partial')
    pending.write_text(json.dumps(report, indent=2)); pending.replace(target)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--state', type=Path, default=Path('neural-state/checkpoint.pt'))
    parser.add_argument('--backbone', type=Path, default=Path('.cache/v82/backbone'))
    parser.add_argument('--data', type=Path, default=Path('.cache/v79/frozen'))
    parser.add_argument('--extra', type=Path, default=Path('.cache/v80/extra'))
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--validation-limit', type=int)
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    start = dt.date.fromisoformat(request['start_date']); end = dt.date.fromisoformat(request['end_date'])
    day = dt.date.fromisoformat(request['session_id']); today = dt.datetime.now(ZoneInfo('Europe/Moscow')).date()
    if (end - start).days != 29 or not start <= day <= end or day > today or today > end:
        raise ValueError('Session outside the thirty-day campaign window')
    if not 1 <= request['budget_steps'] <= 1000 or request['expected_min_step'] < 0:
        raise ValueError('Invalid step budget or rollback guard')
    torch.set_num_threads(args.threads); torch.manual_seed(CONFIG['seed'])
    torch.use_deterministic_algorithms(True)
    train, select = [], []
    for directory, reference, manifest_key in ((args.data, 'reports/v79-frozen.json', 'development_fingerprints'),
                                                (args.extra, 'reports/v80-extra-frozen.json', 'fingerprints')):
        frozen = json.loads(Path(reference).read_text())
        if json.loads((directory / 'frozen.json').read_text()) != frozen:
            raise ValueError('Frozen corpus manifest mismatch')
        for name, output in (('train', train), ('select', select)):
            rows = read_part(directory, name)
            if old_fingerprint(rows) != frozen[manifest_key][name]:
                raise ValueError('Frozen corpus fingerprint mismatch: ' + name)
            output.extend(rows)
    if {r['family'] for r in train} & {r['family'] for r in select}:
        raise ValueError('TRAIN/SELECT family overlap')
    train = subset(train, args.train_limit); select = subset(select, args.validation_limit)
    if not train or not select:
        raise ValueError('Empty training or selection data')
    manifest = json.loads((args.backbone / 'manifest.json').read_text())
    for name, sha in manifest['files'].items():
        with (args.backbone / name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != sha:
                raise ValueError('Backbone checksum mismatch')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        raise ValueError('This campaign is pinned to deterministic CPU continuation')
    metadata = dict(config=CONFIG, backbone=manifest, device_type=device,
                    torch_version=str(torch.__version__), transformers_version=transformers.__version__,
                    cpu_threads=args.threads, campaign=request['campaign'],
                    start_date=request['start_date'], end_date=request['end_date'],
                    train=fingerprint(train), select=fingerprint(select),
                    pilot_only=bool(args.train_limit or args.validation_limit))
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, local_files_only=True)
    backbone = AutoModel.from_pretrained(args.backbone, local_files_only=True,
                                         use_safetensors=True, attn_implementation='eager')
    model = NeuralMail(backbone).to(device)
    total = math.ceil(len(train) / CONFIG['batch_size']) * CONFIG['max_epochs']
    optimizer, scheduler = optimizer_and_scheduler(model, total)
    resumed = args.state.is_file()
    if not resumed and request['expected_min_step'] > 0:
        raise ValueError('Checkpoint missing: refusing a silent restart')
    progress = restore_checkpoint(args.state, model, optimizer, scheduler, metadata) if resumed else new_progress()
    if progress['step'] < request['expected_min_step']:
        raise ValueError('Checkpoint older than the requested rollback guard')
    if not begin_session(progress, request['session_id'], request['budget_steps']):
        cached = progress.get('last_report')
        if not cached:
            report_path = args.state.parent / 'report.json'
            if (not report_path.is_file() or
                (request['session_id'] in progress['completed_sessions'] and
                 json.loads(report_path.read_text())['session'] != request['session_id'])):
                raise ValueError('Legacy completed checkpoint has no matching report; explicit recovery required')
            cached = json.loads(report_path.read_text())
        write_report(args.state, dict(cached, invocation_optimizer_steps=0,
                                     completed_session_replayed=True, checkpoint_restored_in_invocation=True))
        print('Session already completed or training stopped; no optimizer steps executed', flush=True)
        return
    before = trainable_state(model); initial_step = progress['step']
    groups = Counter((r['source'], r['y']) for r in train)
    weights = [min(12., 5000 / groups[r['source'], r['y']]) * (CONFIG['ham_cost'] if r['y'] == 0 else 1.) for r in train]
    mean = sum(weights) / len(weights); weights = [w / mean for w in weights]
    save = lambda: save_checkpoint(args.state, model, optimizer, scheduler, progress, metadata)
    save()
    print(json.dumps(dict(training_started=True, resumed=resumed, global_step=progress['step'],
                          target_step=progress['active_target'], train=len(train), select=len(select))), flush=True)
    losses = run_steps(model, tokenizer, train, weights, optimizer, scheduler, progress, save=save)
    save()  # A validation interruption must not grant another optimization budget.
    metric = validate(model, tokenizer, select)
    better = progress['best_fp'] is None or (metric['fp'], metric['log_loss']) < (progress['best_fp'], progress['best_loss'])
    if better:
        progress['best_fp'] = metric['fp']; progress['best_loss'] = metric['log_loss']; progress['stalled'] = 0
        save_file(trainable_state(model), str(args.state.parent / 'best-adapter.safetensors'))
        (args.state.parent / 'best-development.json').write_text(json.dumps(metric, indent=2))
    else:
        progress['stalled'] += 1
    progress['stopped'] = progress['epoch'] >= CONFIG['max_epochs'] or (progress['epoch'] >= 1 and progress['stalled'] >= 3)
    session_steps = progress['step'] - progress.get('active_start', initial_step)
    progress['completed_sessions'].append(request['session_id'])
    progress['active_session'] = None; progress['active_target'] = None
    progress['active_start'] = None
    after = trainable_state(model)
    adapter_keys = [k for k in after if k.endswith(('.a', '.b'))]
    report = dict(version=82, campaign=request['campaign'], session=request['session_id'], resumed=resumed,
                  checkpoint_restored_in_invocation=resumed, completed_session_replayed=False,
                  session_optimizer_steps=session_steps, invocation_optimizer_steps=progress['step'] - initial_step,
                  global_optimizer_steps=progress['step'],
                  epoch=progress['epoch'], cursor=progress['cursor'], completed_sessions=len(progress['completed_sessions']),
                  stopped=progress['stopped'], train_rows=len(train), select_rows=len(select),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  changed_internal_adapter_tensors=sum(not torch.equal(after[k], before[k]) for k in adapter_keys),
                  internal_adapter_delta_l2=math.sqrt(sum(float((after[k] - before[k]).square().sum()) for k in adapter_keys)),
                  mean_training_loss=sum(losses) / len(losses) if losses else None, development=metric,
                  best_development_fp=progress['best_fp'],
                  pilot_only=metadata['pilot_only'], accepted_for_production=False, production_v80_unchanged=True,
                  test_or_cal_used_for_gradient_or_selection=False, customer_mail_used=False, quality_gain_demonstrated=False)
    progress['last_report'] = report
    save()  # The report is recoverable even if publication is interrupted next.
    write_report(args.state, report)


if __name__ == '__main__':
    main()
