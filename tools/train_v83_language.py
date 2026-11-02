"""Isolated, bounded RU/ES multilingual LoRA pilot; no English campaign mutation."""
import argparse
from collections import Counter
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path

MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
WEIGHT_SHA = 'eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b'
CONFIG = dict(version='v83-multilingual-lora-r8-pilot-v1', model=MODEL,
              revision=REVISION, rank=8, alpha=16, max_length=128, seed=83,
              batch_size=8, learning_rate=.0002, weight_decay=.01,
              max_epochs=3, ham_cost=2.)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_request(request, language, today=None):
    today = today or dt.datetime.now(dt.timezone.utc).date()
    if language not in ('ru', 'es') or request.get('language') != language:
        raise ValueError('Request language does not match the isolated pilot')
    campaign = request.get('campaign')
    if not isinstance(campaign, str) or not campaign.startswith('v83-'):
        raise ValueError('Separate v83 campaign name required')
    session = request.get('session_id')
    try:
        day = dt.date.fromisoformat(session)
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid session date') from exc
    if day.isoformat() != session or day > today:
        raise ValueError('Future or noncanonical session date')
    if 'start_date' in request or 'end_date' in request:
        start=dt.date.fromisoformat(request['start_date']);end=dt.date.fromisoformat(request['end_date'])
        if (end-start).days!=29 or not start<=day<=end or today>end:
            raise ValueError('Outside the thirty-day campaign window')
    if (type(request.get('budget_steps')) is not int or
            not 1 <= request['budget_steps'] <= 500 or
            type(request.get('expected_min_step')) is not int or request['expected_min_step'] < 0):
        raise ValueError('Invalid bounded step budget or rollback guard')


def validate_rows(rows, expected, minimum, language):
    if not rows:
        raise ValueError('Empty language partition')
    identities = set()
    counts = Counter()
    for row in rows:
        if (not isinstance(row, dict) or type(row.get('y')) is not int or row['y'] not in (0, 1) or
                any(not isinstance(row.get(k), str) or not row[k].strip()
                    for k in ('identity', 'family', 'source', 'text')) or
                row.get('language', language) != language):
            raise ValueError('Invalid labelled language row')
        if row['identity'] in identities:
            raise ValueError('Duplicate message identity')
        identities.add(row['identity']); counts[row['y']] += 1
    actual = dict(n=len(rows), spam=counts[1], ham=counts[0])
    if actual != expected or min(counts[0], counts[1]) < minimum:
        raise ValueError('Partition counts or minimum class support failed')
    return actual


def read_partition(root, name):
    if name not in ('train', 'select'):
        raise ValueError('This pilot never opens TEST or CAL')
    with gzip.open(Path(root) / (name + '.jsonl.gz'), 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def check_progress(progress, request):
    if progress['step'] < request['expected_min_step']:
        raise ValueError('Checkpoint older than the rollback guard')
    if progress['active_session'] is not None:
        if progress['active_session'] != request['session_id']:
            raise ValueError('Resume unfinished session before another date')
        if progress['active_target'] - progress['active_start'] != request['budget_steps']:
            raise ValueError('Interrupted session must keep its original budget')


def metric(rows, scores, threshold):
    if len(rows) != len(scores) or not rows or not math.isfinite(threshold):
        raise ValueError('Invalid SELECT score alignment')
    if any(not math.isfinite(float(s)) or not 0 <= s <= 1 for s in scores):
        raise ValueError('Nonfinite or invalid SELECT probabilities')
    spam = sum(row['y'] for row in rows); ham = len(rows) - spam
    tp = sum(row['y'] == 1 and score >= threshold for row, score in zip(rows, scores))
    fp = sum(row['y'] == 0 and score >= threshold for row, score in zip(rows, scores))
    return dict(n=len(rows), spam=spam, ham=ham, tp=tp, fp=fp, fn=spam-tp, tn=ham-fp,
                recall=tp/spam if spam else None, fpr=fp/ham if ham else None,
                fp_per_million_ham=fp/ham*1e6 if ham else None, threshold=threshold)


def validation_report(rows, scores):
    fixed = metric(rows, scores, .5)
    spam_scores = sorted(score for row, score in zip(rows, scores) if row['y'] == 1)
    fitted = None
    if len(spam_scores) >= 100:
        threshold = spam_scores[int(math.floor(.02 * len(spam_scores)))]
        fitted = metric(rows, scores, threshold)
    loss = -sum(row['y'] * math.log(min(max(score, 1e-7), 1-1e-7)) +
                (1-row['y']) * math.log(1-min(max(score, 1e-7), 1-1e-7))
                for row, score in zip(rows, scores)) / len(rows)
    sources = sorted({row['source'] for row in rows})
    by_source = {}
    for source in sources:
        indices = [i for i, row in enumerate(rows) if row['source'] == source]
        by_source[source] = metric([rows[i] for i in indices], [scores[i] for i in indices], .5)
    return dict(partition='SELECT', at_fixed_0_5=fixed, at_fitted_98_recall=fitted,
                fitted_98_status='available' if fitted is not None else 'insufficient_for_98_target',
                log_loss=loss, by_source_fixed_0_5=by_source, is_independent_test=False,
                threshold_selected_on_same_development=fitted is not None,
                million_message_test_performed=False, quality_gain_demonstrated=False)


def frozen_parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if not value.requires_grad:
            digest.update(name.encode())
            digest.update(memoryview(value.detach().cpu().contiguous().numpy()))
    return digest.hexdigest()


def write_report(state, report):
    result = dict(report, checkpoint_sha256=sha(state))
    pending = state.parent / 'report.json.partial'
    pending.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(state.parent / 'report.json')
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


def main():
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    from v82_neural import (NeuralMail, begin_session, fingerprint, new_progress,
                            optimizer_and_scheduler, restore_checkpoint, run_steps,
                            save_checkpoint, trainable_state, text_tokenizer)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--language', choices=('ru', 'es'), required=True)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--backbone', type=Path, default=Path('.cache/v83/backbone'))
    args = parser.parse_args()
    args.state = args.state or Path('language-state') / args.language / 'checkpoint.pt'
    args.data = args.data or Path('.cache/v83') / args.language
    if args.state.resolve() == Path('neural-state/checkpoint.pt').resolve():
        raise ValueError('English campaign state cannot be used for language pilot')
    request = json.loads(args.request.read_text())
    validate_request(request, args.language)
    source_path = args.data / 'manifest.json'
    source = json.loads(source_path.read_text())
    if source.get('training_ready') is not True:
        raise ValueError('Language corpus has not passed class-support audit')
    if source.get('language') != args.language:
        raise ValueError('Frozen language manifest mismatch')
    if any(name not in source.get('fingerprints', {}) or name not in source.get('counts', {})
           for name in ('train', 'select', 'test')):
        raise ValueError('Missing frozen TRAIN/SELECT/TEST manifest')
    # TEST metadata is recorded by the preparer; TEST contents are never opened here.
    train = read_partition(args.data, 'train'); select = read_partition(args.data, 'select')
    validate_rows(train, source['counts']['train'], 100, args.language)
    validate_rows(select, source['counts']['select'], 30, args.language)
    for name, rows in (('train', train), ('select', select)):
        if fingerprint(rows) != source['fingerprints'][name]:
            raise ValueError('Frozen language partition fingerprint mismatch')
    if ({row['identity'] for row in train} & {row['identity'] for row in select} or
            {row['family'] for row in train} & {row['family'] for row in select}):
        raise ValueError('TRAIN/SELECT identity or family leakage')

    manifest = json.loads((args.backbone / 'manifest.json').read_text())
    files = manifest.get('files', {})
    required = {'model.safetensors', 'config.json', 'tokenizer.json',
                'tokenizer_config.json', 'special_tokens_map.json'}
    if (manifest.get('model') != MODEL or manifest.get('revision') != REVISION or
            not required <= set(files) or files['model.safetensors'] != WEIGHT_SHA):
        raise ValueError('Pinned multilingual backbone manifest mismatch')
    for name, expected in files.items():
        if Path(name).name != name or sha(args.backbone / name) != expected:
            raise ValueError('Multilingual backbone file checksum mismatch')
    torch.set_num_threads(3); torch.manual_seed(CONFIG['seed'])
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        raise ValueError('Language pilots are pinned to CPU')
    metadata = dict(language=args.language, campaign=request['campaign'], config=CONFIG,
                    start_date=request.get('start_date'),end_date=request.get('end_date'),
                    backbone=manifest, source_manifest_sha256=sha(source_path),
                    train=fingerprint(train), select=fingerprint(select), device_type='cpu',
                    cpu_threads=3, torch_version=str(torch.__version__),
                    transformers_version=transformers.__version__,
                    pilot_only=True, objective='global_mean_normalized_HAM_cost_2_no_source_reweighting')
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, local_files_only=True)
    backbone = AutoModel.from_pretrained(args.backbone, local_files_only=True,
                                         use_safetensors=True, attn_implementation='eager')
    if len(backbone.encoder.layer) != 12 or backbone.config.hidden_size != 384:
        raise ValueError('Unexpected multilingual architecture')
    model = NeuralMail(backbone, CONFIG)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != 147841:
        raise ValueError('Unexpected multilingual LoRA parameter count')
    total = math.ceil(len(train) / CONFIG['batch_size']) * CONFIG['max_epochs']
    optimizer, scheduler = optimizer_and_scheduler(model, total, CONFIG)
    resumed = args.state.is_file()
    if not resumed and request['expected_min_step'] > 0:
        raise ValueError('Required language checkpoint missing; refusing cold restart')
    progress = restore_checkpoint(args.state, model, optimizer, scheduler, metadata) if resumed else new_progress()
    check_progress(progress, request)
    if not begin_session(progress, request['session_id'], request['budget_steps']):
        cached = progress.get('last_report')
        if not cached:
            raise ValueError('Completed pilot checkpoint has no recoverable report')
        write_report(args.state, dict(cached, invocation_optimizer_steps=0,
                                     completed_session_replayed=True, checkpoint_restored_in_invocation=True))
        return
    initial_step = progress['step']; before = trainable_state(model)
    frozen_before = frozen_parameter_hash(model)
    weights = [CONFIG['ham_cost'] if row['y'] == 0 else 1. for row in train]
    mean = sum(weights) / len(weights); weights = [value / mean for value in weights]
    save = lambda: save_checkpoint(args.state, model, optimizer, scheduler, progress, metadata)
    save()
    print(json.dumps(dict(language=args.language, training_started=True,
                          resumed=resumed, global_step=initial_step,
                          target_step=progress['active_target'], train_rows=len(train))), flush=True)
    losses = run_steps(model, tokenizer, train, weights, optimizer, scheduler, progress, CONFIG, save=save)
    save()  # Validation failure cannot grant another optimizer budget.
    values = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(select), 8):
            batch = select[start:start+8]
            inputs = text_tokenizer(tokenizer, [row['text'] for row in batch], CONFIG['max_length'])
            values.extend(model(**inputs).sigmoid().cpu().tolist())
    development = validation_report(select, values)
    after = trainable_state(model)
    keys = [key for key in after if key.endswith(('.a', '.b'))]
    changes = sum(not torch.equal(after[key], before[key]) for key in keys)
    frozen_unchanged = frozen_parameter_hash(model) == frozen_before
    if len(keys) != 48 or not frozen_unchanged or (progress['step'] > initial_step and changes == 0):
        raise ValueError('Internal adapter learning or frozen backbone verification failed')
    better=progress['best_loss'] is None or development['log_loss']<progress['best_loss']
    if better:
        from safetensors.torch import save_file
        progress['best_loss']=development['log_loss'];progress['stalled']=0
        save_file(after,str(args.state.parent/'best-adapter.safetensors'))
        (args.state.parent/'best-development.json').write_text(json.dumps(dict(development=development,step=progress['step']),indent=2))
    else:
        progress['stalled']+=1
    progress['stopped'] = progress['epoch'] >= CONFIG['max_epochs'] or (progress['epoch']>=1 and progress['stalled']>=3)
    session_steps = progress['step'] - progress['active_start']
    progress['completed_sessions'].append(request['session_id'])
    progress['active_session'] = progress['active_target'] = progress['active_start'] = None
    report = dict(version=83, language=args.language, campaign=request['campaign'],
                  session=request['session_id'], resumed=resumed, checkpoint_restored_in_invocation=resumed,
                  completed_session_replayed=False, requested_budget_steps=request['budget_steps'],
                  session_optimizer_steps=session_steps, invocation_optimizer_steps=progress['step']-initial_step,
                  global_optimizer_steps=progress['step'], epoch=progress['epoch'], cursor=progress['cursor'],
                  completed_sessions=len(progress['completed_sessions']), stopped=progress['stopped'],
                  stop_reason=('max_epochs' if progress['epoch']>=CONFIG['max_epochs'] else 'development_stagnation') if progress['stopped'] else None,
                  train_rows=len(train), select_rows=len(select), trainable_parameters=147841,
                  changed_internal_adapter_tensors=changes, expected_internal_adapter_tensors=48,
                  internal_adapter_delta_l2=math.sqrt(sum(float((after[k]-before[k]).square().sum()) for k in keys)),
                  original_backbone_unchanged=frozen_unchanged,
                  mean_training_loss=sum(losses)/len(losses) if losses else None,
                  development=development, source_manifest_sha256=metadata['source_manifest_sha256'],
                  backbone_revision=REVISION, test_or_cal_opened=False, customer_mail_used=False,
                  accepted_for_production=False, production_v80_unchanged=True,
                  english_v82_campaign_unchanged=True, quality_gain_demonstrated=False, pilot_only=True)
    progress['last_report'] = report
    save()
    write_report(args.state, report)


if __name__ == '__main__':
    main()
