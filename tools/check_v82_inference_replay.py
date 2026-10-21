"""Read-only fixed-500-step SELECT replay: original validate vs inference_mode."""
import argparse
import hashlib
import json
from pathlib import Path

from audit_v82_errors import metric, paired, sha
from summarize_v82_audit_cache import log_loss, validate_alignment


def compare_scores(rows, left, right, cutoff):
    # Also validates lengths, labels and finite probability values.
    log_loss(rows, left); log_loss(rows, right)
    deltas = [abs(float(a) - float(b)) for a, b in zip(left, right)]
    transitions = paired(rows, left, right, cutoff, cutoff)
    transitions.pop('first_lost_spam_identities')
    return dict(n=len(rows), exactly_equal_scores=sum(d == 0 for d in deltas),
                max_absolute_score_difference=max(deltas),
                mean_absolute_score_difference=sum(deltas) / len(rows),
                paired_at_locked_threshold=transitions)


def replay_metrics(rows, scores, report):
    cutoff = report['development']['threshold']
    observed = metric(rows, scores, cutoff)
    loss = log_loss(rows, scores); expected = report['development']['log_loss']
    return dict(at_locked_threshold=observed, log_loss=loss, historical_log_loss=expected,
                log_loss_difference=loss - expected, log_loss_absolute_tolerance=1e-9,
                source_report_counts_reproduced=all(observed[k] == report['development'][k]
                                                    for k in ('n', 'spam', 'ham', 'tp', 'fp')),
                source_report_log_loss_reproduced=abs(loss - expected) <= 1e-9)


def inference_context_conclusion(no_grad, inference, comparison):
    if comparison['max_absolute_score_difference'] == 0:
        return 'contexts_identical_on_this_replay'
    if no_grad['source_report_log_loss_reproduced'] and not inference['source_report_log_loss_reproduced']:
        return 'original_no_grad_reproduces_historical_loss_but_inference_mode_does_not'
    return 'context_difference_observed_but_historical_replay_not_explained'


def main():
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    from train_v82_daily import read_part
    from v82_neural import CONFIG, NeuralMail, fingerprint, load_trainable, text_tokenizer, validate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, default=Path('.cache/v82/backbone'))
    parser.add_argument('--out', type=Path, default=Path('.cache/v82-replay/output'))
    args = parser.parse_args(); request = json.loads(args.request.read_text())
    if request.get('replay_inference_contexts') is not True or request['baseline_step'] != 500:
        raise ValueError('This experiment is restricted to the requested fixed 500-step replay')
    reference = json.loads(Path(request['cached_reference_report']).read_text())
    cache_report_path = args.cache / 'audit.json'; cache_path = args.cache / 'development-scores.npz'
    initial_hashes = {str(p): sha(p) for p in (args.state, cache_report_path, cache_path)}
    if json.loads(cache_report_path.read_text()) != reference:
        raise ValueError('Source artifact audit differs from committed reference')
    if (reference['campaign'] != request['campaign'] or reference['baseline_step'] != 500 or
        not reference['checkpoint_files_unchanged'] or not reference['source_report_counts_reproduced'] or
        reference['optimizer_steps_executed'] != 0 or reference['cal_or_test_opened']):
        raise ValueError('Source audit provenance failed')
    if initial_hashes[str(args.state)] != reference['baseline_checkpoint_sha256']:
        raise ValueError('This is not the fixed baseline checkpoint')
    report = json.loads((args.state.parent / 'report.json').read_text())
    if report != json.loads(Path(request['cached_baseline_report']).read_text()):
        raise ValueError('Baseline artifact report differs from committed daily report')
    manifest = json.loads((args.backbone / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if sha(args.backbone / name) != expected:
            raise ValueError('Backbone checksum drift')
    train, select = [], []
    for directory, published in ((Path('.cache/v79/frozen'), Path('reports/v79-frozen.json')),
                                  (Path('.cache/v80/extra'), Path('reports/v80-extra-frozen.json'))):
        if json.loads((directory / 'frozen.json').read_text()) != json.loads(published.read_text()):
            raise ValueError('Frozen manifest drift')
        train.extend(read_part(directory, 'train')); select.extend(read_part(directory, 'select'))
    fingerprints = dict(train=fingerprint(train), select=fingerprint(select))
    if fingerprints != reference['corpus_fingerprints']:
        raise ValueError('Full TRAIN/SELECT content fingerprint drift')
    cycle = json.loads(Path('research/v82/cycle.json').read_text())
    expected_meta = dict(config=CONFIG, backbone=manifest, device_type='cpu',
                         torch_version=str(torch.__version__), transformers_version=transformers.__version__,
                         cpu_threads=3, campaign=request['campaign'], start_date=cycle['start_date'],
                         end_date=cycle['end_date'], **fingerprints, pilot_only=False)
    torch.set_num_threads(3); torch.manual_seed(CONFIG['seed']); torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available() or cycle['campaign'] != request['campaign']:
        raise ValueError('Replay requires the original CPU campaign')
    checkpoint = torch.load(args.state, map_location='cpu', weights_only=True)
    progress = checkpoint['progress']
    if checkpoint['metadata'] != expected_meta:
        raise ValueError('Strict checkpoint metadata mismatch')
    if (progress['step'] != 500 or progress['active_session'] is not None or
        report['global_optimizer_steps'] != 500 or
        report['checkpoint_sha256'] != initial_hashes[str(args.state)]):
        raise ValueError('Expected matching completed 500-step checkpoint and report')
    with np.load(cache_path, allow_pickle=False) as values:
        if any(values[k].ndim != 1 for k in ('select_identity', 'select_y', 'baseline')):
            raise ValueError('Invalid cache shape')
        identities = values['select_identity'].tolist(); labels = values['select_y'].tolist()
        cached = values['baseline'].tolist()
    validate_alignment(select, identities, labels, cached)
    cached_summary = replay_metrics(select, cached, report)
    if cached_summary['at_locked_threshold'] != reference['paired_select']['baseline']:
        raise ValueError('Cached baseline counts do not reproduce original audit')
    backbone = AutoModel.from_pretrained(args.backbone, local_files_only=True,
                                        use_safetensors=True, attn_implementation='eager')
    model = NeuralMail(backbone); load_trainable(model, checkpoint['adapter']); model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, local_files_only=True)

    def model_hash():
        digest = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    parameters_before = model_hash(); rng_before = torch.get_rng_state().clone()
    class CaptureValidateScores(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.original = model; self.scores = []; self.contexts = set()

        def forward(self, **inputs):
            self.contexts.add((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
            logits = self.original(**inputs)
            self.scores.extend(logits.sigmoid().cpu().tolist())
            if (len(self.scores) - len(logits)) % 2048 == 0:
                print(json.dumps(dict(context='original_validate_no_grad', scored=len(self.scores), total=len(select))), flush=True)
            return logits

    captured = CaptureValidateScores()
    # Calls the original, unmodified validation implementation, not a replacement loop.
    original_validate_result = validate(captured, tokenizer, select)
    no_grad_scores = captured.scores
    if captured.contexts != {(False, False)}:
        raise ValueError('Original validate unexpectedly changed its inference context')
    no_grad_summary = replay_metrics(select, no_grad_scores, report)
    if abs(original_validate_result['log_loss'] - no_grad_summary['log_loss']) > 1e-12:
        raise ValueError('Captured scores do not reproduce original validate loss formula')
    inference_scores = []
    with torch.inference_mode():
        for start in range(0, len(select), 32):
            inputs = text_tokenizer(tokenizer, [r['text'] for r in select[start:start+32]], CONFIG['max_length'])
            inference_scores.extend(model(**inputs).sigmoid().tolist())
            if start % 2048 == 0:
                print(json.dumps(dict(context='auditor_inference_mode', scored=min(start+32, len(select)), total=len(select))), flush=True)
    inference_summary = replay_metrics(select, inference_scores, report)
    cutoff = report['development']['threshold']
    context_comparison = compare_scores(select, no_grad_scores, inference_scores, cutoff)
    if (model_hash() != parameters_before or not torch.equal(rng_before, torch.get_rng_state()) or
        any(p.grad is not None for p in model.parameters()) or
        any(sha(p) != expected for p, expected in initial_hashes.items())):
        raise ValueError('Read-only replay mutated state, RNG, cache, weights or gradients')
    result = dict(campaign=request['campaign'], baseline_step=500, baseline_run_id=request['baseline_run_id'],
                  baseline_checkpoint_sha256=initial_hashes[str(args.state)], corpus_fingerprints=fingerprints,
                  source_audit_run_id=request['cached_audit_run_id'], input_sha256=initial_hashes,
                  original_validate_no_grad=no_grad_summary, auditor_inference_mode=inference_summary,
                  historical_cached_baseline=cached_summary, context_comparison=context_comparison,
                  no_grad_vs_cached=compare_scores(select, cached, no_grad_scores, cutoff),
                  inference_mode_vs_cached=compare_scores(select, cached, inference_scores, cutoff),
                  inference_context_conclusion=inference_context_conclusion(no_grad_summary, inference_summary, context_comparison),
                  original_validate_code_sha256=sha(Path(__file__).with_name('v82_neural.py')),
                  replay_code_sha256=sha(__file__), cpu_capability=torch.backends.cpu.get_cpu_capability(),
                  torch_version=str(torch.__version__), cpu_threads=torch.get_num_threads(),
                  mkldnn_enabled=torch.backends.mkldnn.enabled,
                  float32_matmul_precision=torch.get_float32_matmul_precision(),
                  checkpoint_files_unchanged=True, source_cache_unchanged=True,
                  model_parameters_unchanged=True, rng_unchanged=True,
                  optimizer_instantiated=False, optimizer_steps_executed=0,
                  scored_select_rows_per_context=len(select), inference_batches_executed=2*((len(select)+31)//32),
                  raw_message_text_saved=False, cal_or_test_opened=False,
                  labels_automatically_changed=False, quality_gain_demonstrated=False,
                  independent_quality_test_performed=False, production_v80_unchanged=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'replay.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(args.out / 'replay-scores.npz', select_identity=np.array(identities), select_y=np.array(labels),
                        no_grad=np.array(no_grad_scores), inference_mode=np.array(inference_scores), cached_baseline=np.array(cached))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
