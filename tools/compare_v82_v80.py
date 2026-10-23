"""Read-only paired regression of locked v80 recall-98 and one fixed v82 state."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from audit_v82_errors import length_bucket, metric, paired, sha


def parameter_hash(model):
    return hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes()
                                  for t in model.state_dict().values())).hexdigest()


def initialize_scorer(backbone_path, checkpoint_path, expected_sha):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from v82_neural import CONFIG, NeuralMail, load_trainable
    global _model, _tokenizer, _parameter_hash, _rng
    if sha(checkpoint_path) != expected_sha:
        raise ValueError('Worker checkpoint differs from locked state')
    torch.set_num_threads(3); torch.manual_seed(CONFIG['seed']); torch.use_deterministic_algorithms(True)
    _tokenizer = AutoTokenizer.from_pretrained(backbone_path, local_files_only=True)
    _model = NeuralMail(AutoModel.from_pretrained(backbone_path, local_files_only=True,
                       use_safetensors=True, attn_implementation='eager'))
    load_trainable(_model, torch.load(checkpoint_path,map_location='cpu',weights_only=True)['adapter'])
    _model.eval(); _parameter_hash = parameter_hash(_model); _rng = torch.get_rng_state().clone()


def score_chunk(texts):
    import torch
    from v82_neural import CONFIG, text_tokenizer
    values = []
    with torch.no_grad():
        for start in range(0,len(texts),32):
            inputs = text_tokenizer(_tokenizer,texts[start:start+32],CONFIG['max_length'])
            values.extend(_model(**inputs).sigmoid().tolist())
    if (parameter_hash(_model) != _parameter_hash or not torch.equal(_rng,torch.get_rng_state()) or
        any(p.grad is not None for p in _model.parameters())):
        raise ValueError('Inference worker changed weights, RNG or gradients')
    return values


def neural_parallel(backbone, state, expected_sha, rows, name):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    # Three CPU workers retain the original 3 threads and batch32 in each process.
    # 1024 is a multiple of 32: sharding cannot move any original batch boundary.
    chunks = [[r['text'] for r in rows[start:start+1024]] for start in range(0,len(rows),1024)]
    values = []
    with ProcessPoolExecutor(max_workers=min(3,len(chunks)),mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_scorer,initargs=(str(backbone),str(state),expected_sha)) as pool:
        for result in pool.map(score_chunk,chunks):
            values.extend(result)
            print(json.dumps(dict(cohort=name,scored=len(values),total=len(rows))),flush=True)
    return values


def compare(rows, old, new, old_cutoff, new_cutoff):
    result = paired(rows, old, new, old_cutoff, new_cutoff)
    result.update(compares_two_v82_development_checkpoints=False,
                  is_a_v80_regression_check=True,
                  baseline_version=80, current_version=82)
    result['passes_fp_and_individual_tp_gate'] = (
        result['current']['fp'] < result['baseline']['fp'] and
        result['baseline_spam_detections_lost'] == 0)
    return result


def grouped(rows, old, new, old_cutoff, new_cutoff, key):
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(key(row), []).append(i)
    return {name: compare([rows[i] for i in ix], [old[i] for i in ix],
                          [new[i] for i in ix], old_cutoff, new_cutoff)
            for name, ix in sorted(groups.items())}


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def main():
    import joblib
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    import v80_mail
    from v79_data import fingerprint as identities_fingerprint
    from v81_features import lexical_parallel
    from v82_neural import CONFIG, NeuralMail, fingerprint, load_trainable, text_tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, default=Path('research/v82/v80-comparison.json'))
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--v80-model', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, default=Path('.cache/v82/backbone'))
    parser.add_argument('--out', type=Path, default=Path('.cache/v82-v80-comparison'))
    args = parser.parse_args(); request = json.loads(args.request.read_text())
    if (request['optimizer_steps_authorized'] != 0 or
        request['production_promotion_authorized'] or
        not request['thresholds_locked_before_regression_scoring'] or
        request['v80_profile'] != 'recall-98' or request['v80_candidate'] != 'fast-or'):
        raise ValueError('Unsupported comparison request')
    base_path = args.v80_model.parent / 'base/model.joblib'
    files = (args.request, args.state, args.v80_model, base_path, args.state.parent / 'report.json')
    input_hashes = {str(p): sha(p) for p in files}
    for path, key in ((args.state, 'v82_checkpoint_sha256'), (args.v80_model, 'v80_model_sha256'),
                      (base_path, 'v80_base_sha256')):
        if input_hashes[str(path)] != request[key]:
            raise ValueError('Locked model/checkpoint checksum mismatch')
    report = json.loads((args.state.parent / 'report.json').read_text())
    if report != json.loads(Path(request['v82_source_report']).read_text()):
        raise ValueError('Checkpoint artifact report differs from committed source')
    if (report['checkpoint_sha256'] != request['v82_checkpoint_sha256'] or
        report['development']['threshold'] != request['v82_threshold'] or
        report['global_optimizer_steps'] != request['v82_step'] or report['session'] != request['session']):
        raise ValueError('Source report is not the locked candidate')
    locked = json.loads(Path('reports/v80-policies-locked.json').read_text())
    bundle = joblib.load(args.v80_model)
    if (bundle['version'] != v80_mail.VERSION or locked['model_sha256'] != request['v80_model_sha256'] or
        bundle['policies'] != locked['policies'] or
        locked['policies']['recall-98']['candidate'] != 'fast-or' or
        locked['policies']['recall-98']['cutoff'] != request['v80_threshold']):
        raise ValueError('v80 differs from its published pre-test lock')
    train, select = [], []
    for root, published, key in ((Path('.cache/v79/frozen'), 'reports/v79-frozen.json', 'development_fingerprints'),
                                  (Path('.cache/v80/extra'), 'reports/v80-extra-frozen.json', 'fingerprints')):
        manifest = json.loads(Path(published).read_text())
        if json.loads((root / 'frozen.json').read_text()) != manifest:
            raise ValueError('Frozen corpus manifest mismatch')
        for name, output in (('train', train), ('select', select)):
            rows = read_rows(root / (name + '.jsonl.gz'))
            if identities_fingerprint(rows) != manifest[key][name]:
                raise ValueError('TRAIN/SELECT identity fingerprint mismatch')
            output.extend(rows)
    original = json.loads(Path('reports/v79-frozen.json').read_text())
    extra = json.loads(Path('reports/v80-extra-frozen.json').read_text())
    if bundle['original_manifest'] != original or bundle['extra_manifest'] != extra:
        raise ValueError('v80 training provenance differs from frozen corpus')
    backbone_manifest = json.loads((args.backbone / 'manifest.json').read_text())
    for name, expected in backbone_manifest['files'].items():
        if sha(args.backbone / name) != expected:
            raise ValueError('Pinned backbone checksum mismatch')
    torch.set_num_threads(3); torch.manual_seed(CONFIG['seed']); torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(args.state, map_location='cpu', weights_only=True)
    expected_metadata = dict(config=CONFIG, backbone=backbone_manifest, device_type='cpu',
        torch_version=str(torch.__version__), transformers_version=transformers.__version__, cpu_threads=3,
        campaign=request['campaign'], start_date='2026-10-20', end_date='2026-11-18',
        train=fingerprint(train), select=fingerprint(select), pilot_only=False)
    progress = checkpoint['progress']
    if (checkpoint['metadata'] != expected_metadata or progress['step'] != request['v82_step'] or
        progress['active_session'] is not None or progress['last_report'] != {k: v for k, v in report.items()
                                                                         if k != 'checkpoint_sha256'}):
        raise ValueError('Checkpoint metadata, progress or saved report mismatch')
    development_families = {r['family'] for r in train + select}
    del train, select
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, local_files_only=True)
    model = NeuralMail(AutoModel.from_pretrained(args.backbone, local_files_only=True,
                        use_safetensors=True, attn_implementation='eager'))
    load_trainable(model, checkpoint['adapter']); model.eval()
    def model_hash():
        return parameter_hash(model)
    parameters_before = model_hash(); rng_before = torch.get_rng_state().clone()
    args.out.mkdir(parents=True, exist_ok=True)
    code_hashes = {name: sha(Path(__file__).with_name(name)) for name in
                  ('compare_v82_v80.py', 'audit_v82_errors.py', 'v80_mail.py', 'v77_mail.py',
                   'v81_features.py', 'v82_neural.py')}
    old_cutoff, new_cutoff = request['v80_threshold'], request['v82_threshold']
    results = {}; all_rows, all_old, all_new = [], [], []
    cohorts = [('Nazario-2024-247', read_rows(Path('.cache/v80/extra/fresh-test.jsonl.gz')),
                extra['fingerprints']['fresh-test'])]
    regression = read_rows(Path('.cache/v79/frozen/test.jsonl.gz'))
    if len(regression) != 100000 or identities_fingerprint(regression) != original['test_fingerprint']:
        raise ValueError('Frozen 100000 regression changed')
    for block in original['blocks']:
        rows = regression[(block['block']-1)*10000:block['block']*10000]
        if len(rows) != block['n'] or sum(r['y'] for r in rows) != block['spam']:
            raise ValueError('Frozen block count/labels mismatch')
        cohorts.append((f"block-{block['block']:02d}", rows, block['fingerprint']))
    for name, rows, expected_fingerprint in cohorts:
        if identities_fingerprint(rows) != expected_fingerprint:
            raise ValueError('Frozen cohort identity mismatch')
        if development_families & {r['family'] for r in rows}:
            raise ValueError('Regression intersects TRAIN/SELECT families')
        receipt = dict(input_sha256=input_hashes, code_sha256=code_hashes, cohort=fingerprint(rows))
        cache = args.out / (name + '.npz'); meta = args.out / (name + '-receipt.json')
        if cache.exists():
            if json.loads(meta.read_text()) != receipt:
                raise ValueError('Cannot reuse scores from another evaluation')
            with np.load(cache, allow_pickle=False) as saved:
                if (saved['identity'].tolist() != [r['identity'] for r in rows] or
                    saved['y'].tolist() != [r['y'] for r in rows]):
                    raise ValueError('Cached identities or labels differ')
                old, new = saved['v80'].tolist(), saved['v82'].tolist()
        else:
            fast = lexical_parallel(args.v80_model, [r['text'] for r in rows], workers=3)
            # Published recall-98 is purely lexical; semantic columns and gates are irrelevant.
            old = v80_mail.candidate_scores({}, np.column_stack((fast, np.zeros((len(rows), 3)))))['fast-or'].tolist()
            new = neural_parallel(args.backbone,args.state,request['v82_checkpoint_sha256'],rows,name)
            with torch.no_grad():
                inputs = text_tokenizer(tokenizer,[r['text'] for r in rows[:32]],CONFIG['max_length'])
                serial = model(**inputs).sigmoid().tolist()
            if serial != new[:32]:
                raise ValueError('Parallel worker differs from original serial batch32 inference')
            np.savez_compressed(cache, identity=np.array([r['identity'] for r in rows]),
                                y=np.array([r['y'] for r in rows]), v80=np.array(old), v82=np.array(new))
            meta.write_text(json.dumps(receipt, indent=2))
        result = compare(rows, old, new, old_cutoff, new_cutoff)
        result['by_source'] = grouped(rows, old, new, old_cutoff, new_cutoff, lambda r:r['source'])
        result['by_word_count'] = grouped(rows, old, new, old_cutoff, new_cutoff, lambda r:length_bucket(r['text']))
        result['score_cache_sha256'] = sha(cache); result['receipt'] = receipt
        results[name] = result
        (args.out / (name + '.json')).write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(completed_cohort=name, comparison=compare(rows,old,new,old_cutoff,new_cutoff))), flush=True)
        if name.startswith('block-'):
            all_rows.extend(rows); all_old.extend(old); all_new.extend(new)
    total = compare(all_rows, all_old, all_new, old_cutoff, new_cutoff)
    total['by_source'] = grouped(all_rows, all_old, all_new, old_cutoff, new_cutoff, lambda r:r['source'])
    total['by_word_count'] = grouped(all_rows, all_old, all_new, old_cutoff, new_cutoff, lambda r:length_bucket(r['text']))
    published = json.loads(Path('reports/v80-result.json').read_text())
    expected = published['regression_total']['recall-98']
    parity = all(total['baseline'][k] == expected[k] for k in ('n','spam','ham','tp','fp','tn','fn'))
    parity = parity and results['Nazario-2024-247']['baseline']['tp'] == published['fresh_2024']['v80']['recall-98']['tp']
    if not parity:
        raise ValueError('Recomputed v80 does not reproduce published regression counts')
    unchanged = all(sha(p) == value for p,value in input_hashes.items())
    if (not unchanged or model_hash() != parameters_before or not torch.equal(rng_before, torch.get_rng_state()) or
        any(p.grad is not None for p in model.parameters())):
        raise ValueError('Read-only inference changed files, model, RNG or gradients')
    gate = total['passes_fp_and_individual_tp_gate'] and results['Nazario-2024-247']['baseline_spam_detections_lost'] == 0
    output = dict(request=request, frozen_100000=total, cohorts=results, v80_published_counts_reproduced=parity,
        checkpoint_and_model_files_unchanged=unchanged, model_parameters_unchanged=True, rng_unchanged=True,
        optimizer_instantiated=False, optimizer_steps_executed=0, cal_opened_for_scoring=False,
        thresholds_fitted_on_regression=False, raw_message_text_saved=False, labels_automatically_changed=False,
        independent_blind_test=False, passes_regression_gate=gate, accepted_for_production=False,
        production_v80_unchanged=True, cpu_capability=torch.backends.cpu.get_cpu_capability(),
        torch_version=str(torch.__version__), transformers_version=transformers.__version__, cpu_threads=3,
        inference_cpu_workers_max=3, inference_worker_weights_rng_and_gradients_unchanged=True,
        first_batch_per_cohort_serial_parallel_bitexact=True)
    (args.out / 'comparison.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(dict(comparison_complete=True, regression=total, passes_regression_gate=gate)), flush=True)


if __name__ == '__main__':
    main()
