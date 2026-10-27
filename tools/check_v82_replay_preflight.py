"""Read-only CPU isolation using original batch boundaries and pinned audit scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PARENT_SHA = 'a5cafb2fface8a35c672e075e41e63fe4551a786b536e4e41a23026299b63f68'
CACHE_SHA = '8c09f34e382bffeb0101909df71591ccf1823e5ac2b1fa281510eba0b465e29f'


def batch_starts(scores, threshold, batch_size=32):
    if not scores:
        raise ValueError('Empty cache')
    starts = {i * batch_size for i in range(min(4, (len(scores) + batch_size - 1) // batch_size))}
    starts.add((len(scores) - 1) // batch_size * batch_size)
    closest = sorted(range(len(scores)), key=lambda i: (abs(scores[i] - threshold), i))[:12]
    starts.update(i // batch_size * batch_size for i in closest)
    return sorted(starts)


def worker(mode, output):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    from train_v82_daily import read_part
    from v82_neural import CONFIG, NeuralMail, fingerprint, load_trainable, text_tokenizer

    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    parent = Path('ablation-parent/checkpoint.pt')
    cached = Path('replay-reference/development-scores.npz')
    if sha(parent) != PARENT_SHA or sha(cached) != CACHE_SHA:
        raise ValueError('Pinned checkpoint or reference scores changed')
    reference = json.loads(Path('reports/v82-audits/step-1000-c94a687c5168.json').read_text())
    if json.loads(Path('replay-reference/audit.json').read_text()) != reference:
        raise ValueError('Reference audit JSON mismatch')
    rows = []
    for directory in ('.cache/v79/frozen', '.cache/v80/extra'):
        rows.extend(read_part(Path(directory), 'select'))
    if fingerprint(rows) != '88acb130d5c5174823ccea518339fd63cceef356945a94225e184ef04110892f':
        raise ValueError('SELECT content or order mismatch')
    with np.load(cached, allow_pickle=False) as cache:
        identities = cache['select_identity'].tolist()
        labels = cache['select_y'].tolist()
        expected = cache['baseline'].tolist()
    if identities != [r['identity'] for r in rows] or labels != [r['y'] for r in rows]:
        raise ValueError('Reference identity or label alignment mismatch')
    cutoff = json.loads(Path('reports/v82-daily/2026-10-20.json').read_text())['development']['threshold']
    starts = batch_starts(expected, cutoff)
    torch.set_num_threads(3); torch.manual_seed(82); torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        raise ValueError('CPU only')
    if mode == 'mkldnn_off':
        torch.backends.mkldnn.enabled = False
    backbone_path = Path('.cache/v82/backbone')
    manifest = json.loads((backbone_path / 'manifest.json').read_text())
    if any(sha(backbone_path / name) != value for name, value in manifest['files'].items()):
        raise ValueError('Backbone checksum mismatch')
    checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
    meta = checkpoint['metadata']
    if (meta['config'] != CONFIG or meta['backbone'] != manifest or meta['cpu_threads'] != 3 or
            meta['device_type'] != 'cpu' or meta['torch_version'] != str(torch.__version__) or
            meta['transformers_version'] != transformers.__version__ or meta['pilot_only'] or
            checkpoint['progress']['step'] != 500):
        raise ValueError('Reference metadata mismatch')
    backbone = AutoModel.from_pretrained(backbone_path, local_files_only=True,
                                         use_safetensors=True, attn_implementation='eager')
    model = NeuralMail(backbone); load_trainable(model, checkpoint['adapter']); model.eval()
    tokenizer = AutoTokenizer.from_pretrained(backbone_path, local_files_only=True)
    def model_hash():
        digest = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    before = model_hash(); rng = torch.get_rng_state().clone()
    observed, expected_subset, indices = [], [], []
    with torch.no_grad():
        for start in starts:
            batch = rows[start:start + 32]
            inputs = text_tokenizer(tokenizer, [r['text'] for r in batch], CONFIG['max_length'])
            observed.extend(model(**inputs).sigmoid().cpu().tolist())
            expected_subset.extend(expected[start:start + 32])
            indices.extend(range(start, min(start + 32, len(rows))))
    delta = [abs(a - b) for a, b in zip(observed, expected_subset)]
    exact = sum(a == b for a, b in zip(observed, expected_subset))
    changed = sum((a >= cutoff) != (b >= cutoff) for a, b in zip(observed, expected_subset))
    readonly = (model_hash() == before and torch.equal(rng, torch.get_rng_state()) and
                all(p.grad is None for p in model.parameters()) and sha(parent) == PARENT_SHA and sha(cached) == CACHE_SHA)
    if not readonly:
        raise ValueError('Replay changed protected state')
    result = dict(mode=mode, cpu_capability=torch.backends.cpu.get_cpu_capability(),
                  mkldnn_enabled=torch.backends.mkldnn.enabled, cpu_threads=torch.get_num_threads(),
                  torch_version=str(torch.__version__), transformers_version=transformers.__version__,
                  float32_matmul_precision=torch.get_float32_matmul_precision(),
                  env={k: os.environ.get(k) for k in ('ATEN_CPU_CAPABILITY', 'ONEDNN_MAX_CPU_ISA', 'DNNL_MAX_CPU_ISA')},
                  scored_rows=len(observed), batch_starts=starts, exactly_equal_scores=exact,
                  max_absolute_score_difference=max(delta), changed_decisions_at_historical_threshold=changed,
                  original_batches_preserved=True, model_and_rng_unchanged=readonly,
                  optimizer_instantiated=False, optimizer_steps_executed=0, raw_message_text_saved=False)
    output.mkdir(parents=True, exist_ok=True)
    (output / (mode + '.json')).write_text(json.dumps(result, indent=2))
    np.savez_compressed(output / (mode + '.npz'), identity=np.array([identities[i] for i in indices]),
                        reference=np.array(expected_subset), observed=np.array(observed))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['native', 'aten_avx2', 'avx2_all', 'mkldnn_off'])
    parser.add_argument('--out', type=Path, default=Path('preflight-output'))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.out)
        return
    results = []
    for mode in ('native', 'aten_avx2', 'avx2_all', 'mkldnn_off'):
        env = dict(os.environ)
        for key in ('ATEN_CPU_CAPABILITY', 'ONEDNN_MAX_CPU_ISA', 'DNNL_MAX_CPU_ISA'):
            env.pop(key, None)
        if mode in ('aten_avx2', 'avx2_all'):
            env['ATEN_CPU_CAPABILITY'] = 'avx2'
        if mode == 'avx2_all':
            env['ONEDNN_MAX_CPU_ISA'] = env['DNNL_MAX_CPU_ISA'] = 'AVX2'
        subprocess.run([sys.executable, __file__, '--worker', mode, '--out', str(args.out)], env=env, check=True)
        results.append(json.loads((args.out / (mode + '.json')).read_text()))
    summary = dict(reference_checkpoint_sha256=PARENT_SHA, reference_cache_sha256=CACHE_SHA,
                   cases=results, full_select_replay=False, optimizer_steps_executed=0,
                   interpretation='Sampled diagnostic only; exact match on these batches does not establish full replay parity')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
