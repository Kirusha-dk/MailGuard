"""Read-only SELECT development threshold frontier; never train or change requests."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from v82_neural import NeuralMail, load_trainable, text_tokenizer
from v84_replay import confusion, fp_budget_points
from v84_replay_data import fingerprint, read_rows, sha


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scores', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.state / 'checkpoint.pt'
    before = sha(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    report = json.loads((args.state / 'report.json').read_text())
    assert before == report['checkpoint_sha256']
    assert report['session_complete'] and saved['progress']['step'] == report['global_optimizer_steps']
    frozen = json.loads(Path('research/v84/frozen.json').read_text())
    assert saved['metadata']['corpus'] == frozen
    rows = list(read_rows(Path('.cache/v84/corpus/select.jsonl.gz')))
    assert fingerprint(rows) == frozen['select_fingerprint']
    backbone = Path('.cache/v82/backbone')
    manifest = json.loads((backbone / 'manifest.json').read_text())
    assert manifest == saved['metadata']['backbone']
    for name, digest in manifest['files'].items():
        assert sha(backbone / name) == digest
    torch.set_num_threads(3)
    torch.manual_seed(84)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(backbone, local_files_only=True)
    model = NeuralMail(AutoModel.from_pretrained(backbone, local_files_only=True,
                         use_safetensors=True, attn_implementation='eager'))
    load_trainable(model, saved['adapter'])
    model.eval()
    values = []
    if args.scores.is_file():
        with np.load(args.scores, allow_pickle=False) as cached:
            if str(cached['checkpoint_sha256']) != before:
                raise ValueError('Score cache belongs to another checkpoint')
            values = cached['scores'].tolist()
            if (len(values) > len(rows) or cached['identity'].tolist() !=
                    [r['identity'] for r in rows[:len(values)]] or
                    cached['y'].tolist() != [r['y'] for r in rows[:len(values)]] or
                    not np.isfinite(values).all()):
                raise ValueError('Invalid partial score cache')
        print(f'Restored {len(values)} scores from the same checkpoint', flush=True)
    args.scores.parent.mkdir(parents=True, exist_ok=True)

    def save_scores():
        temporary = args.scores.with_suffix('.partial.npz')
        np.savez_compressed(temporary, scores=np.asarray(values),
            y=np.array([r['y'] for r in rows[:len(values)]]),
            source=np.array([r['source'] for r in rows[:len(values)]]),
            identity=np.array([r['identity'] for r in rows[:len(values)]]),
            checkpoint_sha256=np.array(before))
        temporary.replace(args.scores)

    with torch.inference_mode():
        for start in range(len(values), len(rows), 32):
            values.extend(model(**text_tokenizer(tokenizer,
                [r['text'] for r in rows[start:start + 32]], 192)).sigmoid().tolist())
            if len(values) % 256 == 0 or len(values) == len(rows):
                save_scores()
            if start % 2048 == 0:
                print(f'Scored {len(values)}/{len(rows)} SELECT rows; optimizer steps=0', flush=True)
    y = np.array([r['y'] for r in rows])
    scores = np.asarray(values)
    assert np.isfinite(scores).all()
    source = np.array([r['source'] for r in rows])
    original = report['development']
    measured = confusion(y, scores, original['threshold'])
    for key in ('n', 'spam', 'ham', 'tp', 'fp', 'fn', 'tn'):
        assert measured[key] == original[key], (key, measured[key], original[key])

    def describe(threshold):
        return dict(**confusion(y, scores, float(threshold)),
            by_source={s: confusion(y[source == s], scores[source == s], float(threshold))
                       for s in sorted(set(source))})

    frontier = {budget: describe(point['threshold'])
                for budget, point in fp_budget_points(y, scores).items()}
    result = dict(run_url='https://github.com/Kirusha-dk/MailGuard/actions/runs/42025999444',
        checkpoint_sha256=before, global_optimizer_steps=saved['progress']['step'],
        is_development=True, is_independent_test=False,
        thresholds_selected_on_same_development=True, optimizer_steps_executed=0,
        request_changed=False, production_changed=False,
        completed_training_rows=report['session_rows'], completed_training_steps=report['session_optimizer_steps'],
        original_report_counts_reproduced=True,
        current_fitted98=describe(original['threshold']),
        fixed_parent_threshold=describe(original['fixed_parent_threshold']['threshold']),
        threshold_05=describe(.5), fp_budget_frontier=frontier,
        checkpoint_unchanged=sha(checkpoint) == before,
        note='FP budget thresholds fitted to SELECT HAM labels; exploratory development only. No source-specific exemptions, relabeling, optimizer, or production changes.')
    assert result['checkpoint_unchanged']
    save_scores()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: {name: r[name] for name in ('threshold', 'tp', 'fp', 'fn', 'tn', 'recall', 'fpr')}
                      for k, r in frontier.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
