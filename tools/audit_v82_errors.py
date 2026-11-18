"""Read-only TRAIN/SELECT error audit of completed v82 checkpoints; no fitting."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def metric(rows, scores, cutoff):
    if len(rows) != len(scores) or not math.isfinite(cutoff):
        raise ValueError('Invalid score alignment or threshold')
    if any(not math.isfinite(float(s)) for s in scores):
        raise ValueError('Nonfinite audit scores')
    if any(type(r['y']) is not int or r['y'] not in (0, 1) for r in rows):
        raise ValueError('Invalid audit labels')
    spam = sum(r['y'] for r in rows); ham = len(rows) - spam
    tp = sum(r['y'] == 1 and s >= cutoff for r, s in zip(rows, scores))
    fp = sum(r['y'] == 0 and s >= cutoff for r, s in zip(rows, scores))
    return dict(n=len(rows), spam=spam, ham=ham, tp=tp, fp=fp, fn=spam-tp, tn=ham-fp,
                recall=tp/spam if spam else None, fpr=fp/ham if ham else None)


def length_bucket(text):
    n = len(text.split())
    return '001-030' if n <= 30 else '031-080' if n <= 80 else '081-200' if n <= 200 else '201-500' if n <= 500 else '501+'


def describe(rows, scores, cutoff):
    result = dict(at_locked_development_threshold=metric(rows, scores, cutoff),
                  at_predefined_threshold_0_5=metric(rows, scores, .5), threshold=cutoff)
    for key, grouping in (('by_source', lambda r: r['source']),
                           ('by_word_count', lambda r: length_bucket(r['text'])),
                           ('by_character_window', lambda r: 'head-tail' if len(r['text']) > 1800 else 'whole-text')):
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[grouping(row)].append(i)
        result[key] = {k: metric([rows[i] for i in ix], [scores[i] for i in ix], cutoff)
                       for k, ix in sorted(groups.items())}
    def examples(label, predicate, descending):
        ix = [i for i, row in enumerate(rows) if row['y'] == label and predicate(scores[i])]
        ix.sort(key=lambda i: (-scores[i] if descending else scores[i], rows[i]['identity']))
        # Explicit allowlist: never serialize message text, sender, subject or URLs.
        return [dict(identity=rows[i]['identity'], source=rows[i]['source'], score=float(scores[i]),
                     characters=len(rows[i]['text']), words=len(rows[i]['text'].split()),
                     url_placeholders=rows[i]['text'].split().count('urltoken')) for i in ix[:30]]
    result['highest_scoring_false_positives'] = examples(0, lambda s: s >= cutoff, True)
    result['lowest_scoring_missed_spam'] = examples(1, lambda s: s < cutoff, False)
    return result


def paired(rows, old_scores, new_scores, old_cutoff, new_cutoff):
    baseline = metric(rows, old_scores, old_cutoff); current = metric(rows, new_scores, new_cutoff)
    old = [s >= old_cutoff for s in old_scores]; new = [s >= new_cutoff for s in new_scores]
    lost = [r['identity'] for r, a, b in zip(rows, old, new) if r['y'] == 1 and a and not b]
    return dict(baseline=baseline, current=current, baseline_threshold=old_cutoff, current_threshold=new_cutoff,
                baseline_spam_detections_lost=len(lost),
                spam_detections_gained=sum(r['y'] == 1 and not a and b for r, a, b in zip(rows, old, new)),
                false_positives_removed=sum(r['y'] == 0 and a and not b for r, a, b in zip(rows, old, new)),
                false_positives_added=sum(r['y'] == 0 and not a and b for r, a, b in zip(rows, old, new)),
                first_lost_spam_identities=lost[:30], compares_two_v82_development_checkpoints=True,
                is_a_v80_regression_check=False)


def seen_train(rows, progress, seed, limit):
    if progress['epoch'] == 0:
        order = sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:0:{r['identity']}".encode()).hexdigest())
        seen = order[:progress['cursor']]
    else:
        seen = rows
    sample = sorted(seen, key=lambda r: hashlib.sha256(('v82-audit:' + r['identity']).encode()).hexdigest())[:limit]
    return sample, len(seen)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    from train_v82_daily import read_part
    from v82_neural import CONFIG, NeuralMail, fingerprint, load_trainable, text_tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--baseline-state', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, default=Path('.cache/v82/backbone'))
    parser.add_argument('--out', type=Path, default=Path('.cache/v82-audit/output'))
    args = parser.parse_args(); request = json.loads(args.request.read_text())
    limit = request['train_seen_limit']
    if not 1 <= limit <= 10000:
        raise ValueError('Audit TRAIN limit must be between one and ten thousand')
    torch.set_num_threads(3); torch.manual_seed(CONFIG['seed']); torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, local_files_only=True)
    manifest = json.loads((args.backbone / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if sha(args.backbone / name) != expected:
            raise ValueError('Backbone checksum mismatch')
    train, select = [], []
    for directory, reference in ((Path('.cache/v79/frozen'), 'reports/v79-frozen.json'),
                                 (Path('.cache/v80/extra'), 'reports/v80-extra-frozen.json')):
        if json.loads((directory / 'frozen.json').read_text()) != json.loads(Path(reference).read_text()):
            raise ValueError('Frozen manifest drift')
        train.extend(read_part(directory, 'train')); select.extend(read_part(directory, 'select'))
    fingerprints = dict(train=fingerprint(train), select=fingerprint(select))
    state_hashes = {str(p): sha(p) for p in (args.state, args.baseline_state)}

    def load(path, minimum, exact=None):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        meta, progress = checkpoint['metadata'], checkpoint['progress']
        report = json.loads((path.parent / 'report.json').read_text())
        if (meta['campaign'] != request['campaign'] or meta['config'] != CONFIG or
            meta['pilot_only'] or meta['backbone'] != manifest or meta['device_type'] != 'cpu' or
            meta['torch_version'] != str(torch.__version__) or meta['transformers_version'] != transformers.__version__ or
            meta['cpu_threads'] != 3 or any(meta[k] != v for k, v in fingerprints.items())):
            raise ValueError('Checkpoint configuration or corpus mismatch')
        if (progress['step'] < minimum or (exact is not None and progress['step'] != exact) or
            progress['active_session'] is not None or report['global_optimizer_steps'] != progress['step'] or
            report['checkpoint_sha256'] != state_hashes[str(path)]):
            raise ValueError('Audit requires a matching, completed checkpoint and report')
        backbone = AutoModel.from_pretrained(args.backbone, local_files_only=True,
                                             use_safetensors=True, attn_implementation='eager')
        model = NeuralMail(backbone); load_trainable(model, checkpoint['adapter']); model.eval()
        return model, progress, report

    def score(model, rows, label):
        values = []
        with torch.inference_mode():
            for start in range(0, len(rows), 32):
                inputs = text_tokenizer(tokenizer, [r['text'] for r in rows[start:start+32]], CONFIG['max_length'])
                values.extend(model(**inputs).sigmoid().tolist())
                if start % 2048 == 0:
                    print(json.dumps(dict(audit_split=label, scored=min(start+32, len(rows)), total=len(rows))), flush=True)
        return values

    current, progress, report = load(args.state, request['expected_min_step'])
    sample, seen_count = seen_train(train, progress, CONFIG['seed'], limit)
    if not sample:
        raise ValueError('Checkpoint has no observed training examples')
    train_scores = score(current, sample, 'seen-TRAIN')
    select_scores = score(current, select, 'SELECT')
    cutoff = report['development']['threshold']
    result = dict(campaign=request['campaign'], global_optimizer_steps=progress['step'],
                  current_checkpoint_sha256=state_hashes[str(args.state)],
                  corpus_fingerprints=fingerprints, seen_train_sample_fingerprint=fingerprint(sample),
                  backbone_revision=CONFIG['revision'], train_seen_population=seen_count,
                  seen_train_sample=describe(sample, train_scores, cutoff),
                  select=describe(select, select_scores, cutoff), source_report=report['development'],
                  raw_message_text_saved=False, optimizer_steps_executed=0, cal_or_test_opened=False,
                  independent_quality_test_performed=False, quality_gain_demonstrated=False,
                  production_v80_unchanged=True, labels_automatically_changed=False)
    observed = result['select']['at_locked_development_threshold']
    result['source_report_counts_reproduced'] = all(observed[k] == report['development'][k] for k in ('n', 'spam', 'ham', 'tp', 'fp'))
    del current
    old, old_progress, old_report = load(args.baseline_state, request['baseline_step'], request['baseline_step'])
    baseline_scores = score(old, select, 'baseline-SELECT')
    result['baseline_step'] = old_progress['step']
    result['baseline_checkpoint_sha256'] = state_hashes[str(args.baseline_state)]
    result['paired_select'] = paired(select, baseline_scores, select_scores,
                                     old_report['development']['threshold'], cutoff)
    if any(sha(path) != expected for path, expected in state_hashes.items()):
        raise ValueError('Audit modified a checkpoint')
    result['checkpoint_files_unchanged'] = True
    result['audit_code_sha256'] = sha(__file__)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / 'development-scores.npz',
        select_identity=np.array([r['identity'] for r in select]), select_y=np.array([r['y'] for r in select]),
        current=np.array(select_scores), baseline=np.array(baseline_scores),
        train_identity=np.array([r['identity'] for r in sample]), train_y=np.array([r['y'] for r in sample]),
        train_scores=np.array(train_scores))
    (args.out / 'audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(audit_completed=True, step=progress['step'], checkpoint_files_unchanged=True,
                          select=observed, paired_select=result['paired_select']), indent=2), flush=True)


if __name__ == '__main__':
    main()
