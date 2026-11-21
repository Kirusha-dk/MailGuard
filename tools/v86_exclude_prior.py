"""Screen candidate identities, prefix families and quote windows against prior data.

No classifier scores, gradients, split selection or production promotion occurs.
Counts describe candidate mail, not an approved 100,000-row training cohort.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from v84_replay_data import fingerprint, read_rows, sha, windows, write_rows


def exclude(candidates, prior, output):
    if output.exists():
        raise ValueError('Output already exists; never overwrite a prior inventory')
    output.mkdir(parents=True)
    pool = {}; rejected = set(); stats = Counter(); labels = defaultdict(set)
    for path in candidates:
        for row in read_rows(path):
            if type(row['y']) is not int or row['y'] not in (0, 1):
                raise ValueError('Missing source label')
            stats['input_rows'] += 1
            if row['identity'] in pool:
                stats['duplicate_identity'] += 1
                old = pool[row['identity']]
                if old['y'] != row['y']:
                    rejected.update((row['family'], old['family']))
            else:
                pool[row['identity']] = row
            labels[row['family']].add(row['y'])
    rejected.update(family for family, ys in labels.items() if len(ys) > 1)
    family_ids = defaultdict(set); window_families = defaultdict(set)
    for row in pool.values():
        family_ids[row['family']].add(row['identity'])
        for window in windows(row['text']):
            window_families[window].add(row['family'])
    prior_counts = {}; started = time.monotonic()
    for name, directory, expected in prior:
        for part, expected_fingerprint in expected.items():
            rows = read_rows(directory / (part + '.jsonl.gz'))
            # Fingerprinting a generator checks every exact pinned input row.
            if fingerprint(rows) != expected_fingerprint:
                raise ValueError('Prior exclusion corpus drift: ' + name + '/' + part)
            count = 0
            for row in read_rows(directory / (part + '.jsonl.gz')):
                count += 1
                if row['identity'] in pool:
                    rejected.add(pool[row['identity']]['family'])
                if row['family'] in family_ids:
                    rejected.add(row['family'])
                for window in windows(row['text']):
                    rejected.update(window_families.get(window, ()))
            prior_counts[name + ':' + part] = count
            print(name, part, count, 'rejected families', len(rejected),
                  'seconds', round(time.monotonic() - started), flush=True)
    kept = sorted((r for r in pool.values() if r['family'] not in rejected),
                  key=lambda r: r['identity'])
    write_rows(output / 'candidates.jsonl.gz', kept)
    report = dict(input_files=[dict(sha256=sha(p)) for p in candidates],
        stats=dict(stats), unique_input_rows=len(pool), excluded_rows=len(pool)-len(kept),
        novel_candidate_rows=len(kept), by_class=dict(Counter(str(r['y']) for r in kept)),
        by_source_class=dict(Counter(r['source'] + ':' + str(r['y']) for r in kept)),
        candidate_fingerprint=fingerprint(kept), prior_counts=prior_counts,
        old_identity_family_and20word_window_exclusion_performed=True,
        semantic_near_duplicates_fully_excluded=False,
        splits_frozen=False, training_ready=False, classifier_scoring_performed=False,
        optimizer_steps=0, raw_mail_published=False, customer_mail_used=False)
    (output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, nargs='+', required=True)
    parser.add_argument('--old-v84', type=Path, required=True)
    parser.add_argument('--old-v85', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    v84 = json.loads(Path('research/v84/frozen.json').read_text())
    v85 = json.loads(Path('research/v85/frozen.json').read_text())
    exclude(args.candidates, [
        ('v84', args.old_v84, {k:v84[k + '_fingerprint'] for k in ('train', 'select')}),
        ('v85', args.old_v85, v85['fingerprints']),
    ], args.output)


if __name__ == '__main__':
    main()
