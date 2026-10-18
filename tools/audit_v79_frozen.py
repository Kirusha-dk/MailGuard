"""Independently check frozen identities, block sizes and split isolation."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from v78_data import identifiers
from v79_data import fingerprint


def audit(root):
    manifest = json.loads((root / 'frozen.json').read_text())
    families, seen, result = {}, set(), {}
    for name in ('train', 'select', 'cal', 'test'):
        rows, counts = [], Counter()
        with gzip.open(root / (name + '.jsonl.gz'), 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                assert type(row['y']) is int and row['y'] in (0, 1), 'Invalid label'
                assert identifiers(row['text']) == (row['identity'], row['family']), 'Identity drift'
                assert row['identity'] not in seen, 'Duplicate identity across frozen data'
                seen.add(row['identity'])
                rows.append({k: row[k] for k in ('identity', 'family', 'y', 'source')})
                counts[row['source'] + ':' + str(row['y'])] += 1
        families[name] = {r['family'] for r in rows}
        expected = manifest['test_fingerprint'] if name == 'test' else manifest['development_fingerprints'][name]
        assert fingerprint(rows) == expected, 'Partition fingerprint drift'
        expected_count = 100000 if name == 'test' else manifest['development_counts'][name]
        assert len(rows) == expected_count, 'Partition count drift'
        if name == 'test':
            assert all(r['source'] != 'Merged-public' for r in rows), 'Unverified email provenance'
            for index in range(10):
                block = rows[index * 10000:(index + 1) * 10000]
                frozen = manifest['blocks'][index]
                assert len(block) == frozen['n'] == 10000
                assert fingerprint(block) == frozen['fingerprint'], 'Block order drift'
                assert sum(r['y'] for r in block) == frozen['spam'], 'Block label count drift'
                assert dict(Counter(r['source'] for r in block)) == frozen['sources'], 'Block source drift'
        result[name] = {'n': len(rows), 'families': len(families[name]), 'source_class_counts': dict(counts)}
    names = list(families)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            assert not families[first] & families[second], 'Prefix family leakage'
    return {'passed': True, 'all_identities_recomputed': True, 'all_identities_unique': True,
            'all_partition_and_block_fingerprints_match': True,
            'train_select_cal_test_prefix_groups_disjoint': True,
            'test_contains_only_mail_sources': True, 'partitions': result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('.cache/v79/frozen'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
