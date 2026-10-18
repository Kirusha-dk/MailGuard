"""Independently recompute extra identities and check all original overlaps."""
import argparse
import gzip
import json
from pathlib import Path

from v78_data import identifiers
from v79_data import fingerprint


def read(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def audit(original, extra):
    original_ids, original_families = set(), set()
    for name in ('train', 'select', 'cal', 'test'):
        for row in read(original / (name + '.jsonl.gz')):
            original_ids.add(row['identity']); original_families.add(row['family'])
    manifest = json.loads((extra / 'frozen.json').read_text())
    seen, families, counts = set(), {}, {}
    for name in ('train', 'select', 'cal', 'fresh-test'):
        rows = list(read(extra / (name + '.jsonl.gz')))
        for row in rows:
            if identifiers(row['text']) != (row['identity'], row['family']):
                raise ValueError('Canonical identity mismatch')
            if row['identity'] in original_ids or row['family'] in original_families:
                raise ValueError('Extra mail overlaps an original v79 partition')
            if row['identity'] in seen:
                raise ValueError('Duplicate extra identity')
            if row['y'] != 1:
                raise ValueError('Unexpected curated archive label')
            if name == 'fresh-test' and row['source'] != 'Nazario-2024':
                raise ValueError('Unexpected fresh-test source')
            seen.add(row['identity'])
        if fingerprint(rows) != manifest['fingerprints'][name] or len(rows) != manifest['counts'][name]:
            raise ValueError('Frozen extra partition drift')
        families[name] = {r['family'] for r in rows}
        counts[name] = dict(n=len(rows), prefix_groups=len(families[name]))
    for i, first in enumerate(families):
        for second in list(families)[i + 1:]:
            if families[first] & families[second]:
                raise ValueError('Extra prefix-family leakage')
    return dict(passed=True, extra_identities_recomputed=True,
                no_overlap_with_any_v79_partition=True,
                no_extra_identity_or_prefix_family_leakage=True,
                fresh_2024_is_phishing_only=True, counts=counts)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, default=Path('.cache/v79/frozen'))
    p.add_argument('--extra', type=Path, default=Path('.cache/v80/extra'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); result = audit(a.original, a.extra)
    a.output.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
