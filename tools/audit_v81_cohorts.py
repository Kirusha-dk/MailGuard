"""Verify the additional fitting split and all reused evaluation boundaries."""
import argparse
import json
from collections import Counter
from pathlib import Path

import v77_mail as core
from audit_v80_cohorts import audit, read
from v79_data import fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('.cache/v79/frozen'))
    p.add_argument('--extra', type=Path, default=Path('.cache/v80/extra'))
    p.add_argument('--output', type=Path, default=Path('reports/v81-data-audit.json'))
    a = p.parse_args()
    extra_audit = audit(a.data, a.extra)
    parts = {n: [] for n in ('train', 'select-fit', 'select-choose', 'cal', 'regression-100k', 'regression-2024')}
    for directory, is_extra in ((a.data, False), (a.extra, True)):
        manifest = json.loads((directory / 'frozen.json').read_text())
        for part in ('train', 'select', 'cal', 'fresh-test' if is_extra else 'test'):
            rows = list(read(directory / (part + '.jsonl.gz')))
            expected = (manifest['fingerprints'][part] if is_extra else
                        manifest['test_fingerprint'] if part == 'test' else manifest['development_fingerprints'][part])
            if fingerprint(rows) != expected:
                raise ValueError('Frozen cohort drift')
            for row in rows:
                if part == 'select':
                    name = 'select-fit' if int(core.digest('v80-meta:' + row['family'])[:10], 16) % 2 == 0 else 'select-choose'
                else:
                    name = {'test': 'regression-100k', 'fresh-test': 'regression-2024'}.get(part, part)
                parts[name].append(row)
    ids = {k: {r['identity'] for r in rows} for k, rows in parts.items()}
    families = {k: {r['family'] for r in rows} for k, rows in parts.items()}
    for i, first in enumerate(parts):
        if len(ids[first]) != len(parts[first]):
            raise ValueError('Duplicate identity')
        for second in list(parts)[i + 1:]:
            if ids[first] & ids[second] or families[first] & families[second]:
                raise ValueError('Fitting/evaluation overlap')
    report = dict(passed=True, extra_audit=extra_audit, all_six_cohorts_identity_and_prefix_disjoint=True,
                  original_sources_are_not_disjoint=True, both_evaluation_sets_previously_examined=True,
                  counts={k: dict(n=len(v), spam=sum(r['y'] for r in v),
                                  ham=sum(1-r['y'] for r in v), prefix_groups=len(families[k]),
                                  fingerprint=fingerprint(v), sources=dict(Counter(r['source'] for r in v)))
                          for k, v in parts.items()})
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
