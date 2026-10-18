"""Cross-check canonical text labels without relabelling either dataset.

Canonical equality removes URLs, addresses and numbers. A conflict does not
prove which publisher is right, nor byte-identical original messages. It makes
the merged provider unsuitable as trusted ground truth for this text view.
"""
import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from v78_data import iter_array


def crosscheck(original, external):
    by_id = {r['identity']: r for r in external}
    by_family = defaultdict(set)
    for row in external:
        by_family[row['family']].add(row['y'])
    exact = Counter(); prefix = Counter(); examples = []; seen = set()
    for row in original:
        if row['lang'] != 'en' or row['identity'] in seen:
            continue
        seen.add(row['identity'])
        other = by_id.get(row['identity'])
        if other is not None:
            exact[f"original_{row['y']}_external_{other['y']}"] += 1
            if row['y'] != other['y'] and len(examples) < 5:
                examples.append({k: row[k] for k in ('identity', 'source')} |
                                {'original_label': row['y'], 'external_label': other['y']})
        for label in by_family.get(row['family'], ()):
            prefix[f"original_{row['y']}_external_{label}"] += 1
    return {'exact_content_label_crosscheck': dict(exact),
            'prefix_label_crosscheck': dict(prefix),
            'conflicting_identity_examples_no_text': examples,
            'comparison_view': 'v77 NFKC/casefold, URLs/addresses/numbers replaced; punctuation retained',
            'original_email_revision': 'deb4cc27dc2754c14cb157aace412727f77b41dd',
            'external_provider': 'sniff07/phishing-dataset',
            'external_revision': 'c21f524c014a49626dc61322a5f7b3194ac5c7d1',
            'decision': 'quarantine provider; no relabelling by model predictions',
            'limitation': 'Canonical matches need not be byte-identical original messages; no independent adjudication of every conflict.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', default='.cache/v77/rows.json.gz')
    parser.add_argument('--external', default='.cache/v77/external.json.gz')
    parser.add_argument('--output', default='reports/v78-source-label-audit.json')
    args = parser.parse_args()
    with gzip.open(args.external, 'rt') as stream:
        external = json.load(stream)['rows']
    with gzip.open(args.original, 'rt') as stream:
        report = crosscheck(iter_array(stream), external)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
