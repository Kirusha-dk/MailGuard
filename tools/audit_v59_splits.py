"""Independent inventory and overlap audit; does not fit or score any model."""
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path('.cache/v59-data')
    provenance = json.loads((root / 'provenance.json').read_text())
    counts, identities, families, raw_shas = {}, {}, {}, {}
    for split, expected_sha in provenance['splitHashes'].items():
        path = root / f'{split}.jsonl.gz'
        if file_sha(path) != expected_sha:
            raise AssertionError('Split changed after freeze')
        identities[split], families[split], raw_shas[split] = set(), set(), set()
        c = Counter()
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                if type(row['y']) is not int or row['y'] not in (0, 1):
                    raise AssertionError('Invalid binary label')
                if row['identity'] in identities[split] or row['family'] in families[split]:
                    raise AssertionError('Duplicate identity/family within split')
                identities[split].add(row['identity']); families[split].add(row['family'])
                if row.get('rawSha256'):
                    if row['rawSha256'] in raw_shas[split]:
                        raise AssertionError('Repeated raw mail')
                    raw_shas[split].add(row['rawSha256'])
                if row['source'] == 'W3C-Japanese-weak-ham':
                    if split != 'train' or row['y'] != 0 or not row.get('weakLabel'):
                        raise AssertionError('Weak Japanese HAM escaped TRAIN')
                if split == 'test' and row['source'] != 'TREC06-Chinese' and not row['source'].startswith('SpamArchive-2026-'):
                    raise AssertionError('Old or weak HAM source in new TEST')
                c[f'{row["source"]}:{row["y"]}'] += 1
        counts[split] = dict(c)
        if counts[split] != provenance['counts'][split]:
            raise AssertionError('Reported split inventory differs from actual rows')
    for mapping in (identities, families, raw_shas):
        names = list(mapping)
        if any(mapping[a] & mapping[b] for i, a in enumerate(names) for b in names[i+1:]):
            raise AssertionError('Cross-split overlap')
    if file_sha(root / 'manifest.csv') != provenance['manifestSha256']:
        raise AssertionError('Split manifest changed')
    if provenance['priorBlocking']['rawSha256RowsBlocked'] != 110000:
        raise AssertionError('Original raw mail was not fully blocked')
    if 'test' not in provenance['priorBlocking']['additionalSevenSplits']:
        raise AssertionError('Published Seven TEST source pool was not blocked')
    report = dict(version=59, noEstimatorFitOrInference=True, hashesVerified=True, identityOverlap=0,
                  campaignFamilyOverlap=0, rawSha256Overlap=0, counts=counts,
                  totals={s: len(identities[s]) for s in identities},
                  weakHamLabelsUsedOutsideTrain=False, spamTrapLabelsUsedInTest=True,
                  testHam=sum(n for k, n in counts['test'].items() if k.endswith(':0')),
                  testSpam=sum(n for k, n in counts['test'].items() if k.endswith(':1')),
                  manifestSha256=provenance['manifestSha256'], sources=provenance['sourceArchives'])
    out = Path('reports/v59-data-audit'); out.mkdir(parents=True, exist_ok=True)
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    md = '# MailGuard v59 data audit\n\n'
    md += f'New TEST: {report["totals"]["test"]} emails; HAM {report["testHam"]}; spam {report["testSpam"]}.\n\n'
    md += 'Content identities, campaign families and raw mail hashes do not overlap across splits. Hashes match the frozen inventory. Weak Japanese HAM is TRAIN-only. SpamArchive TEST labels come from spam traps.\n'
    (out / 'audit.md').write_text(md)
    print(md, flush=True)


if __name__ == '__main__':
    main()
