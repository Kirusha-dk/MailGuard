"""Additional curated phishing years; hold out the whole 2024 archive."""
import argparse
import gzip
import json
import mailbox
from collections import Counter
from pathlib import Path

import fasttext
import v77_mail as core
from v78_data import identifiers
from v79_data import normalize, english_prediction, fingerprint
from v80_encoder import file_sha

SEED = 'v80-extra-phishing-whole-2024-held-out-2026-10-18'
ARCHIVE_SHA = {
    2015: '7523e62a5e6e4ee8134ab8b1a628e423097a7779e1003b28422864aa34f1c319',
    2016: '42393e4d686193bfb879d1608c23fd4d3b698ccd4d73f0e1d2276ba709718483',
    2017: '110fba876b38b44035bf065788952674bc68a7b6879c345a515a9ac844b1d704',
    2018: '83f49cf2a76c7b29b205d22d8778e46848e36af0f15ee1df126a543696acd48d',
    2019: 'cab39d6fcb38cf80cdaabb3e6ffd1222777e2fcaac7e8480321b7f729c2e3c0e',
    2020: '99e20bf25f5e8556d37103f2c8140cd75181a62d3df91df8acf9d9aece9f08bb',
    2021: '636707cff42245d1482e4ae8a2c03369717a5e06bf9d1a0f4ec5dcd846661cee',
    2022: 'a6291e2cb990204766fce425423949b403cce4ae6a660fc8d08812e40a063fa7',
    2023: '5007f1c1dc91b15e93e988afa3a1cf402e15c88f765d3057be8bfec50dae1fe8',
    2024: '60afa40a2757170093a2bde31ec9cda4e54d33992203ce04bfbbc86d2e5e5b99',
}


def prepare(data, raw, output):
    known_ids, known_families = set(), set()
    for name in ('train', 'select', 'cal', 'test'):
        with gzip.open(data / (name + '.jsonl.gz'), 'rt') as stream:
            for line in stream:
                r = json.loads(line)
                known_ids.add(r['identity']); known_families.add(r['family'])
    lid = fasttext.load_model(str(data / 'lid.176.ftz'))
    seen, years, counts, hashes = set(), {}, Counter(), {}
    for year in range(2015, 2025):
        path = raw / ('nazario-' + str(year) + '.mbox')
        hashes[str(year)] = file_sha(path)
        if hashes[str(year)] != ARCHIVE_SHA[year]:
            raise ValueError('Phishing archive drift: ' + str(year))
        box = mailbox.mbox(path)
        rows = []
        for message in box:
            counts[str(year) + ':raw'] += 1
            text = normalize(core.parse_eml(message.as_bytes()))
            if not 30 <= len(text) <= 250000 or core.language(text) != 'en' or not english_prediction(lid, text)[0]:
                counts[str(year) + ':unsupported'] += 1; continue
            identity, family = identifiers(text)
            if identity in known_ids or family in known_families:
                counts[str(year) + ':existing_family_excluded'] += 1; continue
            if identity in seen:
                counts[str(year) + ':duplicate'] += 1; continue
            seen.add(identity)
            rows.append(dict(text=text, identity=identity, family=family, y=1,
                             source='Nazario-' + str(year), kind='phish'))
        box.close(); years[year] = rows
        print('Curated phishing year', year, len(rows), flush=True)
    # The later year has priority; earlier overlapping families cannot be fitted.
    protected = {r['family'] for r in years[2024]}
    parts = {k: [] for k in ('train', 'select', 'cal')}
    for year in range(2015, 2024):
        for row in years[year]:
            if row['family'] in protected:
                counts['earlier_year_matches_2024_family_excluded'] += 1; continue
            bucket = int(core.digest(SEED + row['family'])[:12], 16) % 100
            name = 'train' if bucket < 70 else 'select' if bucket < 80 else 'cal'
            parts[name].append(row)
    parts['fresh-test'] = years[2024]
    if len(parts['fresh-test']) < 100:
        raise ValueError('Too few new 2024 emails for the declared stress test')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(seed=SEED, source='https://monkey.org/~jose/phishing/',
                    archive_sha256=hashes, audit=dict(counts),
                    counts={k: len(v) for k, v in parts.items()},
                    fingerprints={k: fingerprint(v) for k, v in parts.items()},
                    fresh_test_year=2024, fresh_test_is_phishing_only=True,
                    fresh_test_can_measure_false_positives=False,
                    fresh_test_is_a_new_publisher=False,
                    regression_100k_was_already_examined_in_v79=True,
                    original_v79_identities_and_prefix_groups_excluded=True)
    for name, rows in parts.items():
        with gzip.open(output / (name + '.jsonl.gz'), 'wt') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
    (output / 'frozen.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('.cache/v79/frozen'))
    p.add_argument('--raw', type=Path, default=Path('.cache/v80/raw'))
    p.add_argument('--out', type=Path, default=Path('.cache/v80/extra'))
    a = p.parse_args(); prepare(a.data, a.raw, a.out)
