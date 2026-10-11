#!/usr/bin/env python3
"""Exercise the offline customer CLI with a real frozen bundle and tiny EML files.

Fixtures test loading, parsing and report accounting, not classifier accuracy.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--report')
    args = parser.parse_args()
    modeldir = Path(args.model_dir).resolve()
    model_version = joblib.load(modeldir / 'model.joblib')['version']
    expected_version = {'v56-consensus':56,'v57-preserved-consensus':57,'v58-calibrated-protected-consensus':58}[model_version]
    checker = Path(__file__).with_name('check_customer_mail.py')
    before = {str(p.relative_to(modeldir)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in modeldir.rglob('*') if p.is_file() and p.suffix in ('.joblib', '.pt')}
    probe_results = {}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        ham, spam = root / 'ham', root / 'spam'
        ham.mkdir(); spam.mkdir()
        (ham / 'meeting.eml').write_text('From: alice@example.org\nTo: bob@example.org\nSubject: Meeting tomorrow\n\nPlease bring the report to our meeting tomorrow.\n')
        (ham / 'japanese.EML').write_text('From: alice@example.org\nTo: bob@example.org\nSubject: 会議の日程\nContent-Type: text/plain; charset=utf-8\n\n明日の会議は午後三時から始まります。よろしくお願いします。\n')
        (spam / 'offer.eml').write_text('From: offers@example.org\nTo: bob@example.org\nSubject: Claim your reward\n\nClaim your free cash prize today at https://offers.example.org/reward\n')
        (spam / 'offer-duplicate.eml').write_bytes((spam / 'offer.eml').read_bytes())
        for profile in ('conservative', 'high-recall'):
            output = root / profile
            command = [sys.executable, str(checker), '--model-dir', str(modeldir), '--spam', str(spam), '--ham', str(ham),
                       '--output', str(output), '--profile', profile, '--batch-size', '2']
            subprocess.run(command, check=True)
            report = json.loads((output / 'summary.json').read_text())
            assert (report['version'], report['profile'], report['found'], report['scored'], report['failed']) == (expected_version, profile, 4, 4, 0)
            assert report['ham'] == 2 and report['spam'] == 2
            assert report['tp'] + report['fn'] == 2 and report['fp'] + report['tn'] == 2
            assert report['duplicateContent'] == 1 and report['conflictingDuplicateLabels'] == 0
            assert report['thresholdChanged'] is False
            with (output / 'predictions.csv').open() as f:
                predictions = list(csv.DictReader(f))
            japanese = next(r for r in predictions if r['file'] == 'japanese.EML')
            print(f"Synthetic Japanese HAM probe / {profile}: predictedSpam={japanese['predicted']}", flush=True)
            probe_results[profile] = dict(japaneseHamPredictedSpam=bool(int(japanese['predicted'])),
                                         syntheticHamFp=report['fp'], syntheticHamCount=2)
            # A second invocation must not overwrite a completed report.
            rerun = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert rerun.returncode != 0
            assert 'Output already contains a report' in rerun.stderr.decode()
    after = {str(p.relative_to(modeldir)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in modeldir.rglob('*') if p.is_file() and p.suffix in ('.joblib', '.pt')}
    assert before == after, 'Customer check changed the model bundle'
    if args.report:
        Path(args.report).write_text(json.dumps(dict(technicalChecksPassed=True, profiles=probe_results,
                                                    independentBenchmark=False,
                                                    note='Four synthetic fixtures test transport and expose individual mistakes; they do not measure production quality.'), indent=2))
    print('Offline customer CLI: both profiles, EML/EML parsing, duplicates, report accounting, and overwrite protection passed.')


if __name__ == '__main__':
    main()
