"""Regression runner contract checks with synthetic messages and no model."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
DOTNET = os.environ.get('MAILGUARD_TEST_DOTNET') or shutil.which('dotnet')
RUNNER = ROOT / 'tools/runtime_regression/bin/Release/net8.0/RuntimeRegression.dll'


@unittest.skipUnless(DOTNET and RUNNER.exists(), 'Build Release runtime regression runner first')
class RuntimeRegressionCliTests(unittest.TestCase):
    def test_hash_only_receipt_is_complete_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'messages/ham').mkdir(parents=True)
            (root / 'messages/spam').mkdir(parents=True)
            (root / 'messages/ham/one.eml').write_bytes(b'Subject: private-subject-canary\r\n\r\nprivate-body-canary')
            (root / 'messages/spam/two.eml').write_bytes(b'Subject: second\r\n\r\nsecond body')
            out = root / 'receipt.jsonl'
            cmd = [DOTNET, str(RUNNER), '--input', str(root / 'messages'), '--output', str(out)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = out.read_text()
            self.assertNotIn('private-body-canary', saved)
            self.assertNotIn('private-subject-canary', saved)
            rows = [json.loads(line) for line in saved.splitlines()]
            self.assertEqual(rows[0]['selectedMessageCount'], 2)
            self.assertFalse(rows[0]['classificationAvailable'])
            self.assertFalse(rows[0]['deploymentAcceptanceEstablished'])
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row['parseSuccess'] for row in rows[1:]))
            retry = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual(out.read_text(), saved)


if __name__ == '__main__':
    unittest.main()
