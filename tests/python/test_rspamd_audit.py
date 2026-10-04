import json
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from rspamd_benchmark_audit import parse_scan, request_scan, check_learning_reply, scan_health, load_frozen_v40

class AuditTests(unittest.TestCase):
    def valid(self):
        return dict(score=6.2, required_score=15, action='add header',
                    symbols={'BAYES_SPAM': {'score': 5.1}})

    def test_valid_scan(self):
        row = parse_scan(self.valid())
        self.assertTrue(row['rspam'])
        self.assertEqual(row['rscore'], 6.2)
        self.assertEqual(scan_health([row])['bayesHits'], 1)

    def test_missing_scores_are_not_zero(self):
        for key in ('score', 'required_score', 'action', 'symbols'):
            with self.subTest(key=key):
                reply = self.valid(); del reply[key]
                with self.assertRaises(ValueError): parse_scan(reply)

    def test_invalid_and_deferred_responses(self):
        for change in ({'score': float('nan')}, {'score': True}, {'error': 'failed'},
                       {'is_skipped': True}, {'action': 'soft reject'},
                       {'symbols': {'BAYES_SPAM': {}}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_scan(self.valid() | change)

    def test_learning_requires_explicit_success(self):
        check_learning_reply({'success': True})
        for reply in ({}, {'success': False}, {'success': 'true'}, {'success': True, 'error': 'x'}):
            with self.subTest(reply=reply), self.assertRaises(ValueError):
                check_learning_reply(reply)

    def test_zero_baseline_and_absent_bayes_fail(self):
        for reply in (self.valid() | {'score': 0}, self.valid() | {'symbols': {}}):
            with self.assertRaises(ValueError): scan_health([parse_scan(reply)])

    def test_full_scan_header_is_sent(self):
        with patch('rspamd_benchmark_audit.urlopen') as mock:
            mock.return_value.__enter__.return_value.read.return_value = json.dumps(self.valid()).encode()
            request_scan('http://localhost:11333/checkv2', b'test')
            request = mock.call_args.args[0]
            self.assertEqual(request.get_header('Flags'), 'pass_all')
            self.assertEqual(request.data, b'test')

class SplitTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        (root / 'eml').mkdir()
        manifest = {'version': 40}
        for i, split in enumerate(('train', 'val', 'test')):
            (root / 'eml' / (split + '.eml')).write_bytes(('message ' + split).encode())
            manifest[split] = [dict(file=split + '.eml', y=i % 2, source='fixture')]
        (root / 'manifest.json').write_text(json.dumps(manifest))
        return root, manifest

    def test_frozen_ids_are_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.fixture(tmp)
            rows, audit = load_frozen_v40(root, dict(train=1, val=1, test=1))
            self.assertEqual(audit, load_frozen_v40(root, dict(train=1, val=1, test=1))[1])
            self.assertEqual(rows['test'][0]['path'].read_bytes(), b'message test')

    def test_train_test_overlap_rejected_even_with_different_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.fixture(tmp)
            (root / 'eml/test.eml').write_bytes(b'message val')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_frozen_v40(root, dict(train=1, val=1, test=1))

    def test_missing_file_stops_instead_of_rebuilding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.fixture(tmp)
            (root / 'eml/test.eml').unlink()
            with self.assertRaises(FileNotFoundError):
                load_frozen_v40(root, dict(train=1, val=1, test=1))

if __name__ == '__main__': unittest.main()

