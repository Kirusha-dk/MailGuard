import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from check_rspamd_runtime import validate_scan


class RspamdRuntimeCheckTests(unittest.TestCase):
    def test_missing_or_invalid_scan_never_becomes_zero_score(self):
        for value in ({}, [], {'error': 'bad'}, {'action': 'no action', 'score': 'NaN', 'required_score': 6},
                      {'action': 'no action', 'score': True, 'required_score': 6}):
            with self.assertRaises(ValueError):
                validate_scan(value)

    def test_valid_zero_score_and_gtube_symbol(self):
        result = validate_scan(dict(action='no action', score=0, required_score=6))
        self.assertEqual(result['score'], 0)
        result = validate_scan(dict(action='reject', score=1000, required_score=6, symbols={'GTUBE': {}}))
        self.assertTrue(result['gtube_detected'])


if __name__ == '__main__':
    unittest.main()
