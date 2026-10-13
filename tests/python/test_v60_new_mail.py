import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from prepare_v60_new_mail import partition


class V60Isolation(unittest.TestCase):
    def test_may_is_always_test(self):
        for n in range(100):
            family = '0' * 52 + f'{n:012x}'
            self.assertEqual(partition(family, 'SpamArchive-2026-05'), 'test')
            self.assertNotEqual(partition(family, 'SpamArchive-2026-04'), 'test')

    def test_used_month_is_rejected(self):
        with self.assertRaises(ValueError):
            partition('0' * 64, 'SpamArchive-2026-09')

    def test_original_ham_never_trains(self):
        self.assertEqual(partition('0' * 64, 'SpamAssassin-original-20030228_hard_ham'), 'test')


if __name__ == '__main__':
    unittest.main()
