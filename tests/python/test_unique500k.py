import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from evaluate_v54_unique500k import Pool, content_keys, csdmc_label
from train_v47_phishing_aware import canonical_fields


class UniqueMailTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.pool = Pool(Path(self.temp.name) / 'pool.sqlite')

    def tearDown(self):
        self.pool.close()
        self.temp.cleanup()

    def row(self, body, subject='First', sender='first@example.org'):
        return canonical_fields(subject, body, sender)

    def test_body_duplicates_ignore_subject_and_sender(self):
        body = 'The attached notes summarize our meeting today.'
        self.pool.add(self.row(body), 0, 'one', '1')
        self.pool.add(self.row(body, 'Other', 'different@example.org'), 0, 'two', '2')
        self.assertEqual(self.pool.count(), 1)

    def test_tracking_variants_are_duplicates(self):
        self.pool.add(self.row('View offer 123 at https://one.org/id=4 for a@one.org'), 1, 'one', '1')
        self.pool.add(self.row('View offer 987 at https://two.org/id=5 for b@two.org'), 1, 'two', '2')
        self.assertEqual(self.pool.count(), 1)

    def test_old_content_and_its_variants_remain_blocked(self):
        self.pool.block(self.row('Confirm reference 123 using https://one.org/id=4'))
        self.pool.add(self.row('Confirm reference 999 using https://two.org/id=5', 'New subject'), 1, 'new', '1')
        self.assertEqual(self.pool.count(), 0)
        self.assertEqual(self.pool.stats['oldContent'], 1)

    def test_conflicting_labels_remove_retained_row(self):
        row = self.row('A sufficiently long email with disputed labels.')
        self.pool.add(row, 0, 'one', '1')
        self.pool.add(row, 1, 'two', '2')
        self.pool.add(row, 0, 'three', '3')
        self.assertEqual(self.pool.count(), 0)

    def test_distinct_content_is_retained(self):
        self.pool.add(self.row('The meeting will cover annual staffing plans.'), 0, 'one', '1')
        self.pool.add(self.row('Congratulations you won an unexpected luxury yacht.'), 1, 'two', '2')
        self.assertEqual(self.pool.count(), 2)

    def test_original_csdmc_mapping_is_reversed(self):
        self.assertEqual(csdmc_label('1'), 0)  # original ham=1
        self.assertEqual(csdmc_label('0'), 1)  # original spam=0
        with self.assertRaises(ValueError):
            csdmc_label('2')


if __name__ == '__main__':
    unittest.main()
