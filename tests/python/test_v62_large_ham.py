import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from evaluate_v62_large_ham import HamPool
from evaluate_v62_500k import measure


class GoldHamAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.pool = HamPool(Path(self.temp.name)/'pool.sqlite')

    def tearDown(self):
        self.pool.db.close()
        self.temp.cleanup()

    def add(self,keys,label=0,identity='id',families=()):
        self.pool.add(keys,families,label,identity,'family','canonical text','source','origin')

    def count(self):
        return self.pool.db.execute('SELECT COUNT(*) FROM samples WHERE valid=1').fetchone()[0]

    def test_normalized_template_duplicates_count_once(self):
        self.add(['identity:a','body:a','template:one'],identity='a')
        self.add(['identity:b','body:b','template:one'],identity='b')
        self.assertEqual(self.count(),1)

    def test_later_spam_label_invalidates_earlier_ham(self):
        self.add(['body:a'])
        self.add(['body:a'],label=1,identity='spam')
        self.assertEqual(self.count(),0)

    def test_earlier_spam_label_prevents_ham_retention(self):
        self.add(['body:a'],label=1)
        self.add(['body:a'],identity='ham')
        self.assertEqual(self.count(),0)

    def test_candidate_bridge_invalidates_both_retained_ham(self):
        self.add(['body:a'],identity='a')
        self.add(['body:b'],identity='b')
        self.add(['body:a','body:b'],identity='bridge')
        self.assertEqual(self.count(),0)

    def test_development_family_blocks_different_identity(self):
        self.pool.block(['family:train'])
        self.add(['body:new'],families=['family:train'])
        self.assertEqual(self.count(),0)

    def test_fp_denominator_is_all_scored_ham(self):
        self.assertEqual(measure({'fp':5,'tn':49995})['hamFpr'],.0001)


if __name__ == '__main__':
    unittest.main()
