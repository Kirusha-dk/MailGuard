import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('evaluation', Path(__file__).resolve().parents[2]/'tools/evaluate_v62_500k.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class HonestEvaluation(unittest.TestCase):
    def test_family_relative_is_excluded_even_with_different_identity(self):
        ids = {'train':{'a'}, 'select':set(), 'cal':set()}
        families = {'train':{'campaign'}, 'select':set(), 'cal':set()}
        self.assertEqual(evaluation.overlap('new-id', 'campaign', ids, families), ['train'])
        self.assertEqual(evaluation.overlap('new-id', 'new-campaign', ids, families), [])

    def test_selection_and_calibration_also_exclude_rows(self):
        ids = {'train':set(), 'select':{'a'}, 'cal':set()}
        families = {'train':set(), 'select':set(), 'cal':{'campaign'}}
        self.assertEqual(evaluation.overlap('a', 'campaign', ids, families), ['select', 'cal'])

    def test_fp_denominator_uses_legitimate_mail(self):
        m = evaluation.measure({'tp':99900, 'fn':100, 'tn':99, 'fp':1})
        self.assertEqual(m['hamFpr'], .01)
        self.assertEqual(m['recall'], .999)

    def test_zero_fp_on_small_ham_set_has_nonzero_uncertainty(self):
        self.assertGreater(evaluation.wilson_upper(0, 775), .004)
        self.assertIsNone(evaluation.measure({'tp':122})['hamFpr'])


if __name__ == '__main__':
    unittest.main()
