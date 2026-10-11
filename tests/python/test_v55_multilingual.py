import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from train_v55_multilingual import views, campaign_key, archive_split, features, transformed, fp_budget, clean, FEATURES


def mail(body, subject='Notice', sender='one.example'):
    return f'SUBJECT: {subject}\nURL_HOSTS: \nEMAIL_DOMAINS: \nSENDER_DOMAIN: {sender}\nBODY: {body}'


class MultilingualTests(unittest.TestCase):
    def test_inline_cjk_url_is_visible(self):
        focus, context = views(mail('ご登録くださいませhttps://offers.example/id=123'))
        self.assertIn('offers.example', focus)
        self.assertNotIn('id=123', context)

    def test_cjk_obfuscation_family_stays_in_one_split(self):
        a = mail('無料登録が可能になりました。今すぐ確認ください。 https://one.example/x=111')
        b = mail('無~料.登_録+が/可:能・に*な.り~ま_し+た。今/す~ぐ.確_認+く/だ:さ・い。 https://two.example/x=999', 'Other')
        self.assertEqual(campaign_key(a), campaign_key(b))
        self.assertEqual(archive_split(campaign_key(a)), archive_split(campaign_key(b)))
        self.assertIn('無料登録', views(b)[0])

    def test_envelope_and_addresses_are_not_source_shortcuts(self):
        a = views(mail('Please contact bruce@untroubled.org for the account details.', sender='source.example'))
        b = views(mail('Please contact another@other.example for the account details.', sender='different.example'))
        self.assertEqual(a, b)
        self.assertNotIn('untroubled', a[0])

    def test_subject_is_in_focus_before_long_padding(self):
        focus, context = views(mail('Claim your reward. ' + 'ordinary reference text ' * 10000, 'Reward expires'))
        self.assertIn('reward expires', focus)
        self.assertIn('claim your reward', focus)
        self.assertLess(len(focus), 2000)
        self.assertLess(len(context), 4000)

    def test_css_comments_do_not_dominate_text(self):
        cleaned = clean('Offer /* reset CSS */ .ExternalClass { width: 100%; } starts today')
        self.assertNotIn('width', cleaned)
        self.assertNotIn('reset', cleaned)
        self.assertIn('starts today', cleaned)
        self.assertIn('offer', cleaned)

    def test_fp_budget_uses_ham_denominator(self):
        y = np.asarray([0] * 5000 + [1] * 495000)
        self.assertEqual(fp_budget(y, .0002), 1)

    def test_sparse_transform_handles_single_row_without_dense_expansion(self):
        focus, context = views(mail('This is an ordinary English meeting invitation.'))
        x = features([{'focus':focus,'context':context}])
        self.assertEqual(x.shape, (1, FEATURES))
        self.assertGreater(x.nnz, 0)
        out = transformed(x, np.ones(FEATURES, dtype=np.float32))
        self.assertAlmostEqual(float(out.multiply(out).sum()), 1., places=5)


if __name__ == '__main__':
    unittest.main()
