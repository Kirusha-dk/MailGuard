from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from audit_v82_errors import metric
from evaluate_v82_firstuse30k import old_keys


class FirstUseAuditTests(unittest.TestCase):
    def test_template_and_family_aliases_catch_changed_tracking_and_numbers(self):
        def message(number,url):
            return ('From: sender@example.org\nTo: receiver@example.net\nSubject: Account notice\n'
                    'Content-Type: text/plain; charset=utf-8\n\n'
                    'Please review your account details and complete the following important verification '
                    'with our team before continuing to use the service. Reference '+number+' '+url).encode()
        a=old_keys(message('123','https://one.example/a'))
        b=old_keys(message('456','https://two.example/b'))
        self.assertTrue(a.intersection(b))
        self.assertTrue(any(k.startswith('raw:') for k in a))

    def test_spam_only_metric_does_not_invent_false_positive_rate(self):
        result=metric([{'y':1},{'y':1}],[.9,.01],.04)
        self.assertEqual((result['tp'],result['fn']),(1,1))
        self.assertEqual(result['recall'],.5)
        self.assertEqual(result['ham'],0)
        self.assertIsNone(result['fpr'])


if __name__=='__main__':unittest.main()
