import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from v59_mail import from_eml, decode_payload
from prepare_v59_new_mail import family_split, Collector
from evaluate_v54_unique500k import Pool
from train_v47_phishing_aware import canonical_fields
from train_v55_multilingual import campaign_key
import tempfile


class NewMailRegressionTests(unittest.TestCase):
    def test_chinese_without_declared_charset(self):
        text = '您好，明天下午开会，请确认时间和地点。'
        raw = b'From: sender@example.com\nSubject: Meeting\n\n' + text.encode('gb2312')
        self.assertIn(text.replace('，', ','), from_eml(raw)['ngram_text'])
        self.assertNotIn('\ufffd', from_eml(raw)['ngram_text'])

    def test_declared_japanese_charset(self):
        text = '明日の会議について確認をお願いします。'
        raw = b'Content-Type: text/plain; charset=iso-2022-jp\n\n' + text.encode('iso2022_jp')
        self.assertIn(text, from_eml(raw)['ngram_text'])

    def test_unencoded_chinese_subject(self):
        raw = b'Subject: ' + '会议时间'.encode('gb2312') + b'\n\n' + '明天下午三点会议'.encode('gb2312')
        self.assertIn('SUBJECT: 会议时间', from_eml(raw)['ngram_text'])

    def test_missing_japanese_charset(self):
        text = '明日の会議について確認をお願いします。'
        raw = b'Subject: Meeting\n\n' + text.encode('iso2022_jp')
        self.assertIn(text, from_eml(raw)['ngram_text'])

    def test_latin_fallback_and_attachment_exclusion(self):
        self.assertEqual(decode_payload('Please send the café receipt'.encode('cp1252')), 'Please send the café receipt')
        raw = (b'MIME-Version: 1.0\nContent-Type: multipart/mixed; boundary=x\n\n'
               b'--x\nContent-Type: text/plain\n\nMeeting tomorrow.\n'
               b'--x\nContent-Type: text/plain\nContent-Disposition: attachment\n\nBuy pills now\n--x--\n')
        self.assertIn('Meeting tomorrow.', from_eml(raw)['ngram_text'])
        self.assertNotIn('Buy pills now', from_eml(raw)['ngram_text'])

    def test_chronological_test_never_enters_development(self):
        for n in range(100):
            key = f'{n:064x}'
            for month in (6, 7, 8, 9):
                self.assertEqual(family_split(key, f'SpamArchive-2026-{month:02d}'), 'test')
            self.assertNotEqual(family_split(key, 'SpamArchive-2026-01'), 'test')
            self.assertEqual(family_split(key, 'W3C-Japanese-weak-ham'), 'train')

    def test_conflicting_templates_cannot_be_retained_as_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = Pool(Path(tmp) / 'blocked.sqlite')
            collector = Collector(pool)
            a = canonical_fields('Example', 'This is a long message https://example.org/日本語 end of message')
            b = canonical_fields('Example', 'This is a long message https://other.example/会议 end of message')
            self.assertNotEqual(campaign_key(a['ngram_text']), campaign_key(b['ngram_text']))
            collector.add(b'', 0, 'TREC06-Chinese', 'a', canonical=a)
            collector.add(b'', 1, 'TREC06-Chinese', 'b', canonical=b)
            self.assertTrue(collector.rows)
            self.assertTrue(all(f in collector.conflicts for f in collector.rows))
            pool.close()


if __name__ == '__main__':
    unittest.main()
