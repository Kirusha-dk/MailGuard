import sys
import unittest
from email.message import EmailMessage
from pathlib import Path
import numpy as np
import io
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import v77_mail as runtime
import train_v77_ru_en as train
from evaluate_v77_external import iter_json_array


class V77Tests(unittest.TestCase):
    def test_streaming_overlap_index_crosses_buffer_boundaries(self):
        rows = [{'text': 'тест, скобки { } [ ] ' * 50, 'identity': 'a'}, {'identity': 'b'}]
        stream = io.StringIO(json.dumps(rows, ensure_ascii=False))
        self.assertEqual(list(iter_json_array(stream, block_size=17)), rows)

    def test_mime_cp1251_and_html_alternative(self):
        mail = EmailMessage()
        mail['Subject'] = 'Согласование встречи'
        mail.set_content('Добрый день, пожалуйста, подтвердите время нашей встречи завтра.', charset='windows-1251')
        mail.add_alternative('<p>Добрый день, встреча завтра.</p>', subtype='html')
        text = runtime.parse_eml(mail.as_bytes())
        self.assertIn('подтвердите время', text)
        self.assertNotIn('<p>', text)
        self.assertEqual(runtime.language(text), 'ru')

    def test_verdict_headers_are_not_features(self):
        body = 'X-Spam-Status: Yes, score=99\nFrom: sender@example.com\n\nPlease find the requested report attached.'
        text = runtime.clean('Monthly report', body)
        self.assertNotIn('x-spam-status', text)
        self.assertNotIn('score', text)
        self.assertIn('requested report', text)

    def test_undeclared_russian_legacy_charsets(self):
        text = 'Добрый день, пожалуйста, подтвердите время нашей встречи завтра.'
        for encoding in ('cp1251', 'koi8-r'):
            raw = b'Content-Type: text/plain\n\n' + text.encode(encoding)
            self.assertIn('подтвердите время', runtime.parse_eml(raw))

    def test_obfuscation_html_and_tail(self):
        text = runtime.clean('', '<style>hidden</style><p>пр\u200bивет &amp; новости</p>')
        self.assertEqual(text, 'привет & новости')
        long = 'alpha ' * 3000 + 'importanttail'
        for kind in ('word', 'char'):
            self.assertTrue(runtime.content_view(long, kind).endswith('importanttail'))

    def test_unsupported_language_abstains(self):
        model = {'version': runtime.VERSION}
        self.assertIsNone(runtime.predict(model, ['你好，这是一个重要的信息内容，请阅读这个消息。'])[0])

    def test_threshold_ties_respect_fp_cap(self):
        labels = [0] * 1000 + [1] * 100
        values = np.array([.2] * 998 + [.8] * 2 + [1.] * 100)
        cfg = train.choose({'word': values}, labels, .001)
        self.assertEqual(cfg['calibration']['fp'], 0)
        self.assertEqual(cfg['calibration']['tp'], 100)
        self.assertGreater(cfg['cutoff'], .8)

    def test_confusion_has_correct_denominators(self):
        m = runtime.measure([1, 1, -1, -1], [1, 0, 1, 0], 0)
        self.assertEqual((m['tp'], m['fp'], m['fn'], m['tn']), (1, 1, 1, 1))
        self.assertEqual(m['recall'], .5)
        self.assertIsNone(runtime.measure([1], [1], 0)['fpr'])

    def test_group_holdout_and_conflicting_labels(self):
        rows = []
        # Explicit identities isolate partition logic from natural-language parsing.
        def make(identity, family, split, lang, y):
            return dict(identity=identity, family=family, official_split=split,
                        lang=lang, y=y, text='placeholder', source='test')
        for lang in ('en', 'ru'):
            for i in range(200):
                key = f'{lang}-development-{i}'
                rows.append(make(key, key, 'train' if lang == 'en' else 'proxy', lang, i % 2))
        for i in range(12):
            rows.append(make(f'test-{i}', f'heldout-family-{i}', 'test', 'en', i % 2))
        rows += [make('leaking-copy', 'heldout-family-0', 'train', 'en', 0),
                 make('contradiction', 'unrelated', 'train', 'en', 0),
                 make('contradiction', 'unrelated', 'train', 'en', 1)]
        fitting, calibration, test, conflicts = train.partition(rows, test_count=10)
        self.assertEqual(len(test), 10)
        self.assertEqual(conflicts, 1)
        self.assertNotIn('leaking-copy', {r['identity'] for r in fitting['en'] + calibration['en']})
        self.assertNotIn('contradiction', {r['identity'] for r in fitting['en'] + calibration['en']})
        self.assertFalse({r['family'] for r in fitting['en']} & {r['family'] for r in calibration['en']})


if __name__ == '__main__':
    unittest.main()
