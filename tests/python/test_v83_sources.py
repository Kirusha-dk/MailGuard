import datetime as dt
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from fetch_v83_sources import parse_turkish_csv
from train_v83_language import validate_request
from v83_data import prepare_rows


class TurkishSourceTests(unittest.TestCase):
    def test_original_last_label_survives_unescaped_text_commas_and_footer(self):
        raw=('Text,Classification\r\n'
             'Türkçe bir ileti, yanlış kaçışlı metin,spam\r\n'
             'Toplantımız yarın,ham\r\n'
             ',,496\r\n').encode('utf-8-sig')
        rows,stats=parse_turkish_csv(raw)
        self.assertEqual([r['y'] for r in rows],[1,0])
        self.assertEqual(rows[0]['body'],'Türkçe bir ileti, yanlış kaçışlı metin')
        self.assertEqual(stats,dict(rejoined_extra_fields=1,excluded_footer=1))
        self.assertTrue(all(r['kind']=='email' and r['language']=='tr' for r in rows))

    def test_unknown_label_and_wrong_schema_are_never_guessed(self):
        for raw in (b'Text,Classification\nhello,unknown\n',
                    b'Text,Classification\nhello\n',b'body,label\nhello,ham\n'):
            with self.assertRaises(ValueError):parse_turkish_csv(raw)

    def test_turkish_content_is_kept_only_with_matching_language_detector(self):
        raw=dict(subject='',body='Merhaba arkadaşlar yarın toplantı için önemli belgeleri hep birlikte yeniden inceleyelim lütfen',
                 y=0,source='public-original-mail',kind='email',language='tr')
        parts,_=prepare_rows([raw],'tr',lambda text:('tr',.99))
        self.assertEqual(sum(map(len,parts.values())),1)
        parts,_=prepare_rows([raw],'tr',lambda text:('es',.99))
        self.assertEqual(sum(map(len,parts.values())),0)

    def test_turkish_window_does_not_reuse_spanish_language_or_state_guard(self):
        request=dict(language='tr',campaign='v83-tr-20261104',start_date='2026-11-04',
            end_date='2026-12-03',session_id='2026-11-04',budget_steps=100,expected_min_step=0)
        validate_request(request,'tr',dt.date(2026,11,4))
        with self.assertRaises(ValueError):validate_request(request,'es',dt.date(2026,11,4))
        with self.assertRaises(ValueError):validate_request(request,'tr',dt.date(2026,12,4))


if __name__=='__main__':unittest.main()
