import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import v83_data as data


class V83DataTests(unittest.TestCase):
    def raw(self, text=None, y=0, source='public-email'):
        return dict(subject='', body=text or 'Добрый день пожалуйста подтвердите время нашей общей встречи завтра после обеда коллеги',
                    y=y, source=source, kind='email', language='ru')

    def lid(self, text):
        return 'ru', .99

    def test_unicode_is_preserved_and_label_and_source_kind_are_required(self):
        row = data.make_row('', 'Mañana tendremos una reunión importante con nuestros compañeros para revisar todos los documentos.',
                            0, 'public-email', 'es')
        self.assertIn('mañana', row['text'])
        for label in (True, 'ham', None, 2):
            with self.assertRaises(ValueError):
                data.make_row('', self.raw()['body'], label, 'public-email', 'ru')
        for raw in (dict(self.raw(), kind='telegram'), {k: v for k, v in self.raw().items() if k != 'kind'},
                    dict(self.raw(), source_kind='translated-email')):
            with self.assertRaises(ValueError):
                data.prepare_rows([raw], 'ru', self.lid)
        self.assertIsNone(data.make_row('', 'short', 0, 'mail', 'ru'))

    def test_conflicting_identity_is_permanently_excluded(self):
        rows = [self.raw(y=0), self.raw(y=1), self.raw(y=0)]
        parts, stats = data.prepare_rows(rows, 'ru', self.lid)
        self.assertEqual(sum(map(len, parts.values())), 0)
        self.assertEqual(stats['conflicting_identities_excluded'], 1)

    def test_duplicate_resolution_and_fingerprints_are_order_independent(self):
        rows = [self.raw(source='z-provider'), self.raw(source='a-provider')]
        first, stats = data.prepare_rows(rows, 'ru', self.lid)
        second, _ = data.prepare_rows(reversed(rows), 'ru', self.lid)
        self.assertEqual(first, second)
        self.assertEqual(stats['duplicates'], 1)
        chosen = next(row for values in first.values() for row in values)
        self.assertEqual(chosen['source'], 'a-provider')

    def test_family_prefix_cannot_cross_partitions(self):
        prefix = ' '.join(['русское слово'] * 30)
        rows = [self.raw(prefix + ' различный хвост', y=0), self.raw(prefix + ' другой хвост', y=1)]
        parts, _ = data.prepare_rows(rows, 'ru', self.lid)
        populated = [values for values in parts.values() if values]
        self.assertEqual(len(populated), 1)
        self.assertEqual(len(populated[0]), 2)
        self.assertEqual(populated[0][0]['family'], populated[0][1]['family'])

    def test_language_filter_is_outcome_blind_and_enforces_confidence(self):
        for detected, confidence in [('en', .99), ('ru', .79)]:
            parts, stats = data.prepare_rows([self.raw(y=0), self.raw(y=1)], 'ru', lambda text: (detected, confidence))
            self.assertEqual(sum(map(len, parts.values())), 0)
            self.assertEqual(stats['language_excluded'], 2)
        with self.assertRaises(ValueError):
            data.prepare_rows([self.raw()], 'ru', lambda text: ('ru', float('nan')))

    def test_freeze_small_corpus_audits_not_training_and_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a', Path(directory) / 'b'
            manifest = data.freeze([self.raw()], 'ru', self.lid, a, {'url': 'https://example.invalid/public'})
            second = data.freeze([self.raw()], 'ru', self.lid, b, {'url': 'https://example.invalid/public'})
            self.assertFalse(manifest['training_ready'])
            self.assertEqual(manifest, second)
            for name in ('train', 'select', 'test'):
                self.assertEqual((a / (name + '.jsonl.gz')).read_bytes(), (b / (name + '.jsonl.gz')).read_bytes())
                with gzip.open(a / (name + '.jsonl.gz'), 'rt', encoding='utf-8') as stream:
                    rows = [json.loads(line) for line in stream]
                digest = hashlib.sha256()
                for row in rows:
                    canonical = json.dumps({k: row[k] for k in ('identity', 'family', 'source', 'y', 'text')},
                                           ensure_ascii=False, sort_keys=True).encode()
                    digest.update(len(canonical).to_bytes(8, 'big')); digest.update(canonical)
                self.assertEqual(manifest['fingerprints'][name], digest.hexdigest())
            with self.assertRaises(FileExistsError):
                data.freeze([self.raw()], 'ru', self.lid, a, {})


if __name__ == '__main__':
    unittest.main()
