import io
import json
import sys
import tarfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import v78_data as data
import v78_mail as runtime
import train_v78_en as train
from audit_v78_source_labels import crosscheck


class V78Tests(unittest.TestCase):
    def test_genspam_features_exclude_publisher_metadata(self):
        payload = '''<MESSAGE><DATE>leaky_date</DATE><FROM>leaky_sender.ac.uk</FROM>
<SUBJECT><TEXT_NORMAL>Project meeting tomorrow</TEXT_NORMAL></SUBJECT>
<MESSAGE_BODY><TEXT_NORMAL>^ Hello &NAME, please review the engineering report at &WEBSITE before our meeting.</TEXT_NORMAL></MESSAGE_BODY></MESSAGE>'''
        blob = io.BytesIO()
        with tarfile.open(fileobj=blob, mode='w:gz') as archive:
            raw = payload.encode(); member = tarfile.TarInfo('train_GEN.ems'); member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
        rows, counts = data.genspam_rows(blob.getvalue())
        self.assertEqual(len(rows), 1)
        text = rows[0]['text']
        self.assertNotIn('leaky_sender', text)
        self.assertNotIn('leaky_date', text)
        self.assertNotIn('train_gen', text)
        self.assertIn('engineering report', text)
        self.assertIn('urltoken', text)
        self.assertEqual(rows[0]['y'], 0)

    def test_both_sides_of_conflicting_labels_are_excluded(self):
        text = 'Please read our detailed project status report before the review meeting.'
        rows = [data.make_row(text, y, 'source') for y in (0, 1)]
        kept, audit = data.unique(rows)
        self.assertEqual(kept, [])
        self.assertEqual(audit['conflicting_identities_excluded'], 1)

    def test_spam_verdict_markers_are_removed(self):
        self.assertEqual(data.normalize('[SPAM] Please read the important announcement.'),
                         data.normalize('Please read the important announcement.'))

    def test_punctuation_conversions_do_not_create_new_identity(self):
        self.assertEqual(data.identifiers("we don't need new equipment"),
                         data.identifiers("we don ' t need new equipment"))

    def test_english_classifier_abstains_on_russian(self):
        self.assertEqual(runtime.predict({'version': runtime.VERSION},
            ['Добрый день, пожалуйста, подтвердите время завтрашней встречи.']), [None])

    def test_threshold_is_conservative_for_ties_and_rejects_nan(self):
        values = np.array([0.] * 998 + [1., 1.] + [2.] * 100)
        truth = [0] * 1000 + [1] * 100
        threshold = train.cutoff(values, truth, .001)
        self.assertGreater(threshold, 1.)
        with self.assertRaises(ValueError):
            train.cutoff([float('nan')] * 1000, [0] * 1000, .001)

    def test_unknown_source_is_rejected(self):
        with self.assertRaises(ValueError):
            data.source_blob('unlisted-file')

    def test_streaming_cache_and_audit_preserve_labels(self):
        original = [{'identity': 'same', 'family': 'group', 'lang': 'en', 'source': 'source', 'y': 0}]
        external = [{'identity': 'same', 'family': 'group', 'y': 1}]
        report = crosscheck(data.iter_array(io.StringIO(json.dumps(original)), block_size=7), external)
        self.assertEqual(report['exact_content_label_crosscheck']['original_0_external_1'], 1)
        self.assertEqual(original[0]['y'], 0)
        self.assertEqual(external[0]['y'], 1)


if __name__ == '__main__':
    unittest.main()
