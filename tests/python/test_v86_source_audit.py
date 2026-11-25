"""Engineering fixtures verify data exclusion; they are never training emails."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from v84_replay_data import fingerprint, read_rows, write_rows
from v86_exclude_prior import exclude


def row(identity, family, text, label=1):
    return dict(identity=identity, family=family, text=text, y=label, source='fixture')


class SourceExclusionTest(unittest.TestCase):
    def test_quote_match_excludes_whole_candidate_family(self):
        phrase = ' '.join('word' + chr(97+i) for i in range(25))
        old = [row('old', 'old-family', phrase, 0)]
        candidates = [row('quoted', 'group', 'introduction ' + phrase),
                      row('same-family', 'group', 'different body'),
                      row('new', 'new-family', 'unrelated native email fixture')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); write_rows(root/'old.jsonl.gz', old)
            write_rows(root/'input.jsonl.gz', candidates)
            report = exclude([root/'input.jsonl.gz'],
                [('prior', root, {'old':fingerprint(old)})], root/'output')
            self.assertEqual(report['novel_candidate_rows'], 1)
            self.assertEqual([r['identity'] for r in read_rows(root/'output/candidates.jsonl.gz')], ['new'])
            self.assertFalse(report['training_ready'])
            self.assertEqual(report['optimizer_steps'], 0)

    def test_conflicting_labels_never_become_training_candidates(self):
        candidates = [row('a', 'same', 'first text', 0),
                      row('b', 'same', 'second text', 1)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); write_rows(root/'input.jsonl.gz', candidates)
            report = exclude([root/'input.jsonl.gz'], [], root/'output')
            self.assertEqual(report['novel_candidate_rows'], 0)

    def test_prior_fingerprint_mismatch_fails_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rows(root/'input.jsonl.gz', [row('a', 'b', 'new email fixture')])
            write_rows(root/'old.jsonl.gz', [row('c', 'd', 'old email fixture')])
            with self.assertRaisesRegex(ValueError, 'corpus drift'):
                exclude([root/'input.jsonl.gz'], [('prior',root,{'old':'wrong'})], root/'output')
            self.assertFalse((root/'output/candidates.jsonl.gz').exists())


if __name__ == '__main__':
    unittest.main()
