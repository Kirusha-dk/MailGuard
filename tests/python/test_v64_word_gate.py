import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from v64_runtime import parts,word_features,DIMENSIONS


def row(subject='meeting',head='please confirm tomorrow',tail='thanks'):
    return dict(focus=f'subject {subject} hosts example.invalid start {head}',
                context=f'subject {subject} body {head} tail {tail}')


class WordExpertIsolation(unittest.TestCase):
    def test_subject_with_marker_words_is_parsed_without_truncation(self):
        r=row(subject='body hosts start meeting')
        self.assertEqual(parts(r)[0],'body hosts start meeting')
        self.assertEqual(parts(r)[1],'please confirm tomorrow')

    def test_labels_and_source_have_no_effect_on_any_expert(self):
        for branch in DIMENSIONS:
            a=word_features([dict(row(),y=0,source='ham')],branch)
            b=word_features([dict(row(),y=1,source='spam')],branch)
            self.assertEqual((a!=b).nnz,0)
            self.assertEqual(a.shape,(1,DIMENSIONS[branch]))

    def test_subject_evidence_is_unchanged_by_quoted_body(self):
        a=word_features([row(head='meeting agenda')],'subject')
        b=word_features([row(head='buy cheap pills')],'subject')
        self.assertEqual((a!=b).nnz,0)

    def test_batching_is_exact(self):
        records=[row(),row(subject='会议',head='请确认会议时间',tail='谢谢')]
        for branch in DIMENSIONS:
            batch=word_features(records,branch)
            for i,r in enumerate(records):
                self.assertEqual((batch[i]!=word_features([r],branch)).nnz,0)


if __name__=='__main__':unittest.main()
