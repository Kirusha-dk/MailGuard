import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from prepare_v64_transfer import UniquePool,view_keys,genspam_messages

def row(name,label):
    return dict(identity=name,family=name,y=label,source='fixture',script='latin',origin=name,
                focus='subject test hosts  start body',context='subject test body body tail body')

class FirstUseProtocol(unittest.TestCase):
    def test_conflicting_labels_invalidate_retained_mail(self):
        with tempfile.TemporaryDirectory() as d:
            pool=UniquePool(Path(d)/'db');pool.add(row('a',0),['body:shared']);pool.add(row('b',1),['body:shared'])
            self.assertEqual(pool.count(),0);pool.close()

    def test_bridge_to_previous_mail_invalidates_candidate_and_variants(self):
        with tempfile.TemporaryDirectory() as d:
            pool=UniquePool(Path(d)/'db',{'old:key'});pool.add(row('a',0),['new:key'])
            pool.add(row('b',0),['new:key','old:key']);pool.add(row('c',0),['new:key','variant:key'])
            self.assertEqual(pool.count(),0);pool.close()

    def test_bridge_between_two_retained_messages_invalidates_both(self):
        with tempfile.TemporaryDirectory() as d:
            pool=UniquePool(Path(d)/'db');pool.add(row('a',1),['key:a']);pool.add(row('b',1),['key:b'])
            pool.add(row('c',1),['key:a','key:b']);self.assertEqual(pool.count(),0);pool.close()

    def test_view_fingerprint_ignores_changed_subject_and_hosts(self):
        a=row('a',0);b=dict(a,focus='subject changed hosts other start body',context='subject changed body body tail body')
        self.assertEqual(view_keys(a),view_keys(b))

    def test_gen_parser_uses_published_text_fields_only(self):
        text='<MESSAGE><FROM>metadata</FROM><SUBJECT><TEXT_NORMAL>Subject &amp; name</TEXT_NORMAL></SUBJECT><MESSAGE_BODY><TEXT_NORMAL>Body one</TEXT_NORMAL><TEXT_NORMAL>Body two</TEXT_NORMAL></MESSAGE_BODY></MESSAGE>'
        self.assertEqual(list(genspam_messages(text)),[('Subject & name','Body one\nBody two')])

if __name__=='__main__':unittest.main()
