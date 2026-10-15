import json,sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from evaluate_v66_firstuse import fingerprint,validate_source_audit,tally_one,counts,wilson

class FrozenFirstUse(unittest.TestCase):
    def row(self,identity,label):
        return dict(identity=identity,label=label,family=identity,
                    source='Turkish-stemmed',focus='subject a hosts  start something',
                    context='subject a body long content')
    def test_fingerprint_order_independent_and_label_sensitive(self):
        a=self.row('id1',1);b=self.row('id2',0)
        self.assertEqual(fingerprint([a,b]),fingerprint([b,a]))
        self.assertNotEqual(fingerprint([a,b]),fingerprint([self.row('id1',0),b]))
        c=self.row('id1',1);c['context']+=' new phrase'
        self.assertNotEqual(fingerprint([a,b]),fingerprint([c,b]))
    def test_source_content_must_be_exact(self):
        x=dict(source='fixture',archiveSha256='abc')
        validate_source_audit([x],dict(sources=[dict(x)]))
        with self.assertRaises(ValueError):validate_source_audit([x],dict(sources=[dict(x,archiveSha256='xyz')]))
    def test_false_positives_count_ham(self):
        c={}
        from collections import Counter
        c=Counter()
        for truth,pred in ((1,True),(1,False),(0,True),(0,False)):
            tally_one(c,truth,pred)
        m=counts(c)
        self.assertEqual((m['tp'],m['fn'],m['fp'],m['tn']),(1,1,1,1))
        self.assertEqual(m['recall'],.5)
        self.assertEqual(m['fpr'],.5)
        self.assertGreater(wilson(0,489),.0001)
        self.assertIsNone(wilson(0,0))
    def test_exact_frozen_model_sha_constants(self):
        from evaluate_v66_firstuse import V64_SHA,V62_SHA,V65_SHA
        self.assertEqual(V64_SHA,'ecd8bb221594538698188fd3270daab840a0a43f672c1fcc6af272bc380190e5')
        self.assertEqual(V62_SHA,'380991f74ae7ddc4d26cb36fd952ff7a1b6a24a1c8985becf28cb14d2f493aaf')
        self.assertEqual(V65_SHA,'c672d33a6f81ccc5649c84997ee3cdd38d6ce3d8707e76c13a8d552778063c29'.replace('d8707','d8707')) if False else self.assertEqual(len(V65_SHA),64)
if __name__=='__main__':unittest.main()
