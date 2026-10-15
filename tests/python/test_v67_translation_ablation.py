import json,sys,unittest
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from v67_translation_ablation import stable_subset,OR,tally,metrics,TRANSLATOR_REVISION,MAX_TRANSLATE_MESSAGES

class V67Translation(unittest.TestCase):
    def fixture(self):
        return [
            dict(identity=f'{i:064x}',source='Turkish-stemmed',label=i%2)
            for i in range(20)
        ]+[dict(identity='f'*63+'0',source='CW-L-phishing',label=1)]
    def test_label_blind_identity_selection(self):
        before=self.fixture()
        a=[r['identity'] for r in stable_subset(before,8)]
        for r in before:r['label']=1-r['label']
        b=[r['identity'] for r in stable_subset(before,8)]
        self.assertEqual(a,b)
        self.assertEqual(len(a),8)
        self.assertTrue(all(i!='f'*63+'0' for i in a))
    def test_or_monotone_and_explicit_fp(self):
        old=(True,False,False,False)
        eng=(False,True,True,False)
        self.assertEqual(OR(old,eng),(True,True,True,False))
        results=Counter()
        for label,pred in zip((1,1,0,0),OR(old,eng)):
            tally(results,label,pred)
        m=metrics(results)
        self.assertEqual((m['tp'],m['fn'],m['fp'],m['tn']),(2,0,1,1))
    def test_no_ham_is_not_zero_fpr(self):
        c=Counter()
        tally(c,1,True)
        m=metrics(c)
        self.assertIsNone(m['fpr'])
        self.assertIsNone(m['fprWilson95Upper'])
    def test_pinned_translator_and_bounded_cohort(self):
        self.assertEqual(len(TRANSLATOR_REVISION),40)
        self.assertEqual(MAX_TRANSLATE_MESSAGES,384)
if __name__=='__main__':unittest.main()
