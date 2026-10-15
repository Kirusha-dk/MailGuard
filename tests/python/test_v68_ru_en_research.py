import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from v68_ru_en_research import (make_row,ingest,partition,sample_fingerprint,
      thresholds,confusion,TEST_PER_LANGUAGE,language_fraction)

class ResearchSplitTests(unittest.TestCase):
    def test_language_gate(self):
        self.assertIsNotNone(make_row('ru',1,'Вам пришло предложение от компании оплатить доставку заказа'))
        self.assertIsNone(make_row('ru',1,'Your shipping document has arrived online tomorrow'))
        self.assertIsNotNone(make_row('en',0,'Hello everyone, please review our invoice below'))
        self.assertIsNone(make_row('en',0,'Привет, проверили счета вашей бухгалтерии по всем поставкам'))
    def test_dedup_does_not_relabel_disagreements(self):
        rows,stats=ingest([
            (1,'Your invoice is overdue please send it to the accounting department'),
            (0,'Your invoice is overdue please send it to the accounting department')
        ],'en')
        self.assertFalse(rows)
        self.assertEqual(stats['conflicting_labels'],1)
    def test_group_split_invariant(self):
        r=make_row('en',1,'Please verify your credentials at the secure portal immediately after login')
        same=make_row('en',0,'Please verify your credentials at the secure portal immediately after login')
        self.assertEqual(r['family'],same['family'])
        self.assertEqual(r['split'],same['split'])
    def test_calibration_uses_ham_only(self):
        y=[0]*350+[1,1]
        scores=[i/100. for i in range(350)]+[100.0,-100.0]
        cutoff=thresholds(scores,y)
        altered=thresholds([*scores[:350],-500.0,800.0],y)
        self.assertEqual(cutoff,altered)
        self.assertEqual(confusion(scores,y,cutoff['conservative'])['fp'],0)
        self.assertEqual(confusion(scores,y,cutoff['conservative'])['tp'],1)
    def test_fp_denominator_and_bounds(self):
        m=confusion([1.,-1.,1.,-1.],[1,1,0,0],0.)
        self.assertEqual((m['tp'],m['fn'],m['fp'],m['tn']),(1,1,1,1))
        self.assertEqual(m['fpr'],.5)
        self.assertGreater(m['fpr95upper'],m['fpr'])
        self.assertEqual(TEST_PER_LANGUAGE,5000)
    def test_fingerprint_does_not_expose_labels(self):
        example=make_row('ru',0,'Здравствуйте, в команду пришел очень хороший ответ о работе')
        from v68_ru_en_research import sample_fingerprint
        p={'ru':{'train':[example],'cal':[],'test':[]}}
        raw=sample_fingerprint(p)
        self.assertNotIn('y',raw['ru']['train'])
        self.assertEqual(raw['ru']['train']['n'],1)

if __name__=='__main__':unittest.main()
