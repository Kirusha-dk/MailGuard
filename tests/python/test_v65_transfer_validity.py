import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from audit_v65_transfer_validity import audit,wilson_upper,min_zero_fp_ham


def metrics(tp,fn,fp,tn):
    n=tp+fn+fp+tn
    return dict(n=n,spam=tp+fn,ham=fp+tn,tp=tp,fn=fn,fp=fp,tn=tn,
                recall=tp/(tp+fn) if tp+fn else None,
                fpr=fp/(fp+tn) if fp+tn else None)


class TransferValidityTests(unittest.TestCase):
    def fixture(self):
        # Two previously scored sources, one used for source-specific adaptation.
        a=metrics(8,2,0,5)
        b=metrics(7,3,1,4)
        whole=metrics(15,5,1,9)
        d={'v65-conservative-rescue':whole,
           'v64-conservative':whole,'v64-high-recall':whole}
        v65=dict(version=65,modelLockedBeforeHoldout=True,
                 testLabelsUsedForTraining=False,holdoutCount=30,
                 splitCounts={"('train', 'GenSpam', 1)":2,"('cal', 'GenSpam', 0)":2},
                 results={'all-holdout':d,
                   'source/GenSpam':{k:a for k in d},
                   'source/Nazario-2025':{k:b for k in d}})
        v64=dict(version=64,audit={'bySource':{'GenSpam':100,'Nazario-2025':10}})
        previous=dict(version=60,counts={'train':{'TREC-06:0':33,'Ling:1':4},
                                          'cal':{'Enron:0':2},'test':{}})
        return v65,v64,previous

    def test_zero_fp_requires_nearly_30000_ham(self):
        self.assertTrue(29900<min_zero_fp_ham()<30100)
        self.assertGreater(wilson_upper(0,2119),1e-4)
        self.assertGreater(wilson_upper(2,2119),1e-4)

    def test_previously_evaluated_not_first_use(self):
        result=audit(*self.fixture())
        self.assertEqual(result['firstUseSources'],[])
        self.assertFalse(result['fpRequirementDemonstrated'])
        self.assertEqual(result['bySource']['GenSpam']['status'],'trained_or_calibrated_by_v65')
        self.assertEqual(result['bySource']['Nazario-2025']['status'],'previously_scored_source')
        self.assertIn('Ling',result['oldDevelopmentSources'])
        self.assertIn('TREC-06',result['oldDevelopmentSources'])

    def test_unknown_source_is_first_use(self):
        v65,v64,previous=self.fixture()
        v64['audit']['bySource'].pop('Nazario-2025')
        result=audit(v65,v64,previous)
        self.assertEqual(result['firstUseSources'],['Nazario-2025'])

    def test_mismatch_rejected(self):
        v65,v64,previous=self.fixture()
        v65['results']['source/GenSpam']['v65-conservative-rescue']['tp']=99
        with self.assertRaisesRegex(ValueError,'Incorrect spam count'):
            audit(v65,v64,previous)

    def test_no_fp_without_ham_not_evidence(self):
        self.assertIsNone(wilson_upper(0,0))
        with self.assertRaises(ValueError):wilson_upper(3,2)


if __name__=='__main__':unittest.main()
