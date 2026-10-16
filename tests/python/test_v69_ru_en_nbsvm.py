import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools"))
from v69_ru_en_nbsvm import (vectorizer,nb_logratio,combine,choose_calibration,
                            cal_allowance,accumulate,EXPECTED_TEST_IDS,PROFILES)
from v68_ru_en_research import confusion
class V69ResearchChecks(unittest.TestCase):
    def test_nb_ratio(self):
        X=vectorizer("word").transform(["free winner prize","free winner","meeting project","meeting now"])
        r=nb_logratio(X,np.array([1,1,0,0]))
        self.assertTrue(np.isfinite(r).all())
        self.assertGreater(r.max(),0)
        self.assertLess(r.min(),0)
    def test_calibration_uses_only_cal(self):
        y=np.array([0]*1000+[1]*200)
        rng=np.random.default_rng(69)
        a=np.r_[np.linspace(-2,2,1000),rng.normal(1,2,200)]
        b=np.r_[np.linspace(-1,3,1000),rng.normal(2,1,200)]
        signals=combine(a,b)
        pick=choose_calibration(signals,y,0)
        self.assertEqual(pick["calibrationFP"],0)
        self.assertIn(pick["candidate"],signals)
        self.assertEqual(int(np.sum(signals[pick["candidate"]][:1000]>=pick["cutoff"])),0)
    def test_cal_fp_budget(self):
        self.assertEqual(cal_allowance(1200,PROFILES[0]),0)
        self.assertEqual(cal_allowance(1200,PROFILES[1]),1)
        with self.assertRaises(ValueError):cal_allowance(1000,"unknown")
    def test_accounting(self):
        a=confusion([2,0,-2,1],[1,1,0,0],.5)
        b=confusion([-1,5,-1,4],[0,1,0,1],0)
        final=accumulate((a,b))
        self.assertEqual((final["tp"],final["fn"],final["fp"],final["tn"]),(3,1,1,3))
    def test_retrospective_identifiers_pinned(self):
        self.assertEqual(len(EXPECTED_TEST_IDS),2)
        self.assertTrue(all(len(x)==64 for x in EXPECTED_TEST_IDS.values()))
if __name__=="__main__":unittest.main()
