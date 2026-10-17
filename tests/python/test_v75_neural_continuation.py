import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools"))
from v75_neural_continuation import (
    deterministic,split_data,calibrate,budget,standardizer,blend,
    select_alpha,ALPHAS,PROFILES,TRAIN_PER_CLASS,SIZES,V74_SHA,V69_SHA)

class V75QualityChecks(unittest.TestCase):
    def test_label_blind_sample_deterministic(self):
        rows=[dict(identity=f"{i:064x}",family=f"{i:064x}",y=i%2) for i in range(30)]
        a=[r["identity"] for r in deterministic(rows,10,"sample")]
        for r in rows:r["y"]=1-r["y"]
        b=[r["identity"] for r in deterministic(list(reversed(rows)),10,"sample")]
        self.assertEqual(a,b)
    def test_calibration_never_uses_spam_for_cutoff(self):
        h=np.arange(1000)/1000.
        s=np.arange(300)/200.
        y=np.r_[[0]*len(h),[1]*len(s)]
        a=calibrate(np.r_[h,s],y,0)
        b=calibrate(np.r_[h,s+10000],y,0)
        self.assertEqual(a["cutoff"],b["cutoff"])
        self.assertEqual(a["fp"],0)
        c=calibrate(np.r_[h,s],y,1)
        self.assertLessEqual(c["fp"],1)
    def test_select_only_threshold_guards(self):
        rng=np.random.default_rng(75)
        truth=[0]*500+[1]*250
        base=np.r_[rng.normal(-1,1,500),rng.normal(1,1,250)]
        neural=np.r_[rng.normal(-1,1,500),rng.normal(0.3,1,250)]
        scales=dict(base=standardizer(base,truth),neural=standardizer(neural,truth))
        for policy in PROFILES:
            chosen=select_alpha(base,neural,truth,scales,policy)
            self.assertIn(chosen["alpha"],ALPHAS)
            self.assertLessEqual(chosen["selectMetrics"]["fp"],budget(truth,policy))
            val=blend(base,neural,scales,chosen["alpha"])
            self.assertEqual(len(val),len(truth))
    def test_low_sample_or_nan_rejected(self):
        with self.assertRaises(ValueError):
            calibrate([float("nan")]*400,[0]*250+[1]*150,0)
    def test_size_and_hash_guards(self):
        self.assertEqual(sum(2*n for n in TRAIN_PER_CLASS.values()),10000)
        self.assertEqual(len(V74_SHA),64)
        self.assertEqual(len(V69_SHA),64)
        self.assertEqual(len(SIZES),2)
if __name__=="__main__":unittest.main()
