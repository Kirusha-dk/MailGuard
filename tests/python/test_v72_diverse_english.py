import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools"))
from v72_diverse_english import (partition_external,training_mix,choose_multi_source,
                                fingerprint,PROFILES,CANDIDATES,TRAIN_EXT_MULTIPLIER,
                                MIN_TRAIN_EXT,MIN_CAL_EXT)

class DiverseSourceGuardTests(unittest.TestCase):
    def test_training_repeats_only_external_train(self):
        a=[dict(identity="a",family="one",text="abc",y=0)]
        b=[dict(identity="b",family="two",text="xyz",y=1)]
        result=training_mix(a,b)
        self.assertEqual(len(result),1+TRAIN_EXT_MULTIPLIER)
        self.assertEqual(result[0]["identity"],"a")
        self.assertEqual([x["identity"] for x in result[1:]],["b"]*TRAIN_EXT_MULTIPLIER)
        with self.assertRaisesRegex(ValueError,"overlap"):training_mix(a,a)

    def test_score_choice_obeys_all_cal_sources(self):
        rng=np.random.default_rng(72)
        # Two different score distributions. Group-specific FPR caps are
        # enforced jointly, not just pooled.
        inputs={}
        for key,n0,n1,shift in (
            ("en-known",11000,2000,0.0),
            ("csdmc",1400,500,.5)
        ):
            labels=np.r_[np.zeros(n0,dtype=np.int8),np.ones(n1,dtype=np.int8)]
            ham=rng.normal(-2+shift,1,n0)
            spam=rng.normal(1+shift,1,n1)
            vals=np.r_[ham,spam]
            inputs[key]=({k:vals*(1+i*.015) for i,k in enumerate(CANDIDATES)},labels)
        for profile in PROFILES:
            choice=choose_multi_source(inputs,profile)
            self.assertIn(choice["candidate"],CANDIDATES)
            for key,(scores,labels) in inputs.items():
                fp=int(np.count_nonzero(scores[choice["candidate"]][labels==0]>=choice["cutoff"]))
                max_fp=0 if profile=="zero-cal-fp" else int(np.count_nonzero(labels==0)*.001)
                self.assertLessEqual(fp,max_fp)

    def test_selection_does_not_need_external_test_data(self):
        base={"en-known":({k:np.r_[np.linspace(-2,2,11000),np.linspace(1,4,500)]
                           for k in CANDIDATES},
                          np.r_[np.zeros(11000,dtype=np.int8),np.ones(500,dtype=np.int8)]),
              "csdmc":({k:np.r_[np.linspace(-3,2,900),np.linspace(1,5,400)]
                        for k in CANDIDATES},
                       np.r_[np.zeros(900,dtype=np.int8),np.ones(400,dtype=np.int8)])}
        first=choose_multi_source(base,"zero-cal-fp")
        second=choose_multi_source(base,"zero-cal-fp")
        self.assertEqual(first,second)
        self.assertEqual(first["sourceCAL"]["csdmc"]["fp"],0)

    def test_partition_family_stable(self):
        # At least 2000 train / 350 cal; 60 distinct group prefixes.
        items=[]
        for i in range(90):
            fam=f"{i:064x}"
            for j in range(48):
                items.append(dict(identity=f"{i:04x}{j:04x}",family=fam,text="text",
                                  y=(i+j)%2))
        p=partition_external(items)
        fam_train={r["family"] for r in p["train"]}
        fam_cal={r["family"] for r in p["cal"]}
        self.assertFalse(fam_train & fam_cal)
        self.assertGreater(len(p["train"]),MIN_TRAIN_EXT)
        self.assertGreater(len(p["cal"]),MIN_CAL_EXT)
        before=fingerprint(p["train"])
        for r in p["train"]:r["y"]=1-r["y"]
        self.assertEqual(before,fingerprint(p["train"]))

if __name__=="__main__":unittest.main()
