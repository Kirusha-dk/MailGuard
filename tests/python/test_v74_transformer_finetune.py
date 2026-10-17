import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools"))
from v74_transformer_finetune import (select,freeze_splits,cal_cutoffs,
                                    unlock_trainable,TRAIN_PER_CLASS,PROFILES)

class TransformerResearchTests(unittest.TestCase):
    def test_selection_blind_to_test_outcomes(self):
        rows=[dict(identity=f"{i:064x}",family=f"{i:064x}",y=i%2,text="example") for i in range(50)]
        a=[r["identity"] for r in select(rows,20,"cal")]
        for row in rows:row["y"]=1-row["y"]
        b=[r["identity"] for r in select(list(reversed(rows)),20,"cal")]
        self.assertEqual(a,b)
        self.assertEqual(len(a),20)
    def test_train_balanced_strata(self):
        rows=[dict(identity=f"{i:064x}",y=i%2) for i in range(30)]
        taken=select(rows,5,"train",balanced=True)
        self.assertEqual(sum(r["y"] for r in taken),5)
        self.assertEqual(len(taken),10)
    def test_conservative_fpr_bound(self):
        spam=np.linspace(-1.0,2.0,500)
        ham=np.linspace(-3.0,1.0,1000)
        scores=np.r_[ham,spam]
        y=np.array([0]*len(ham)+[1]*len(spam))
        settings=cal_cutoffs(scores,y)
        self.assertEqual(settings["zero-cal-fp"]["calFP"],0)
        self.assertLessEqual(settings["research-0p1pct-cal-fpr"]["calFP"],1)
        # No TEST outcomes enter calibration cutoffs.
        flipped=y.copy();flipped[len(ham):]=1
        self.assertEqual(settings,cal_cutoffs(scores,flipped))
    def test_bad_cal_class_distribution_rejected(self):
        with self.assertRaises(ValueError):
            cal_cutoffs([0.1]*600,[0]*600)
    def test_trainable_transformer_layers(self):
        import torch
        from types import SimpleNamespace
        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bert=torch.nn.Module()
                self.bert.encoder=torch.nn.Module()
                self.bert.encoder.layer=torch.nn.ModuleList([
                    torch.nn.Linear(768,768) for _ in range(12)])
                self.bert.pooler=torch.nn.Linear(768,768)
                self.classifier=torch.nn.Linear(768,2)
        model=Fake();count=unlock_trainable(model)
        self.assertGreater(count,1_000_000)
        self.assertFalse(any(p.requires_grad for p in model.bert.encoder.layer[8].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.bert.encoder.layer[11].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))

if __name__=="__main__":unittest.main()
