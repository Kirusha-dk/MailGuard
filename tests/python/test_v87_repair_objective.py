"""Engineering tensors only; no invented email is a quality measurement."""
import math,sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from v87_repair_objective import threshold_aware_loss,roll_forward_exposures

class ObjectiveTest(unittest.TestCase):
    def test_new_domain_gradient_uses_actual_operating_boundary(self):
        logit=math.log(.04897100105881691/(1-.04897100105881691))
        for label,expected in [(0.,.5),(1.,-.75)]:
            x=torch.tensor([logit],requires_grad=True)
            threshold_aware_loss(x,torch.tensor([label]),torch.tensor([False])).backward()
            self.assertAlmostEqual(x.grad.item(),expected,places=6)
    def test_old_replay_keeps_original_calibration(self):
        x=torch.tensor([-3.,1.]);y=torch.tensor([0.,1.])
        self.assertTrue(torch.equal(threshold_aware_loss(x,y,torch.tensor([True,True])),torch.nn.functional.binary_cross_entropy_with_logits(x,y)))
    def test_weight_rollback_cannot_reset_or_replay_usage(self):
        d=dict(identity='engineering-fixture',before=2,after=3)
        counts=roll_forward_exposures({'engineering-fixture':2},[d])
        self.assertEqual(counts['engineering-fixture'],3)
        with self.assertRaises(ValueError):roll_forward_exposures(counts,[d])
        with self.assertRaises(ValueError):roll_forward_exposures(counts,[dict(d,before=3,after=4)])
        self.assertEqual(roll_forward_exposures({},[dict(identity='new-fixture',before=0,after=1)])['new-fixture'],1)
if __name__=='__main__':unittest.main()
