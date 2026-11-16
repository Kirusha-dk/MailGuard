"""Engineering fixtures only; these are never model-training emails."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from train_v85_hard import hard_order
class HardMiningTests(unittest.TestCase):
 def test_errors_receive_priority_with_correct_support_and_cap(self):
  # FP HAM at index0 and FN SPAM at index1; remaining rows classified correctly.
  y=[0,1]+[0,1]*30;s=np.array([.9,.001]+[.001,.9]*30);ids=[str(i) for i in range(len(y))]
  order,wrong=hard_order(y,s,ids,10)
  self.assertEqual(list(np.flatnonzero(wrong)),[0,1]);self.assertEqual(order[:6],[order[0],order[1]]*3)
  counts=np.bincount(order,minlength=len(y));self.assertEqual(counts[0],3);self.assertEqual(counts[1],3);self.assertTrue((counts[2:]<=1).all());self.assertTrue((counts[2:]>0).any());self.assertEqual(order,hard_order(y,s,ids,10)[0])
 def test_target_does_not_repeat_correct_rows(self):
  ids=[str(i) for i in range(100)];order,wrong=hard_order([0]*100,np.zeros(100),ids,2)
  self.assertEqual(len(order),32);self.assertEqual(len(set(order)),32);self.assertFalse(wrong.any())
if __name__=='__main__':unittest.main()
