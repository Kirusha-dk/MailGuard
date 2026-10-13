import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from v63_neighbours import neighbours


class NeighbourStability(unittest.TestCase):
    def test_mixed_label_duplicates_use_whole_boundary_group(self):
        anchors=np.tile(np.array([[1.,0.]],dtype=np.float32),(10,1))
        labels=np.array([0,0,0,1,1,1,1,1,1,1])
        result=neighbours(anchors[:1],anchors,labels)
        self.assertAlmostEqual(float(result[0,3]),.7,places=6)
        permutation=np.arange(10)[::-1]
        np.testing.assert_array_equal(result,neighbours(anchors[:1],anchors[permutation],labels[permutation]))

    def test_caller_batch_size_and_order_preserve_each_decision_feature(self):
        rng=np.random.default_rng(63)
        anchors=rng.normal(size=(48,32)).astype(np.float32)
        anchors/=np.linalg.norm(anchors,axis=1,keepdims=True)
        anchors[10:20]=anchors[0];labels=np.arange(48)%2
        queries=np.concatenate([anchors[:4],anchors[10:20],anchors[40:48]])
        expected=neighbours(queries,anchors,labels,batch_size=1)
        np.testing.assert_array_equal(expected,neighbours(queries,anchors,labels,batch_size=16))
        np.testing.assert_array_equal(expected,neighbours(queries[::-1],anchors,labels,batch_size=7)[::-1])


if __name__=='__main__':unittest.main()
