import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from v58_runtime import residual_threshold


class ResidualCalibrationTests(unittest.TestCase):
    def test_inherited_ham_does_not_raise_residual_boundary(self):
        labels=np.array([0,0,1]);base=np.array([True,False,False]);scores=np.array([.999,.8,.95])
        t=residual_threshold(labels,base,scores,.7)
        self.assertGreater(t,.8)
        self.assertLess(t,.95)
        self.assertEqual(int(((labels==0)&~base&(scores>=t)).sum()),0)

    def test_new_threshold_never_lowers_frozen_boundary(self):
        self.assertEqual(residual_threshold([0,1],[False,False],[.4,.9],.8),.8)

    def test_float32_zero_additional_fp(self):
        scores=np.array([.987662971,.995],dtype=np.float32)
        t=residual_threshold([0,1],[False,False],scores,.9)
        self.assertFalse(scores[0]>=t)
        self.assertTrue(scores[1]>=t)

    def test_no_residual_ham_disables_new_votes(self):
        self.assertEqual(residual_threshold([0,1],[True,False],[.1,.9],.5),2.)


if __name__=='__main__':unittest.main()
