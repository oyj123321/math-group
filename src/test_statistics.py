import unittest
import numpy as np
from analyze_v2 import crossed_bootstrap
from paired_missingness import bayes_prediction, theoretical_risk
from run import data, perturb
import torch

class Statistics(unittest.TestCase):
    def test_constant_and_paired_zero(self):
        values=np.stack([np.full((5,40),2.),np.zeros((5,40))])
        intervals=crossed_bootstrap(values,200)
        np.testing.assert_allclose(intervals,[[2,0],[2,0]])

    def test_sign_and_seed_uncertainty(self):
        values=np.broadcast_to(np.arange(1,6)[None,:,None],(1,5,80)).copy()
        bounds=crossed_bootstrap(values,400)
        self.assertLess(bounds[0,0],3)
        self.assertGreater(bounds[1,0],3)
        self.assertGreater(bounds[0,0],0)
        np.testing.assert_allclose(crossed_bootstrap(-values,400),-bounds[::-1])

    def test_invalid_input(self):
        with self.assertRaises(ValueError):crossed_bootstrap(np.zeros((5,5)),100)
        with self.assertRaises(ValueError):crossed_bootstrap(np.zeros((1,0,5)),100)
        with self.assertRaises(ValueError):crossed_bootstrap(np.full((1,5,5),np.nan),100)

    def test_bayes_boundaries(self):
        x,y,t=data(1000,73)
        np.testing.assert_allclose(bayes_prediction(x,t).numpy(),y.numpy(),atol=1e-6)
        z=perturb(x,'mask',1,1)
        self.assertTrue(torch.allclose(bayes_prediction(z,t),torch.zeros_like(y)))
        np.testing.assert_allclose(theoretical_risk(0,[1,1,1,1]),[0,0,0,0],atol=1e-12)
        sine_var=(1-np.exp(-2))/2
        np.testing.assert_allclose(theoretical_risk(8,[1,1,1,np.sqrt(sine_var)]),[1,1,1,1])

    def test_bayes_population_risk(self):
        x,y,t=data(100000,923)
        z=perturb(x,'mask',.25,826)
        loss=(bayes_prediction(z,t)-y).square()
        expected=theoretical_risk(2,[1,1,1,1])
        observed=np.array([float(loss[t==k].mean()) for k in range(4)])
        np.testing.assert_allclose(observed,expected,atol=.025,rtol=0)

if __name__=='__main__':unittest.main()
