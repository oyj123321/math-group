import unittest
import numpy as np
import torch
from run import MoE,data
from factorial_e4 import factorial_predictions
from analyze_factorial import contrasts


class FactorialChecks(unittest.TestCase):
    def test_unchanged_input_collapses_cells(self):
        torch.manual_seed(10);torch.set_num_threads(1)
        model=MoE(8);x,_,_=data(40,1)
        with torch.no_grad():cells,_=factorial_predictions(model,x,x)
        self.assertTrue(all(torch.equal(cells['00'],v) for v in cells.values()))

    def test_crossed_amplitude_is_cached_scalar(self):
        class Fake:
            def routing(self,x):
                clean=bool(x[0,0]==0)
                return torch.full((3,),0 if clean else 1),torch.full((3,),.8 if clean else .6),torch.zeros(3,4)
            def __call__(self,x,route):
                j,w,p=route
                return (j.float()+2)*w,p,j
        cells,_=factorial_predictions(Fake(),torch.zeros(3,28),torch.ones(3,28))
        for k,expected in {'00':1.6,'01':1.2,'10':2.4,'11':1.8}.items():
            self.assertTrue(torch.allclose(cells[k],torch.full((3,),expected)))

    def test_signed_factorial_algebra(self):
        r=np.random.default_rng(1)
        losses={k:r.normal(size=(5,40)) for k in ['00','01','10','11']}
        e=contrasts(losses)
        self.assertTrue(np.allclose(e['joint_restore'],e['identity_restore']+e['amplitude_restore']-e['interaction']))

if __name__=='__main__':unittest.main()
