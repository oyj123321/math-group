import unittest
import numpy as np
import torch
from run import data, perturb, normalize, MoE, budget, matched_dense

class Invariants(unittest.TestCase):
    def test_repeatable_and_disjoint(self):
        x,y,t=data(40,1)
        xx,yy,tt=data(40,1)
        self.assertTrue(torch.equal(x,xx))
        self.assertFalse(torch.equal(x,data(40,2)[0]))
        self.assertEqual(torch.bincount(t).tolist(),[10]*4)
        self.assertTrue(torch.isfinite(y).all())

    def test_corruption_scope(self):
        x,y,t=data(20,2)
        self.assertTrue(torch.equal(x,perturb(x,'noise',0,4)))
        z=perturb(x,'noise',1,4)
        self.assertTrue(torch.equal(x[:,:12],z[:,:12]))
        self.assertTrue(torch.equal(x[:,20:],z[:,20:]))
        m=perturb(x,'mask',.25,5)
        self.assertTrue(torch.equal(x[:,:4],m[:,:4]))
        self.assertTrue(torch.equal(x[:,12:20],m[:,12:20]))
        self.assertTrue((m[:,20:].sum(1)==2).all())
        self.assertTrue(torch.equal(x,perturb(x,'mask',0,5)))
        for bad in [-.1,1.1]:
            with self.assertRaises(ValueError): perturb(x,'mask',bad,1)
        with self.assertRaises(ValueError): data(0,1)
        with self.assertRaises(ValueError): perturb(x,'bad',1,1)

    def test_routing_intervention_and_gradients(self):
        torch.manual_seed(1)
        x,y,t=data(40,1)
        m=MoE(8)
        pred,p,index=m(x)
        self.assertTrue(torch.allclose(pred,m(x,m.routing(x))[0]))
        (pred-y).square().mean().backward()
        self.assertGreater(sum(float(q.grad.abs().sum()) for q in m.router.parameters()),0)
        # If every expert is identical and gate weights are held fixed, IDs cannot affect predictions.
        for e in m.experts[1:]: e.load_state_dict(m.experts[0].state_dict())
        r=m.routing(x)
        a=m(x,r)[0]; b=m(x,((r[0]+1)%4,r[1],r[2]))[0]
        self.assertTrue(torch.allclose(a,b,atol=1e-6))

    def test_new_controls(self):
        x,y,t=data(40,1)
        mask=perturb(x,'mask',.25,5)
        blind=perturb(x,'mask_blind',.25,5)
        self.assertTrue(torch.equal(mask[:,:20],blind[:,:20]))
        self.assertTrue((blind[:,20:]==0).all())
        cue=perturb(x,'cue',.5,6)
        self.assertEqual(int((cue[:,:4].sum(1)==0).sum()),20)
        self.assertTrue(torch.equal(cue[:,4:],x[:,4:]))
        self.assertTrue(torch.equal(x,perturb(x,'cue',0,6)))
        with self.assertRaises(ValueError): perturb(x,'cue',1.1,1)
        b=budget(MoE(128))
        active=budget(matched_dense(b['active_linear_macs_per_item'],'macs'))
        total=budget(matched_dense(b['parameters'],'parameters'))
        self.assertLessEqual(abs(active['active_linear_macs_per_item']/b['active_linear_macs_per_item']-1),.001)
        self.assertLessEqual(abs(total['parameters']/b['parameters']-1),.001)

if __name__=='__main__': unittest.main()
