import unittest
import torch
from types import SimpleNamespace
from e4 import kl_consistency, noisy_view, fit_e4
from run import MoE, data, normalize


class E4Checks(unittest.TestCase):
    def test_noise_scope_and_replay(self):
        x=torch.randn(40,28)
        def gen(): return torch.Generator().manual_seed(17)
        a=noisy_view(x,.5,gen());b=noisy_view(x,.5,gen())
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(a[:,:12],x[:,:12]))
        self.assertTrue(torch.equal(a[:,20:],x[:,20:]))
        self.assertTrue(torch.equal(noisy_view(x,0,gen()),x))

    def test_kl_direction_gradients_and_identity(self):
        teacher=torch.tensor([[.1,.2,.3,.4]],requires_grad=True)
        logits=torch.tensor([[1.,2.,3.,4.]],requires_grad=True)
        p=logits.softmax(-1)
        loss=kl_consistency(teacher,p)
        expected=torch.nn.functional.kl_div(p.log(),teacher.detach(),reduction='batchmean')
        self.assertTrue(torch.allclose(loss,expected))
        self.assertGreater(float(loss.detach()),0)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(logits.grad.abs().sum()),0)
        self.assertLess(abs(float(kl_consistency(p.detach(),p.detach()))),1e-7)

    def test_zero_weight_matches_augmentation(self):
        torch.set_num_threads(1)
        sets,_,_=normalize(data(40,1),data(20,2))
        args=SimpleNamespace(device='cpu',seed=7,epochs=2,batch=20,lr=.002,
                             balance=.1,kl_weight=0,noise_sd=.5)
        torch.manual_seed(7)
        a=MoE(8);b=MoE(8);b.load_state_dict(a.state_dict())
        ha=fit_e4(a,sets[0],sets[1],args,'augmentation')
        hb=fit_e4(b,sets[0],sets[1],args,'consistency')
        self.assertEqual(ha,hb)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))

if __name__=='__main__':unittest.main()
