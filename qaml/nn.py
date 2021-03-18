import torch
import torch.nn.functional as F

class BoltzmannMachine(torch.nn.Module):
    r"""Boltzmann Machine.

    Args:
        n_vis (int, optional): The size of visible layer
        n_hid (int, optional): The size of hidden layer
    """

    def __init__(self, V_in, H_out):
        super(BM, self).__init__()
        # Visible linear bias
        self.bv = torch.nn.Parameter(torch.randn(V_in), requires_grad=True)
        # Hidden linear bias
        self.bh = torch.nn.Parameter(torch.randn(H_out), requires_grad=True)
        # Visible-Hidden quadratic bias
        self.W = torch.nn.Parameter(torch.randn(H_out, V_in), requires_grad=True)

    def sample(self):
        return

    def forward(self, input):
        return

BM = BoltzmannMachine

class LimitedBoltzmannMachine(BoltzmannMachine):

    def __init__(self,V_in,H_out):
        pass

    def forward(self, input):
        pass

LBM = LimitedBoltzmannMachine

class RestrictedBoltzmannMachine(BoltzmannMachine):


    def forward(self, input, k=1):
        if self.training:

            for _ in range(k):
                pH_v = torch.sigmoid(F.linear(input, self.W, self.bh))
                pV_h = torch.sigmoid(F.linear(pH_v, self.W.t(), self.bv))
                input.data = pV_h.bernoulli()
            return pH_v

        else:
            return torch.sigmoid(F.linear(input, self.W, self.bh))

RBM = RestrictedBoltzmannMachine

class QuantumAssistedRestrictedBoltzmannMachine(RestrictedBoltzmannMachine):


    def forward(self, input, k=1):
        if self.training:

            for _ in range(k):
                pH_v = torch.sigmoid(F.linear(input, self.W, self.bh))
                pV_h = torch.sigmoid(F.linear(pH_v, self.W.t(), self.bv))
                input.data = pV_h.bernoulli()
            return pH_v

        else:
            return torch.sigmoid(F.linear(input, self.W, self.bh))

QARBM = QuantumAssistedRestrictedBoltzmannMachine
