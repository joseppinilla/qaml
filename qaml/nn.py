import torch
import torch.nn.functional as F

class BoltzmannMachine(torch.nn.Module):
    r"""Boltzmann Machine.

    """

    def __init__(self):
        super(BM, self).__init__()
        self._matrix = None #TODO

    @property
    def matrix(self):
        #TODO
        return self._matrix

    def energy(self, visible, hidden):
        return

    def free_energy(self, input, beta=1.0):
        return

    def forward(self, input):
        return

BM = BoltzmannMachine

class RestrictedBoltzmannMachine(BoltzmannMachine):
    r""" Restricted Boltzmann Machine.

    Args:
        V (int): The size of visible layer
        H (int): The size of hidden layer
    """

    V : int # Visible nodes
    H : int # Hidden nodes

    def __init__(self, V, H):
        super(RestrictedBoltzmannMachine, self).__init__()
        self.V = V
        self.H = H
        # Visible linear bias
        self.bv = torch.nn.Parameter(torch.ones(V)*0.5, requires_grad=True)
        # Hidden linear bias
        self.bh = torch.nn.Parameter(torch.zeros(H), requires_grad=True)
        # Visible-Hidden quadratic bias
        self.W = torch.nn.Parameter(torch.randn(H, V)*0.1, requires_grad=True)

    def generate(self, hidden=None, k=1):
        p_h_k = hidden.clone()
        for _ in range(k):
            p_v_k = torch.sigmoid(F.linear(p_h_k, self.W.t(), self.bv))
            v_k = p_v_k.bernoulli()
            p_h_k = torch.sigmoid(F.linear(v_k, self.W, self.bh))
        return v_k

    def forward(self, input):
        return torch.sigmoid(F.linear(input, self.W, self.bh))

    def free_energy(self, input, beta=1.0):
        """ E(v) = -a \cdot v - \sum_j(\log(1+\exp(b+vW)))_j """

        visible_contribution = torch.dot(self.bv, input)

        vW_h = F.linear(input, self.W, self.bh)
        hidden_contribution = torch.sum(F.softplus(vW_h,beta))

        return -(visible_contribution + hidden_contribution)

RBM = RestrictedBoltzmannMachine

class LimitedBoltzmannMachine(BoltzmannMachine):

    def __init__(self,V_in,H_out):
        pass

    def forward(self, input):
        pass

LBM = LimitedBoltzmannMachine
