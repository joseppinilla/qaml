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
        # TODO using matrix
        return

    def free_energy(self, visible, beta=1.0):
        return

    def forward(self, visible):
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

    def generate(self, hidden):
        return torch.sigmoid(F.linear(hidden, self.W.t(), self.bv))

    def forward(self, visible):
        return torch.sigmoid(F.linear(visible, self.W, self.bh))

    def energy(self, visible, hidden):
        linear = torch.dot(visible, self.bv.T) + torch.dot(hidden, self.bh.T)
        quadratic = torch.dot(torch.inner(visible, self.W), hidden)
        return -(linear + quadratic)

    def free_energy(self, visible, beta=1.0, h_reduction="sum"):
        """ Also called "effective energy", this expression differs from energy
        in that the compounded contributions of the hidden units is added to the
        visible unit contributions.

            E(v) = -a \cdot v - \sum_j(\log(1+\exp(b+vW)))_j

        Args:
            visible (tensor):

            beta (float):

            h_reduction (string, optional):
            Default: "mean"

        """

        # Visible contributions
        first_term = torch.dot(self.bv, visible)

        # Hidden and quadratic contributions
        vW_h = F.linear(visible, self.W, self.bh)
        if h_reduction=="sum":
            second_term = torch.sum(F.softplus(vW_h,beta))
        elif h_reduction=="mean":
            second_term = torch.mean(F.softplus(vW_h,beta))
        else:
            raise ValueError("Unsupported h reduction")

        return -(first_term + second_term)

RBM = RestrictedBoltzmannMachine

class LimitedBoltzmannMachine(BoltzmannMachine):

    def __init__(self,V_in,H_out):
        pass

    def forward(self, visible):
        pass

LBM = LimitedBoltzmannMachine
