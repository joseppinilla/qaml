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
        # TODO: The adjacency matrix representation of the BM. This exists
        # regarless of the topology.
        return self._matrix

    def energy(self, visible, hidden):
        # TODO using matrix
        return

    def free_energy(self, visible, beta=1.0):
        # TODO using matrix
        return

    def forward(self, visible):
        return

BM = BoltzmannMachine

class RestrictedBoltzmannMachine(BoltzmannMachine):
    r"""A Boltzmann Machine with connectivity restricted to only between
    visible-hidden units.

    Args:
        V (int): The size of the visible layer.
        H (int): The size of the hidden layer.
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
        """Sample from visible. P(V) = σ(HW^T + b) """
        return torch.sigmoid(F.linear(hidden, self.W.t(), self.bv))

    def forward(self, visible):
        """ Sample from hidden. P(H) = σ(VW^T + c) """
        return torch.sigmoid(F.linear(visible, self.W, self.bh))

    def energy(self, visible, hidden):
        """Compute the Energy of a certain state or batch of states.

                E(v,h) = -bV - cH - VW^TH
        Args:
            visible (Tensor):

            hidden (Tensor):
        """
        # Visible and Hidden contributions (D,V)·(V,1) + (D,H)·(H,1) -> (D,1)
        linear = torch.matmul(visible, self.bv.T) + torch.matmul(hidden, self.bh.T)
        # Quadratic contributions (D,V)·(V,H) -> (D,H)x(1,H) -> (D,H)
        quadratic = visible.matmul(self.W.T).mul(hidden)
        # sum_j((D,H)) -> (D,1)
        return -(linear + torch.sum(quadratic,dim=-1))

    def free_energy(self, visible, beta=1.0):
        """Also called "effective energy", this expression differs from energy
        in that the compounded contributions of the hidden units is added to the
        visible unit contributions. For one input vector:

            F(v) = -bV - \sum_j(\log(1 + \exp(VW^T + c)))_j

        Args:
            visible (Tensor): Input vector of size RBM.V

            beta (float):

        """
        # Visible contributions (D,V)(1,V) -> (D,1)
        first_term = torch.matmul(visible,self.bv)
        # Quadratic contributions (D,V)(H,V)^T  + (1,H) -> (D,H)
        vW_h = F.linear(visible, self.W, self.bh)
        # Compounded Hidden contributions sum_j(log(exp(1 + (D,H)))) -> (D,1)
        second_term = torch.sum(F.softplus(vW_h,beta),dim=-1)

        return -(first_term + second_term)

RBM = RestrictedBoltzmannMachine

class LimitedBoltzmannMachine(BoltzmannMachine):
    """A Boltzmann Machine with added connections between visible-hidden
    and hidden-hidden units.
    """
    def __init__(self,V_in,H_out):
        pass

    def forward(self, visible):
        pass

LBM = LimitedBoltzmannMachine
