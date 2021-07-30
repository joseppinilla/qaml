import torch
import torch.nn.functional as F

class BoltzmannMachine(torch.nn.Module):
    r"""Boltzmann Machine.
    Args:
        V (int): The size of the visible layer.
        H (int): The size of the hidden layer.
        beta (float, optional): Inverse temperature for the distribution.
    """
    V : int # Visible nodes
    H : int # Hidden nodes
    beta : float # Inverse-temperature

    def __init__(self, V, H, beta=1.0):
        super(BM, self).__init__()
        self.V = V
        self.H = H
        if torch.is_tensor(beta):
            self.register_buffer('beta', beta)
        else:
            self.beta = beta

    @property
    def matrix(self):
        """A tirangular matrix representation of the network, this exists
        regardless of the topology. PyTorch doesn't have an efficient triangular
        matrix representation.
        """
        return self._matrix

    def energy(self, visible, hidden):
        # TODO using matrix
        return

    def free_energy(self, visible):
        # TODO using matrix
        return

    def forward(self, visible):
        return

BM = BoltzmannMachine

class RestrictedBoltzmannMachine(BoltzmannMachine):
    r"""A Boltzmann Machine with connectivity restricted to only between visible
    and hidden units.

    Args:
        V (int): The size of the visible layer.
        H (int): The size of the hidden layer.
        beta (float, optional): Inverse temperature for the distribution.
    """

    def __init__(self, V, H, beta=1.0):
        super(RestrictedBoltzmannMachine, self).__init__(V,H,beta)
        # Visible linear bias
        self.b = torch.nn.Parameter(torch.ones(V)*0.5, requires_grad=True)
        # Hidden linear bias
        self.c = torch.nn.Parameter(torch.ones(H)*0.5, requires_grad=True)
        # Visible-Hidden quadratic bias
        self.W = torch.nn.Parameter(torch.randn(H, V)*0.1, requires_grad=True)

    @property
    def matrix(self):
        """A triangular matrix representation of biases and edge weights"""
        A = torch.diag(torch.cat(self.b,self.c))
        A[self.V:,:self.V] = self.W
        return A

    def generate(self, hidden):
        """Sample from visible. P(V) = σ(HW^T + b)"""
        return torch.sigmoid(F.linear(hidden, self.W.T, self.b)*self.beta)

    def forward(self, visible):
        """Sample from hidden. P(H) = σ(VW^T + c)"""
        return torch.sigmoid(F.linear(visible, self.W, self.c)*self.beta)

    def energy(self, visible, hidden):
        """Compute the Energy of a state or batch of states.
                E(v,h) = -bV - cH - VW^TH
        Args:
            visible (Tensor):
            hidden (Tensor):
        """
        # Visible and Hidden contributions (D,V)·(V,1) + (D,H)·(H,1) -> (D,1)
        linear = torch.matmul(visible, self.b.T) + torch.matmul(hidden, self.c.T)
        # Quadratic contributions (D,V)·(V,H) -> (D,H)x(1,H) -> (D,H)
        quadratic = visible.matmul(self.W.T).mul(hidden)
        # sum_j((D,H)) -> (D,1)
        return -self.beta*(linear + torch.sum(quadratic,dim=-1))

    def free_energy(self, visible):
        """Also called "effective energy", this expression differs from energy
        in that the compounded contributions of the hidden units is added to the
        visible unit contributions. For one input vector:

            F(v) = -bV - \sum_j(\log(1 + \exp(VW^T + c)))_j

        Args:
            visible (Tensor): Input vector of size RBM.V
        """
        # Visible contributions (D,V)(1,V) -> (D,1)
        first_term = torch.matmul(visible,self.b)
        # Quadratic contributions (D,V)(H,V)^T  + (1,H) -> (D,H)
        vW_h = F.linear(visible, self.W, self.c)
        # Compounded Hidden contributions sum_j(log(exp(1 + (D,H)))) -> (D,1)
        second_term = torch.sum(F.softplus(vW_h),dim=-1)

        return -self.beta*(first_term + second_term)

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
