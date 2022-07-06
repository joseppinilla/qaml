import torch
import dimod
import numpy as np
import torch.nn.functional as F

class EnergyBasedModel(torch.nn.Module):
    r"""Energy-Based Model
    Args:
        V (int): The size of the visible layer.
        H (int): The size of the hidden layer.
    """
    V : int # Visible nodes
    H : int # Hidden nodes

    def __init__(self, V, H):
        super(EnergyBasedModel, self).__init__()
        self.V = V
        self.H = H

    @property
    @torch.no_grad()
    def matrix(self):
        """A triangular matrix representation of the network, this exists
        regardless of the topology. PyTorch doesn't have an efficient triangular
        matrix representation.
        """
        V,H = self.V,self.H
        A = torch.diag(torch.cat((self.b,self.c)))
        A[:V,V:] = self.W.T
        if self.vv is not None:
            vi,vj = np.triu_indices(V,1)
            A[vi,vj] = self.vv
        if self.hh is not None:
            hi,hj = np.triu_indices(H,1)
            A[V+hi,V+hj] = self.hh
        return A.detach().numpy()

    def energy(self, visible, hidden):
        # TODO using matrix
        return

    def free_energy(self, visible):
        # TODO using matrix
        return

    def generate(self, visible):
        raise NotImplementedError("This model doesn't support generate()")

    def forward(self, visible):
        raise NotImplementedError("This model doesn't support forward()")

EBM = EnergyBasedModel

class BoltzmannMachine(EnergyBasedModel):
    r"""Boltzmann Machine
    Args:
        V (int): The size of the visible layer.
        H (int): The size of the hidden layer.
    """
    V : int # Visible nodes
    H : int # Hidden nodes

    def __init__(self, V, H):
        super(BoltzmannMachine, self).__init__(V,H)
        self.V = V
        self.H = H
        # Visible linear bias
        self.b = torch.nn.Parameter(torch.randn(V)*0.1,requires_grad=True)
        # Hidden linear bias
        self.c = torch.nn.Parameter(torch.randn(H)*0.1,requires_grad=True)
        # Visible-Visible quadratic bias
        self.vv = torch.nn.Parameter(torch.randn(V*(V-1)//2)*0.1,requires_grad=True)
        # Hidden-Hidden quadratic bias
        self.hh = torch.nn.Parameter(torch.randn(H*(H-1)//2)*0.1,requires_grad=True)
        # Visible-Hidden quadratic bias
        self.W = torch.nn.Parameter(torch.randn(H,V)*0.1,requires_grad=True)


BM = BoltzmannMachine

class RestrictedBoltzmannMachine(EnergyBasedModel):
    r"""A Boltzmann Machine with connectivity restricted to only between visible
    and hidden units.

    Args:
        V (int): The size of the visible layer.
        H (int): The size of the hidden layer.
    """
    V : int # Visible nodes
    H : int # Hidden nodes

    def __init__(self, V, H):
        super(RestrictedBoltzmannMachine, self).__init__(V,H)
        self.V = V
        self.H = H
        # Visible linear bias
        self.b = torch.nn.Parameter(torch.randn(V)*0.1,requires_grad=True)
        # Hidden linear bias
        self.c = torch.nn.Parameter(torch.randn(H)*0.1,requires_grad=True)
        # Visible-Visible quadratic bias
        self.vv = None
        # Hidden-Hidden quadratic bias
        self.hh = None
        # Visible-Hidden quadratic bias
        self.W = torch.nn.Parameter(torch.randn(H,V)*0.1,requires_grad=True)

    def generate(self, hidden, scale=1.0):
        """Sample from visible. P(V) = σ(HW^T + b)"""
        return torch.sigmoid(F.linear(hidden, self.W.T, self.b)*scale)

    def forward(self, visible, scale=1.0):
        """Sample from hidden. P(H) = σ(VW^T + c)"""
        return torch.sigmoid(F.linear(visible, self.W, self.c)*scale)

    @torch.no_grad()
    def energy(self, visible, hidden, scale=1.0):
        """Compute the Energy of a state or batch of states.
                E(v,h) = -bV - cH - VW^TH
        Args:
            visible (Tensor): Input vector of size RBM.V
            hidden (Tensor): Hidden node vector of size RBM.H
        """
        # Visible and Hidden contributions (D,V)·(V,1) + (D,H)·(H,1) -> (D,1)
        linear = torch.matmul(visible, self.b.T) + torch.matmul(hidden, self.c.T)
        # Quadratic contributions (D,V)·(V,H) -> (D,H)x(1,H) -> (D,H)
        quadratic = visible.matmul(self.W.T).mul(hidden)
        # sum_j((D,H)) -> (D,1)
        return -scale*(linear + torch.sum(quadratic,dim=-1))

    @torch.no_grad()
    def free_energy(self, visible, scale=1.0):
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

        return -scale*(first_term + second_term)

    @torch.no_grad()
    def partition_function(self, scale=1.0):
        """Compute the partition function"""
        sequence = torch.tensor([0.,1.],requires_grad=False).repeat(self.H,1)
        h_iter = torch.cartesian_prod(*sequence)
        first_term = torch.matmul(scale*self.c.T,h_iter.T).exp()
        second_term = (1+(scale*F.linear(h_iter,self.W.T,self.b)).exp()).prod(1)
        return (first_term*second_term).sum()

    @torch.no_grad()
    def log_likelihood(self, *tensors, scale=1.0):
        """Compute the log-likelihood for a state or iterable of states
        Warning: This function computes the partition function.
        """
        # Model
        sequence = torch.tensor([0.,1.],requires_grad=False).repeat(self.H,1)
        h_iter = torch.cartesian_prod(*sequence)
        first_term = torch.matmul(scale*self.c.T,h_iter.T).exp()
        second_term = (1+(scale*F.linear(h_iter,self.W.T,self.b)).exp()).prod(1)
        model_term = (first_term*second_term).sum().log()
        # Data
        for visible in tensors:
            v_sequence = visible.repeat(len(h_iter),1)
            data_energies = self.energy(v_sequence,h_iter,scale=scale)
            data_term = torch.exp(-data_energies).sum().log()
            yield (data_term - model_term).item()

RBM = RestrictedBoltzmannMachine

class LimitedBoltzmannMachine(EnergyBasedModel):
    """ TODO: A Boltzmann Machine with added connections between visible-hidden
    and hidden-hidden units.
    """
    def __init__(self,V,H):
        super(LimitedBoltzmannMachine, self).__init__(V,H)
        self.V = V
        self.H = H
        # Visible linear bias
        self.b = torch.nn.Parameter(torch.randn(V)*0.1,requires_grad=True)
        # Hidden linear bias
        self.c = torch.nn.Parameter(torch.randn(H)*0.1,requires_grad=True)
        # Visible-Visible quadratic bias
        self.vv = None
        # Hidden-Hidden quadratic bias
        self.hh = torch.nn.Parameter(torch.randn(H*(H-1)//2)*0.1,requires_grad=True)
        # Visible-Hidden quadratic bias
        self.W = torch.nn.Parameter(torch.randn(H,V)*0.1,requires_grad=True)

LBM = LimitedBoltzmannMachine
