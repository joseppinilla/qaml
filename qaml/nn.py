import torch
import dimod
import numpy as np
import networkx as nx
import torch.nn.functional as F

class EnergyBasedModel(torch.nn.Module):
    r"""Energy-Based Model
    Args:
        V (int): The size of the visible layer.
        H (int): The size of the hidden layer.
    """
    V : int # Number of Visible nodes
    H : int # Number of Hidden nodes
    vartype : dimod.vartypes.Vartype

    _networkx_graph = None

    def __init__(self, V, H, vartype=dimod.BINARY):
        super(EnergyBasedModel, self).__init__()
        self.V = V
        self.H = H
        self.vartype = dimod.as_vartype(vartype)

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

    def to_networkx_graph(self, node_attribute_name='bias',
                          edge_attribute_name='bias'):

        G = nx.Graph()
        matrix = self.matrix.copy()
        diag = np.diagonal(matrix)

        linear = ((v, {node_attribute_name: bias,
                       'vartype': self.vartype,
                       'subset': 0 if v<self.V else 1})
                        for v, bias in enumerate(diag))
        G.add_nodes_from(linear)

        quadratic = ((u, v, {edge_attribute_name : matrix[u,v],
                             'vartype': self.vartype})
                             for u,v in zip(*np.triu_indices(len(G),1))
                             if matrix[u,v]) # TODO: NaN instead of 0? with mask
        G.add_edges_from(quadratic)

        self._networkx_graph = G
        return self._networkx_graph

    def change_vartype(self, vartype):
        if vartype is dimod.SPIN and self.vartype is dimod.BINARY:
            self.vartype = vartype
            raise NotImplementedError("Method not implemented.")
        elif vartype is dimod.BINARY and self.vartype is dimod.SPIN:
            self.vartype = vartype
            raise NotImplementedError("Method not implemented.")
        else:
            raise ValueError(f"No match between {vartype} to {self.vartype}")

    @property
    def networkx_graph(self):
        if self._networkx_graph is None:
            return self.to_networkx_graph()
        else:
            return self._networkx_graph

    def energy(self, visible, hidden):
        # TODO using matrix
        return

    def free_energy(self, visible):
        # TODO using matrix
        return

    def generate(self, *args, **kwargs):
        raise NotImplementedError("This model doesn't support generate()")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This model doesn't support forward()")

    @property
    def visible(self):
        return torch.arange(self.V)

    @property
    def hidden(self):
        return torch.arange(self.H) + self.V

    def __len__(self):
        return self.V + self.H

    def __repr__(self):
        return f'EnergyBasedModel({self.V},{self.H},{self.vartype.name})'

EBM = EnergyBasedModel

class BoltzmannMachine(EnergyBasedModel):
    r"""A Boltzmann Machine with full connectivity.

    Args:
        V (int): The size of the visible layer.
        H (int): The size of the hidden layer.
    """

    def __init__(self, V, H=0, vartype=dimod.BINARY, h=[-.1,.1], J=[-.1,.1]):
        super(BoltzmannMachine, self).__init__(V,H,vartype)

        lin = torch.distributions.uniform.Uniform(*h)
        quad = torch .distributions.uniform.Uniform(*J)

        # Visible linear bias
        self.b = torch.nn.Parameter(lin.rsample((V,)),requires_grad=True)
        # Hidden linear bias
        self.c = torch.nn.Parameter(lin.rsample((H,)),requires_grad=True)
        # Visible-Visible quadratic bias
        self.vv = torch.nn.Parameter(quad.rsample((V*(V-1)//2,)),requires_grad=True)
        # Hidden-Hidden quadratic bias
        self.hh = torch.nn.Parameter(quad.rsample((H*(H-1)//2,)),requires_grad=True)
        # Visible-Hidden quadratic bias
        self.W = torch.nn.Parameter(quad.rsample((H,V)),requires_grad=True)

    def generate(self, *args, **kwargs):
        return None

    def forward(self, *args, **kwargs):
        return None

    @torch.no_grad()
    def energy(self, visible, hidden, scale=1.0):
        """Compute the Energy of a state or batch of states.
        Args:
            visible (Tensor): Input vector of size RBM.V
            hidden (Tensor): Hidden node vector of size RBM.H
        """
        # Visible and Hidden contributions (D,V)·(V,1) + (D,H)·(H,1) -> (D,1)
        linear = torch.matmul(visible,self.b.T) + torch.matmul(hidden,self.c.T)
        # Quadratic contributions (D,V)·(V,H) -> (D,H)x(1,H) -> sum_j((D,H)) -> (D,1)
        quadratic = torch.sum(visible.matmul(self.W.T).mul(hidden),dim=-1)

        # Visible-Visible contributions
        vi,vj = np.triu_indices(self.V,1)
        vis_vis = torch.sum((visible[:,vi].mul(visible[:,vj])).mul(self.vv),dim=-1)

        # Hidden-Hidden contributions
        hi,hj = np.triu_indices(self.H,1)
        hid_hid = torch.sum((hidden[:,hi].mul(hidden[:,hj])).mul(self.hh),dim=-1)

        # sum_j((D,H)) -> (D,1)
        return -scale*(linear + quadratic + vis_vis + hid_hid)

BM = BoltzmannMachine

class RestrictedBoltzmannMachine(EnergyBasedModel):
    r"""A Boltzmann Machine with restricted connectivity to visible and hidden.

    Args:
        V (int): The size of the visible layer.
        H (int): The size of the hidden layer.
    """

    def __init__(self, V, H, vartype=dimod.BINARY, h=[-4,4], J=[-1,1]):
        super(RestrictedBoltzmannMachine, self).__init__(V,H,vartype)

        lin = torch.distributions.uniform.Uniform(*h)
        quad = torch .distributions.uniform.Uniform(*J)

        # Visible linear bias
        self.b = torch.nn.Parameter(lin.rsample((V,)),requires_grad=True)
        # Hidden linear bias
        self.c = torch.nn.Parameter(lin.rsample((H,)),requires_grad=True)
        # Visible-Visible quadratic bias
        self.vv = None
        # Hidden-Hidden quadratic bias
        self.hh = None
        # Visible-Hidden quadratic bias
        self.W = torch.nn.Parameter(quad.rsample((H,V)),requires_grad=True)

    def generate(self, hidden, scale=1.0):
        """Sample from visible. P(V) = σ(HW^T + b)"""
        if self.vartype is dimod.BINARY:
            return torch.sigmoid(F.linear(hidden,self.W.T,self.b)*scale)
        elif self.vartype is dimod.SPIN:
            return torch.sigmoid(2.0*F.linear(hidden,self.W.T,self.b)*scale)

    def forward(self, visible, scale=1.0):
        """Sample from hidden. P(H) = σ(VW^T + c)"""
        if self.vartype is dimod.BINARY:
            return torch.sigmoid(F.linear(visible,self.W,self.c)*scale)
        elif self.vartype is dimod.SPIN:
            return torch.sigmoid(2.0*F.linear(visible,self.W,self.c)*scale)

    @torch.no_grad()
    def energy(self, visible, hidden, scale=1.0):
        """Compute the Energy of a state or batch of states.
                E(v,h) = -bV - cH - VW^TH
        Args:
            visible (Tensor): Input vector of size RBM.V
            hidden (Tensor): Hidden node vector of size RBM.H
        """
        # Visible and Hidden contributions (D,V)·(V,1) + (D,H)·(H,1) -> (D,1)
        linear = torch.matmul(visible,self.b.T) + torch.matmul(hidden,self.c.T)
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
        if self.vartype is dimod.BINARY:
            sequence = torch.tensor([0.,1.],requires_grad=False).repeat(self.H,1)
        elif self.vartype is dimod.SPIN:
            sequence = torch.tensor([-1.,1.],requires_grad=False).repeat(self.H,1)
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
        if self.vartype is dimod.BINARY:
            sequence = torch.tensor([0.,1.],requires_grad=False).repeat(self.H,1)
        elif self.vartype is dimod.SPIN:
            sequence = torch.tensor([-1.,1.],requires_grad=False).repeat(self.H,1)
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

    def __repr__(self):
        return f'RestrictedBoltzmannMachine({self.V},{self.H},{self.vartype.name})'

RBM = RestrictedBoltzmannMachine

class LimitedBoltzmannMachine(EnergyBasedModel):
    r"""A Restricted Boltzmann Machine with connectivity between hidden units.

        Args:
            V (int): The size of the visible layer.
            H (int): The size of the hidden layer.
    """

    def __init__(self, V, H, vartype=dimod.BINARY, h=[-4,4], J=[-1,1]):
        super(LimitedBoltzmannMachine, self).__init__(V,H,vartype)

        lin = torch.distributions.uniform.Uniform(*h)
        quad = torch .distributions.uniform.Uniform(*J)

        # Visible linear bias
        self.b = torch.nn.Parameter(lin.rsample((V,)),requires_grad=True)
        # Hidden linear bias
        self.c = torch.nn.Parameter(lin.rsample((H,)),requires_grad=True)
        # Visible-Visible quadratic bias
        self.vv = None #TODO: Empty Tensor instead? change in module.matrix
        # Hidden-Hidden quadratic bias
        self.hh = torch.nn.Parameter(quad.rsample((H*(H-1)//2,)),requires_grad=True)
        # Visible-Hidden quadratic bias
        self.W = torch.nn.Parameter(quad.rsample((H,V)),requires_grad=True)

    def generate(self, hidden, scale=1.0):
        """Sample from visible. P(V) = σ(HW^T + b)"""
        if self.vartype is dimod.BINARY:
            return torch.sigmoid(F.linear(hidden,self.W.T,self.b)*scale)
        elif self.vartype is dimod.SPIN:
            return torch.sigmoid(2.0*F.linear(hidden,self.W.T,self.b)*scale)

    def forward(self, *args, **kwargs):
        return None

    def __repr__(self):
        return f'LimitedBoltzmannMachine({self.V},{self.H},{self.vartype.name})'

LBM = LimitedBoltzmannMachine
