import torch
import dimod
import warnings

import numpy as np

class NetworkSampler(torch.nn.Module):
    r""" Sample generator for the probabilistic model provided.
    Args:
        model (e.g BoltzmannMachine): Generative Network Model
        beta (float, optional): Inverse temperature for the distribution.
    """

    return_prob : bool # Forward returns prob or samples

    def __init__(self, model, beta=1.0):
        super(NetworkSampler, self).__init__()
        self.model = model

        # Sampler stores states
        visible_unknown = torch.full((1,model.V),float('NaN'))
        self.register_buffer('prob_v',visible_unknown)
        hidden_unknown = torch.full((1,model.H),float('NaN'))
        self.register_buffer('prob_h',hidden_unknown)

        # Sampler inverse temperature
        self.register_buffer('beta',torch.as_tensor(beta))

    @torch.no_grad()
    def sample(self, prob):
        try:
            if self.model.vartype is dimod.BINARY:
                return torch.ge(prob,torch.rand(prob.shape)).to(prob.dtype)
            elif self.model.vartype is dimod.SPIN:
                return (2.0*torch.ge(prob,torch.rand(prob.shape))-1.0)
        except RuntimeError as e:
            warnings.warn(f"Invalid probability vector: {self.prob}")
            return torch.zeros_like(self.prob)

    @torch.no_grad()
    def sample_v(self):
        prob_v = self.prob_v.clone()
        return self.sample(prob_v)

    @torch.no_grad()
    def sample_h(self):
        prob_h = self.prob_h.clone()
        return self.sample(prob_h)

class BinaryQuadraticModelSampler(NetworkSampler):
    sample_kwargs = {'num_reads':10,'seed':None}

    _qubo = None
    _ising = None
    _networkx_graph = None

    sampleset = None
    return_prob = False

    def __init__(self, model, beta=1.0):
        super(BinaryQuadraticModelSampler, self).__init__(model,beta)
        self.child = dimod.RandomSampler() # RandomSampler used as placeholder

    def to_bqm(self, fixed_vars={}):
        """Obtain the Binary Quadratic Model of the network, from
        the full matrix representation. Edges with 0.0 biases are ignored.
        Note: The following could also be done for BINARY networks:
        >>> self._bqm = dimod.BinaryQuadraticModel(model.matrix,dimod.BINARY)
        but BQM(M,SPIN) adds linear biases to offset instead of the diagonal"""

        model = self.model
        quadratic = model.matrix
        diag = np.diagonal(quadratic)
        offset = 0.0

        # Fill in the linear biases using the diagonal
        self._bqm = dimod.BinaryQuadraticModel(model.vartype,offset=-offset)
        self._bqm.add_linear_from_array(-diag)
        # Zero out the diagonal, and use the rest of the matrix to fill up BQM
        new_quadratic = np.array(quadratic, copy=True)
        np.fill_diagonal(new_quadratic, 0)
        self._bqm.add_quadratic_from_dense(-new_quadratic)

        self._bqm.fix_variables(fixed_vars)

        return self._bqm

    @property
    def bqm(self):
        if self._bqm is None:
            return self.to_bqm()
        else:
            return self._bqm

    def to_qubo(self, fixed_vars={}):
        self._qubo = self.to_bqm(fixed_vars).binary
        return self._qubo

    @property
    def qubo(self):
        if self._qubo is None:
            return self.to_qubo()
        else:
            return self._qubo

    def to_ising(self, fixed_vars={}):
        self._ising = self.to_bqm(fixed_vars).spin
        return self._ising

    @property
    def ising(self):
        if self._ising is None:
            return self.to_ising()
        else:
            return self._ising

    def to_networkx_graph(self,**kwargs):
        self._networkx_graph = self._ising.to_networkx_graph(**kwargs)
        return self._networkx_graph

    @property
    def networkx_graph(self):
        if self._networkx_graph is None:
            return self.to_networkx_graph()
        else:
            return self._networkx_graph

    def sample_bm(self, fixed_vars=[], **kwargs):
        rnd_kwargs = {**self.sample_kwargs,**kwargs}
        if len(fixed_vars) == 0:
            bqm = self.to_bqm()
            bqm.scale(float(self.beta))
            return self.child.sample(bqm,**rnd_kwargs)
        else:
            samplesets = []
            for fixed in fixed_vars:
                bqm = self.to_bqm(fixed)
                bqm.scale(float(self.beta))
                samplesets.append(self.child.sample(bqm,**rnd_kwargs))
            return samplesets


    def forward(self, input_data=None, **kwargs):
        # Execute model forward hooks and pre-hooks
        _ = self.model.forward()

        if input_data is None:
            fixed_vars = []
        else:
            num_batches,remainder = divmod(torch.numel(input_data),self.model.V)
            if remainder: raise RuntimeError("Invalid input_data shape.")
            shaped = input_data.reshape(-1,self.model.V)
            fixed_vars = [{i:v.item() for i,v in enumerate(d)} for d in shaped]

        self.sampleset = self.sample_bm(fixed_vars,**kwargs)

        if input_data is None:
            # sample_bm returns a Sampleset
            return_prob = False
            samples = self.sampleset.record.sample.copy()
            sampletensor = torch.tensor(samples,dtype=torch.float32)
            return sampletensor.split([self.model.V,self.model.H],1)
        elif num_batches == 1:
            # sample_bm returns a list [Sampleset]
            return_prob = False
            sampleset = self.sampleset[0]
            num_reads = len(sampleset)
            samples = sampleset.record.sample.copy()
            sampletensor = torch.tensor(samples,dtype=torch.float32)
            inputtensor = shaped.expand(num_reads,self.model.V)
            return inputtensor, sampletensor
        else:
            # sample_bm returns a list of |input_data|*[Sampleset]
            # each Sampleset has <num_reads> entries
            return_prob = True
            inputtensor = input_data.detach().copy()
            sampletensor = torch.Tensor()
            for sampleset in self.sampleset:
                samples = self.sampleset.record.sample.copy()
                sampletensor = torch.tensor(samples,dtype=torch.float32)

            return inputtensor,


BQMSampler = BinaryQuadraticModelSampler


class BatchBinaryQuadraticModelNetworkSampler(BQMSampler):
    def __init__(self, model, beta=1.0):
        super(BatchBinaryQuadraticModelSampler, self).__init__(model,beta)

BatchBQMSampler = BatchBinaryQuadraticModelNetworkSampler
