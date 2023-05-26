import dimod
import torch
import numpy as np

from qaml.sampler.base import BinaryQuadraticModelNetworkSampler

class ExactNetworkSampler(BinaryQuadraticModelNetworkSampler):

    sample_kwargs = {}

    _Z = None
    _energies = None
    _probabilities = None

    def __init__(self, model, beta=1.0):
        BinaryQuadraticModelNetworkSampler.__init__(self,model,beta)
        self.child = dimod.ExactSolver()

    @property
    def probabilities(self):
        if self._probabilities is None:
            return self.get_probabilities()
        else:
            return self._probabilities

    @property
    def energies(self):
        if self._energies is None:
            return self.get_energies()
        else:
            return self._energies

    @property
    def Z(self):
        if self._Z is None:
            beta = self.beta
            energies = self.get_energies()
            self._Z = np.exp(-beta*energies).sum()
        return self._Z

    def get_energies(self):
        if self.sampleset is None:
            sampleset = self.sample_bm()
        else:
            sampleset = self.sampleset
        energies = sampleset.record['energy']
        # For compatibility
        self._energies = np.ascontiguousarray(energies,dtype='float64')
        return self._energies

    @torch.no_grad()
    def get_probabilities(self):
        beta = self.beta
        energies = self.get_energies()
        self._Z = np.exp(-beta*energies).sum()
        probabilities = np.exp(-beta*energies)/self._Z
        self._probabilities = probabilities.to(torch.float32)
        return self._probabilities

    # @torch.no_grad()
    # def sample_bm(self,fixed_vars={}):
    #     bqm = self.to_bqm(fixed_vars)
    #     bqm.scale(float(self.beta))
    #     sampleset = self.child.sample(bqm)
    #     return sampleset

    def forward(self, input_data=None, num_reads=None, mask=None, **ex_kwargs):
        # Execute model forward hooks and pre-hooks
        _ = self.model.forward()

        if mask is None: mask = np.ones(self.model.V)

        if input_data is None:
            fixed_vars = []
            self.sampleset = self.sample_bm(fixed_vars)
        else:
            num_batches,remainder = divmod(torch.numel(input_data),self.model.V)
            if num_batches>1: raise RuntimeError("Batches not supported.")
            if remainder: raise RuntimeError("Invalid input_data shape.")
            fixed_vars = [{i:v.item() for i,v in enumerate(input_data) if mask[i]}]
            self.sampleset = self.sample_bm(fixed_vars)[0]

        P = self.get_probabilities()
        samples = np.ascontiguousarray(self.sampleset.record.sample,dtype='float64')
        if num_reads is None:
            tensorset = torch.Tensor(samples)
            prob = torch.matmul(torch.Tensor(P),tensorset).unsqueeze(0)
            return prob.split([self.model.V,self.model.H],1)
        else:
            idx = torch.multinomial(P,num_reads,replacement=True)
            samples = samples[idx]
            tensorset = torch.Tensor(samples)
            return tensorset.split([self.model.V,self.model.H],1)
