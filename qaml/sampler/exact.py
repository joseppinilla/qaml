import dimod
import torch
import numpy as np

from qaml.sampler.base import BQMSampler

class ExactNetworkSampler(BQMSampler):

    _Z = None
    _energies = None
    _probabilities = None

    def __init__(self, model, beta=1.0):
        BinaryQuadraticModelSampler.__init__(self,model,beta)
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

    @torch.no_grad()
    def get_sampleset(self,fixed_vars={}):
        bqm = self.to_bqm(fixed_vars)
        bqm.scale(float(self.beta))
        self.sampleset = self.child.sample(bqm)
        return self.sampleset

    def get_energies(self):
        if self.sampleset is None:
            sampleset = self.get_sampleset()
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

    def forward(self, input_data=[], num_reads=None, **ex_kwargs):
        fixed_vars = {i:v.item() for i,v in enumerate(input_data)}

        solutions = self.get_sampleset(fixed_vars)
        P = self.get_probabilities()
        Z = self.Z

        if num_reads is None:
            tensorset = torch.Tensor(solutions.record.sample)
            prob = torch.matmul(torch.Tensor(P),tensorset).unsqueeze(0)
            if input_data == []:
                return prob.split([self.model.V,self.model.H],1)
            else:
                return input_data.expand(len(prob),self.model.V), prob

        else:
            idx = torch.multinomial(P,num_reads,replacement=True)
            samples = solutions.record.sample[idx]
            tensorset = torch.Tensor(samples)
            if input_data == []:
                return tensorset.split([self.model.V,self.model.H],1)
            else:
                return input_data.expand(num_reads,self.model.V), tensorset

        return vs, hs


class ExactEmbeddedNetworkSampler(ExactNetworkSampler):

    embedding = None

    def __init__(self, model, embedding, target_graph, beta=1.0):
        BinaryQuadraticModelSampler.__init__(self,model,beta)

        self._networkx_graph = target_graph
        nodes,edges = list(target_graph.nodes), list(target_graph.edges)
        str_sampler = dimod.StructureComposite(dimod.ExactSolver(),nodes,edges)

        if not isinstance(embedding,dwave.embedding.EmbeddedStructure):
            edgelist = self.networkx_graph.edges
            embedding = dwave.embedding.EmbeddedStructure(edgelist,embedding)

        self.embedding = embedding
        self.child = dwave.system.FixedEmbeddingComposite(str_sampler,embedding)
