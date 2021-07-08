import torch
import dimod
import warnings
import minorminer
import dwave.system
import dwave.embedding

import numpy as np

class NetworkSampler(torch.nn.Module):
    def __init__(self, model):
        super(NetworkSampler, self).__init__()
        self.model = model
        # Sampler stores states
        visible_unknown = torch.Tensor([float('NaN')]*model.V)
        self.prob_v = torch.nn.Parameter(visible_unknown, requires_grad=False)

        hidden_unknown = torch.Tensor([float('NaN')]*model.H)
        self.prob_h = torch.nn.Parameter(hidden_unknown, requires_grad=False)

    @property
    def scaling_factor(self):

        h_range = self.properties['h_range']
        J_range = self.properties['j_range']

        bqm = self.binary_quadratic_model
        h = bqm.linear
        J = bqm.quadratic

        alpha = max( max(h.max()/max(h_range),0),
                max(h.min()/min(h_range),0),
                max(J.max()/max(J_range),0),
                max(J.min()/min(J_range),0))

        return 1.0/alpha

    @property
    def binary_quadratic_model(self):
        bias_v = self.model.b.data.numpy()
        bias_h = self.model.c.data.numpy()
        W = self.model.W.data.numpy()
        V = self.model.V

        lin_V = {i: -b.item() for i,b in enumerate(bias_v)}
        lin_H = {j: -c.item() for j,c in enumerate(bias_h)}
        linear = {**lin_V,**{V+j: c for j,c in lin_H.items()}}

        quadratic = {(i,V+j): -W[j][i].item() for i in lin_V for j in lin_H}

        return dimod.BinaryQuadraticModel(linear,quadratic,'BINARY')

    def sample_visible(self):
        try:
            return self.prob_v.bernoulli()
        except RuntimeError as e:
            warnings.warn(f"Invalid probability vector: {self.prob_v}")
            return torch.zeros_like(self.prob_v)

    def sample_hidden(self):
        try:
            return self.prob_h.bernoulli()
        except RuntimeError as e:
            warnings.warn(f"Invalid probability vector: {self.prob_h}")
            return torch.zeros_like(self.prob_h)

class PersistentGibbsNetworkSampler(NetworkSampler):
    """ Sampler for Persistent Constrastive Divergence training with k steps.

        Args:
            model (torch.nn.Module): PyTorch `Module` with `forward` method.

            num_chains (int): PCD keeps N chains at all times. This number must
                match the batch size.

    """
    def __init__(self, model, num_chains):
        super(PersistentGibbsNetworkSampler, self).__init__(model)
        self.prob_v.data = torch.rand(num_chains,model.V)

    def forward(self, num_samples, k=1, init=None):
        prob_vk = self.prob_v if init is None else init
        prob_hk = self.model(prob_vk)

        for _ in range(k):
            prob_vk.data = self.model.generate(prob_hk)
            prob_hk = self.model.forward(prob_vk.bernoulli())

        self.prob_v.data = prob_vk
        self.prob_h.data = prob_hk
        return prob_vk[:num_samples], prob_hk[:num_samples]

class GibbsNetworkSampler(NetworkSampler):

    def __init__(self, model):
        super(GibbsNetworkSampler, self).__init__(model)

    def forward(self, v0, k=1):
        prob_vk = v0.clone()
        prob_hk = self.model(prob_vk)

        for _ in range(k):
            prob_vk.data = self.model.generate(prob_hk)
            prob_hk.data = self.model.forward(prob_vk.bernoulli())

        self.prob_v.data = prob_vk.data
        self.prob_h.data = prob_hk.data
        return prob_vk, prob_hk

class SimulatedAnnealingNetworkSampler(NetworkSampler,dimod.SimulatedAnnealingSampler):

    sa_kwargs = {"num_sweeps":1000}

    def __init__(self, model):
        NetworkSampler.__init__(self,model)
        dimod.SimulatedAnnealingSampler.__init__(self)

    def forward(self, num_reads=100, **kwargs):
        bqm = self.binary_quadratic_model
        sa_kwargs = {**self.sa_kwargs,**kwargs}
        sampleset = self.sample(bqm,num_reads=num_reads,**sa_kwargs)

        sampletensor = torch.Tensor(sampleset.record.sample.copy())
        samples_v,samples_h = sampletensor.split([self.model.V,self.model.H],1)

        return samples_v, samples_h

class ExactNetworkSampler(NetworkSampler,dimod.ExactSolver):

    def __init__(self, model):
        NetworkSampler.__init__(self,model)
        dimod.ExactSolver.__init__(self)

    def forward(self, num_reads=100, **ex_kwargs):
        beta = self.model.beta
        bqm = self.binary_quadratic_model
        solutions = self.sample(bqm,**ex_kwargs)

        Z = sum(np.exp(-beta*E) for E in solutions.record['energy'])
        P = torch.Tensor([np.exp(-beta*E)/Z for E in solutions.record['energy']])
        samples = [solutions.record['sample'][i] for i in torch.multinomial(P,num_reads,replacement=True)]

        sampletensor = torch.Tensor(samples)
        samples_v,samples_h = sampletensor.split([self.model.V,self.model.H],1)

        return samples_v, samples_h

class QuantumAnnealingNetworkSampler(NetworkSampler,dwave.system.DWaveSampler):

    sample_kwargs = {"answer_mode":'raw',
                     "num_spin_reversal_transforms":0,
                     "auto_scale":False,
                     "anneal_schedule":[(0.0,0.0),(0.5,0.5),(10.5,0.5),(11.0,1.0)]}

    embed_kwargs = {"chain_strength":1.6,
                    #"chain_strength":dwave.embedding.chain_strength.scaled,
                    "smear_vartype":dimod.BINARY}

    unembed_kwargs = {"chain_break_fraction":False,
                      "chain_break_method":dwave.embedding.chain_breaks.majority_vote}

    def __init__(self, model, embedding=None,
                 failover=False, retry_interval=-1, **config):
        NetworkSampler.__init__(self,model)
        dwave.system.DWaveSampler.__init__(self,failover,retry_interval,**config)
        self.networkx_graph = self.to_networkx_graph()
        self.sampleset = None
        if embedding is None:
            T = self.networkx_graph
            if 'Restricted' in repr(self.model):
                cache = minorminer.busclique.busgraph_cache(T)
                embedding = cache.find_biclique_embedding(model.V,model.H)
            else:
                S = self.binary_quadratic_model.quadratic
                embedding = minorminer.find_embedding(S,T)
        self.embedding = embedding

    def embed_bqm(self, visible=None, hidden=None, **kwargs):
        embed_kwargs = {**self.embed_kwargs,**kwargs}

        bqm = self.binary_quadratic_model

        bqm.scale(self.scaling_factor)
        bqm.scale(1.0/self.model.beta)

        max_linear = max(bqm.linear.values())

        # TODO: This may be done better using an auxiliary "clamped" node
        if visible is not None:
            for v,bias in enumerate(visible):
                bqm.set_linear(v,bias.item()*max_linear*2)

        if hidden is not None:
            for h,bias in enumerate(hidden,start=self.model.V):
                bqm.set_linear(h,bias.item()*max_linear*2)

        if self.embedding is None:
            return bqm

        return dwave.embedding.embed_bqm(bqm,
                                         embedding=self.embedding,
                                         target_adjacency=self.networkx_graph,
                                         **embed_kwargs)


    def unembed_sampleset(self,**kwargs):
        unembed_kwargs = {**self.unembed_kwargs,**kwargs}
        if self.embedding is None:
            return self.sampleset

        return dwave.embedding.unembed_sampleset(self.sampleset,
                                                 self.embedding,
                                                 self.binary_quadratic_model,
                                                 **unembed_kwargs)

    def forward(self, visible=None, hidden=None, num_reads=100,
                embed_kwargs={}, unembed_kwargs={}, **kwargs):
        sample_kwargs = {**self.sample_kwargs,**kwargs}
        embed_kwargs = {**self.embed_kwargs,**embed_kwargs}
        unembed_kwargs = {**self.unembed_kwargs,**unembed_kwargs}

        bqm = self.embed_bqm(visible,hidden,**embed_kwargs)

        self.sampleset = self.sample(bqm,num_reads=num_reads,**sample_kwargs)

        sampleset = self.unembed_sampleset(**unembed_kwargs)

        sampletensor = torch.tensor(sampleset.record.sample,dtype=torch.float32)
        samples_v,samples_h = sampletensor.split([self.model.V,self.model.H],1)

        return samples_v, samples_h
