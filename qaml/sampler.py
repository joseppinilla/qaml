import copy
import torch
import dimod
import warnings
import minorminer
import dwave.system
import dwave.embedding

import numpy as np
import networkx as nx
import dwave_networkx as dnx

from minorminer.utils.polynomialembedder import processor

class NetworkSampler(torch.nn.Module):

    scalar : float

    def __init__(self, model):
        super(NetworkSampler, self).__init__()
        self.model = model
        # Sampler stores states
        visible_unknown = torch.Tensor([float('NaN')]*model.V)
        self.prob_v = torch.nn.Parameter(visible_unknown, requires_grad=False)

        hidden_unknown = torch.Tensor([float('NaN')]*model.H)
        self.prob_h = torch.nn.Parameter(hidden_unknown, requires_grad=False)

    @property
    def binary_quadratic_model(self):
        bias_v = self.model.b.data.numpy()
        bias_h = self.model.c.data.numpy()
        W = self.model.W.data.numpy()
        V = self.model.V

        lin_V = {i: -b.item() for i,b in enumerate(bias_v)}
        lin_H = {j: -c.item() for j,c in enumerate(bias_h)}
        linear = {**lin_V,**{V+j: c for j,c in lin_H.items()}}

        # Values from RBM are for QUBO formulation i.e. BINARY [0,1]
        quadratic = {}
        for i in lin_V:
            for j in lin_H:
                weight = -W[j][i].item()
                if weight == 0: continue
                quadratic[(i,V+j)] = weight

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

    scalar : float

    sample_kwargs = {"answer_mode":'raw',
                     "num_spin_reversal_transforms":0,
                     "anneal_schedule":[(0.0,0.0),(0.5,0.5),(10.5,0.5),(11.0,1.0)],
                     "auto_scale":False}

    embed_kwargs = {"chain_strength":1.6}

    unembed_kwargs = {"chain_break_fraction":False,
                      "chain_break_method":dwave.embedding.chain_breaks.majority_vote}

    def __init__(self, model, embedding=None,
                 failover=False, retry_interval=-1, **config):
        NetworkSampler.__init__(self,model)
        dwave.system.DWaveSampler.__init__(self,failover,retry_interval,**config)
        self.networkx_graph = self.to_networkx_graph()
        self.sampleset = None
        if embedding is None:
            if 'Restricted' in repr(self.model):
                cache = minorminer.busclique.busgraph_cache(self.networkx_graph)
                embedding = cache.find_biclique_embedding(model.V,model.H)
            else:
                S = self.binary_quadratic_model.quadratic
                embedding = minorminer.find_embedding(S,self.networkx_graph)
        if not isinstance(embedding,dwave.embedding.EmbeddedStructure):
            self.embedding = dwave.embedding.EmbeddedStructure(self.networkx_graph.edges,embedding)
        else:
            self.embedding = embedding
        self.scalar = 1.0

    def embed_bqm(self, visible=None, hidden=None, auto_scale=False, **kwargs):
        embedding = self.embedding
        bqm = self.binary_quadratic_model
        embed_kwargs = {**self.embed_kwargs,**kwargs}

        if auto_scale:
            # Same as target auto_scale but retains scalar
            norm_args = {'bias_range':self.properties['h_range'],
                         'quadratic_range':self.properties['j_range']}
            self.scalar = bqm.normalize(**norm_args)
        else:
            bqm.scale(1.0/float(self.model.beta))

        target_bqm = self.embedding.embed_bqm(bqm, **embed_kwargs)

        return target_bqm

    def unembed_sampleset(self,**kwargs):
        unembed_kwargs = {**self.unembed_kwargs,**kwargs}

        sampleset = dwave.embedding.unembed_sampleset(self.sampleset,
                                                      self.embedding,
                                                      self.binary_quadratic_model,
                                                      **unembed_kwargs)
        return sampleset

    def forward(self, num_reads, visible=None, hidden=None, auto_scale=False,
                embed_kwargs={}, unembed_kwargs={}, **kwargs):

        sample_kwargs = {**self.sample_kwargs,**kwargs}
        embed_kwargs = {**self.embed_kwargs,**embed_kwargs}
        unembed_kwargs = {**self.unembed_kwargs,**unembed_kwargs}

        target_bqm = self.embed_bqm(visible,hidden,auto_scale,**embed_kwargs)

        self.sampleset = self.sample(target_bqm,num_reads=num_reads,**sample_kwargs)

        sampleset = self.unembed_sampleset(**unembed_kwargs)

        sampletensor = torch.tensor(sampleset.record.sample,dtype=torch.float32)
        samples_v,samples_h = sampletensor.split([self.model.V,self.model.H],1)

        return samples_v, samples_h

QASampler = QuantumAnnealingNetworkSampler

class AdachiNetworkSampler(QASampler,dwave.system.DWaveSampler):
    """ Prune (currently unpruned) units in a tensor from D-Wave graph using the
    method in [1-2]. Prune only disconnected edges.

    [1] Adachi, S. H., & Henderson, M. P. (2015). Application of Quantum
    Annealing to Training of Deep Neural Networks.
    https://doi.org/10.1038/nature10012

    [2] Job, J., & Adachi, S. (2020). Systematic comparison of deep belief
    network training using quantum annealing vs. classical techniques.
    http://arxiv.org/abs/2009.00134

    """

    disjoint_chains : dict

    def __init__(self, model, embedding=None,
                 failover=False, retry_interval=-1, **config):
        QASampler.__init__(self,model,{},failover,retry_interval,**config)
        self.target_bqm = None
        self.embedding_orig = None
        self.networkx_graph = self.to_networkx_graph()
        if embedding is None:
            if self.networkx_graph.graph['family'] == 'pegasus':
                self.template_graph = dnx.pegasus_graph(16)
                helper = minorminer.utils.pegasus._pegasus_fragment_helper
                _processor, _converter = helper(16, self.template_graph)
                _left,_right = _processor.tightestNativeBiClique(model.V,model.H,chain_imbalance=None)
                left,right = _converter(range(model.V),_left),_converter(range(model.H),_right)
                embedding = dwave.embedding.EmbeddedStructure(self.template_graph.edges,
                                {**left,**{k+model.V:v for k,v in right.items()}})

            elif self.networkx_graph.graph['family'] == 'chimera':
                self.template_graph = dnx.chimera_graph(16,16)
                self.embedder = processor(self.template_graph.edges(), M=16, N=16, L=4)
                left,right = self.embedder.tightestNativeBiClique(model.V,model.H,chain_imbalance=None)
                embedding = dwave.embedding.EmbeddedStructure(self.template_graph.edges,
                                {v:chain for v,chain in enumerate(left+right)})
            else:
                raise RuntimeError("Graph `family` not compatible.")
        else:
            if self.networkx_graph.graph['family'] == 'pegasus':
                self.template_graph = dnx.pegasus_graph(16)
            elif self.networkx_graph.graph['family'] == 'chimera':
                self.template_graph = dnx.chimera_graph(16,16)
            embedding = dwave.embedding.EmbeddedStructure(self.template_graph.edges,embedding)


        self.embedding_orig = copy.deepcopy(embedding)
        new_embedding = {}

        # Create "sub-chains" to allow gaps in chains
        self.disjoint_chains = {}
        for x in self.embedding_orig:
            emb_x = self.embedding_orig[x]
            chain_edges = self.embedding_orig._chain_edges[x]
            # Very inneficient but does the job of creating chain subgraphs
            chain_graph = nx.Graph()
            chain_graph.add_nodes_from([v for v in emb_x if self.networkx_graph.has_node(v)])
            chain_graph.add_edges_from([(emb_x[i],emb_x[j]) for i,j in chain_edges if self.networkx_graph.has_edge(emb_x[i],emb_x[j])])
            chain_subgraphs = [chain_graph.subgraph(c) for c in nx.connected_components(chain_graph)]

            if len(chain_subgraphs)>1:
                for i,G in enumerate(chain_subgraphs):
                    new_embedding[f'{x}_ADACHI_SUBCHAIN_{i}'] = tuple(G.nodes)
                self.disjoint_chains[x] = len(chain_subgraphs)
            elif len(chain_subgraphs)==1:
                new_embedding[x] = list(chain_graph.nodes)
            else:
                raise RuntimeError(f"No subgraphs were found for chain: {x}")

        self.embedding = dwave.embedding.EmbeddedStructure(self.networkx_graph.edges,new_embedding)

    def embed_bqm(self, visible=None, hidden=None, auto_scale=False, **kwargs):
        embed_kwargs = {**self.embed_kwargs,**kwargs}

        # Create new BQM including new subchains
        bqm_orig = self.binary_quadratic_model.copy()
        smear_vartype = bqm_orig.vartype

        target_bqm = bqm_orig.empty(bqm_orig.vartype)

        chain_strength = embed_kwargs['chain_strength']
        for v, bias in bqm_orig.linear.items():

            if v in self.embedding:
                chain = self.embedding[v]
                b = bias / len(chain)
                target_bqm.add_variables_from({u: b for u in chain})

                if smear_vartype is dimod.SPIN:
                    for p, q in self.embedding.chain_edges(v):
                        target_bqm.add_interaction(p, q, -chain_strength)
                        offset += strength
                else:
                    # this is in spin, but we need to respect the vartype
                    for p, q in self.embedding.chain_edges(v):
                        target_bqm.add_interaction(p, q, -4*chain_strength)
                        target_bqm.add_variable(p, 2*chain_strength)
                        target_bqm.add_variable(q, 2*chain_strength)
            elif v in self.disjoint_chains:
                dis_chain = []
                for i in range(self.disjoint_chains[v]):
                    v_sub = f'{v}_ADACHI_SUBCHAIN_{i}'
                    dis_chain+=[q for q in self.embedding[v_sub]]

                    if smear_vartype is dimod.SPIN:
                        for p, q in self.embedding.chain_edges(v_sub):
                            target_bqm.add_interaction(p, q, -chain_strength)
                            offset += strength
                    else:
                        # this is in spin, but we need to respect the vartype
                        for p, q in self.embedding.chain_edges(v_sub):
                            target_bqm.add_interaction(p, q, -4*chain_strength)
                            target_bqm.add_variable(p, 2*chain_strength)
                            target_bqm.add_variable(q, 2*chain_strength)

                b = bias / len(dis_chain)
                target_bqm.add_variables_from({u: b for u in dis_chain})
            else:
                raise MissingChainError(v)

        target_bqm.add_offset(bqm_orig.offset)

        for (u, v), bias in bqm_orig.quadratic.items():
            if u in self.disjoint_chains:
                interactions =  []
                for i in range(self.disjoint_chains[u]):
                    u_sub = f'{u}_ADACHI_SUBCHAIN_{i}'
                    interactions+=list(self.embedding.interaction_edges(u_sub, v))
            elif v in self.disjoint_chains:
                interactions =  []
                for i in range(self.disjoint_chains[v]):
                    v_sub = f'{v}_ADACHI_SUBCHAIN_{i}'
                    interactions+=list(self.embedding.interaction_edges(u, v_sub))
            else:
                interactions = list(self.embedding.interaction_edges(u, v))

            if interactions:
                b = bias / len(interactions)
                target_bqm.add_interactions_from((u, v, b) for u, v in interactions)

        if auto_scale:
            # Same as target auto_scale but retains scalar
            norm_args = {'bias_range':self.properties['h_range'],
                         'quadratic_range':self.properties['j_range']}
            self.scalar = target_bqm.normalize(**norm_args)
        else:
            target_bqm.scale(1.0/float(self.model.beta))

        self.target_bqm = target_bqm

        return target_bqm

    def unembed_sampleset(self, **kwargs):
        unembed_kwargs = {**self.unembed_kwargs,**kwargs}

        embedding = {v:(q for q in chain if q in self.networkx_graph) for v,chain in self.embedding_orig.items()}

        sampleset = dwave.embedding.unembed_sampleset(self.sampleset,
                                                      embedding,
                                                      self.binary_quadratic_model,
                                                      **unembed_kwargs)

        return sampleset
