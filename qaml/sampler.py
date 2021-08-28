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
    r""" Sample generator for the probabilistic model provided.
    Args:
        model (e.g BotlzmannMachine): Generative Network Model
        beta (float, optional): Inverse temperature for the distribution.
    """

    beta : float # Inverse-temperature to match sampler
    scalar : float # Scaling factor to fit sampler's range

    def __init__(self, model, beta=1.0):
        super(NetworkSampler, self).__init__()
        self.model = model
        # Sampler stores states
        visible_unknown = torch.Tensor([float('NaN')]*model.V)
        self.prob_v = torch.nn.Parameter(visible_unknown, requires_grad=False)

        hidden_unknown = torch.Tensor([float('NaN')]*model.H)
        self.prob_h = torch.nn.Parameter(hidden_unknown, requires_grad=False)

        if torch.is_tensor(beta):
            self.register_buffer('beta', beta)
        else:
            self.beta = beta

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
    def __init__(self, model, num_chains, beta=1.0):
        super(PersistentGibbsNetworkSampler, self).__init__(model,beta)
        self.prob_v.data = torch.rand(num_chains,model.V)

    def forward(self, num_samples, k=1, init=None):
        beta = self.beta
        prob_vk = self.prob_v.clone() if init is None else init.clone()
        prob_hk = self.prob_h.clone()

        for _ in range(k):
            prob_hk.data = self.model.forward(prob_vk.bernoulli(),scale=beta)
            prob_vk.data = self.model.generate(prob_hk.bernoulli(),scale=beta)

        self.prob_v.data = prob_vk.data
        self.prob_h.data = prob_hk.data
        return prob_vk[:num_samples], prob_hk[:num_samples]

class GibbsNetworkSampler(NetworkSampler):

    def __init__(self, model, beta=1.0):
        super(GibbsNetworkSampler, self).__init__(model,beta)

    def forward(self, v0, k=1):
        beta = self.beta
        prob_vk = v0.clone()
        prob_hk = self.prob_h.clone()

        for _ in range(k):
            prob_hk.data = self.model.forward(prob_vk.bernoulli(),scale=beta)
            prob_vk.data = self.model.generate(prob_hk.bernoulli(),scale=beta)

        self.prob_v.data = prob_vk.data
        self.prob_h.data = prob_hk.data
        return prob_vk, prob_hk

""" The next samplers formulate the model as a Binary Quadratic Model (BQM) """
class BinaryQuadraticModelSampler(NetworkSampler):

    _bqm = None
    _networkx_graph = None

    sampleset = None

    def __init__(self, model, beta=1.0):
        super(BinaryQuadraticModelSampler, self).__init__(model,beta)

    def to_binary_quadratic_model(self):
        bias_v = self.model.b.data.numpy()
        bias_h = self.model.c.data.numpy()
        W = self.model.W.data.numpy()
        V = self.model.V
        # Linear biases
        lin_V = {i: -b.item() for i,b in enumerate(bias_v)}
        lin_H = {j: -c.item() for j,c in enumerate(bias_h)}
        linear = {**lin_V,**{V+j: c for j,c in lin_H.items()}}
        # Quadratic weights
        quadratic = {}
        for i in lin_V:
            for j in lin_H:
                quadratic[(i,V+j)] = -W[j][i].item()
        # Using "BINARY" formulation to preserve model
        self._bqm = dimod.BinaryQuadraticModel(linear,quadratic,'BINARY')
        return self._bqm

    def to_networkx_graph(self):
        self._networkx_graph = self.bqm.to_networkx_graph()
        return self._networkx_graph

    @property
    def bqm(self):
        if self._bqm is None:
            return self.to_binary_quadratic_model()
        else:
            return self._bqm

    @property
    def networkx_graph(self):
        if self._networkx_graph is None:
            return self.to_networkx_graph()
        else:
            return self._networkx_graph

BQMSampler = BinaryQuadraticModelSampler

class SimulatedAnnealingNetworkSampler(dimod.SimulatedAnnealingSampler,
                                       BinaryQuadraticModelSampler):
    sa_kwargs = {"num_sweeps":1000}

    def __init__(self, model, beta=1.0):
        BinaryQuadraticModelSampler.__init__(self,model,beta)
        dimod.SimulatedAnnealingSampler.__init__(self)

    def forward(self, num_reads=100, **kwargs):
        bqm = self.to_binary_quadratic_model()
        bqm.scale(float(self.beta))
        sa_kwargs = {**self.sa_kwargs,**kwargs}
        sampleset = self.sample(bqm,num_reads=num_reads,**sa_kwargs)
        samples = sampleset.record.sample.copy()
        sampletensor = torch.tensor(samples,dtype=torch.float32)
        samples_v,samples_h = sampletensor.split([self.model.V,self.model.H],1)

        self.sampleset = sampleset
        return samples_v, samples_h

SASampler = SimulatedAnnealingNetworkSampler

class ExactNetworkSampler(dimod.ExactSolver,BinaryQuadraticModelSampler):

    def __init__(self, model, beta=1.0):
        BinaryQuadraticModelSampler.__init__(self,model,beta)
        dimod.ExactSolver.__init__(self)

    def forward(self, num_reads=100, **ex_kwargs):
        beta = self.beta
        bqm = self.to_binary_quadratic_model()

        solutions = self.sample(bqm,**ex_kwargs)
        energies = solutions.record['energy']
        Z = np.exp(-beta*energies).sum()
        P = torch.Tensor(np.exp(-beta*energies/Z))
        samples = [solutions.record['sample'][i]
                   for i in torch.multinomial(P,num_reads,replacement=True)]

        sampletensor = torch.Tensor(samples)
        samples_v,samples_h = sampletensor.split([self.model.V,self.model.H],1)

        self.sampleset = solutions
        return samples_v, samples_h

class QuantumAnnealingNetworkSampler(dwave.system.DWaveSampler,
                                     BinaryQuadraticModelSampler):
    sample_kwargs = {"answer_mode":'raw',
                     "num_spin_reversal_transforms":0,
                     "anneal_schedule":[(0.0,0.0),(0.5,0.5),(10.5,0.5),(11.0,1.0)],
                     "auto_scale":False}

    embed_kwargs = {"chain_strength":1.6}

    unembed_kwargs = {"chain_break_fraction":False,
                      "chain_break_method":dwave.embedding.chain_breaks.majority_vote}

    embedding = None
    target_bqm = None
    target_sampleset = None

    def __init__(self, model, embedding=None, beta=1.0,
                 failover=False, retry_interval=-1, **config):
        BinaryQuadraticModelSampler.__init__(self,model,beta=beta)
        dwave.system.DWaveSampler.__init__(self,failover,retry_interval,**config)
        if embedding is None:
            if 'Restricted' in repr(self.model):
                cache = minorminer.busclique.busgraph_cache(self.networkx_graph)
                embedding = cache.find_biclique_embedding(model.V,model.H)
            else:
                S = self.bqm.quadratic
                embedding = minorminer.find_embedding(S,self.networkx_graph)
            if not embedding:
                warnings.warn("Embedding not found")
        if not isinstance(embedding,dwave.embedding.EmbeddedStructure):
            edgelist = self.networkx_graph.edges
            embedding = dwave.embedding.EmbeddedStructure(edgelist,embedding)
        self.embedding = embedding
        self.scalar = 1.0

    def to_networkx_graph(self):
        self._networkx_graph = dwave.system.DWaveSampler.to_networkx_graph(self)
        return self._networkx_graph

    def embed_bqm(self, visible=None, hidden=None, auto_scale=False, **kwargs):
        embedding = self.embedding
        bqm = self.to_binary_quadratic_model()
        embed_kwargs = {**self.embed_kwargs,**kwargs}

        if auto_scale:
            # Same as target auto_scale but retains scalar
            norm_args = {'bias_range':self.properties['h_range'],
                         'quadratic_range':self.properties['j_range']}
            self.scalar = bqm.normalize(**norm_args)
        else:
            bqm.scale(1.0/float(self.beta))

        return self.embedding.embed_bqm(bqm,**embed_kwargs)

    def unembed_sampleset(self,**kwargs):
        unembed_kwargs = {**self.unembed_kwargs,**kwargs}

        sampleset = dwave.embedding.unembed_sampleset(self.target_sampleset,
                                                      self.embedding,
                                                      self.bqm,
                                                      **unembed_kwargs)
        return sampleset

    def forward(self, num_reads, visible=None, hidden=None, auto_scale=False,
                embed_kwargs={}, unembed_kwargs={}, **kwargs):

        embed_kwargs = {**self.embed_kwargs,**embed_kwargs}
        sample_kwargs = {**self.sample_kwargs,**kwargs,'num_reads':num_reads}
        unembed_kwargs = {**self.unembed_kwargs,**unembed_kwargs}

        self.target_bqm = self.embed_bqm(visible,hidden,auto_scale,**embed_kwargs)
        self.target_sampleset = self.sample(self.target_bqm,**sample_kwargs)
        sampleset = self.unembed_sampleset(**unembed_kwargs)

        sampletensor = torch.tensor(sampleset.record.sample.copy(),dtype=torch.float32)
        samples_v,samples_h = sampletensor.split([self.model.V,self.model.H],1)

        self.sampleset = sampleset
        return samples_v, samples_h

QASampler = QuantumAnnealingNetworkSampler

class AdachiQASampler(QASampler):
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

    embedding = None
    embedding_orig = None

    target_bqm = None
    target_sampleset = None

    def __init__(self, model, embedding=None, beta=1.0,
                 failover=False, retry_interval=-1, **config):
        QASampler.__init__(self,model,{},beta,failover,retry_interval,**config)

        topology_type = self.properties['topology']['type']
        shape = self.properties['topology']['shape']
        if topology_type == 'pegasus':
            self.template_graph = dnx.pegasus_graph(shape[0])
            if embedding is None:
                helper = minorminer.utils.pegasus._pegasus_fragment_helper
                _embedder, _converter = helper(16, self.template_graph)
                _left,_right = _embedder.tightestNativeBiClique(model.V,model.H,chain_imbalance=None)
                left,right = _converter(range(model.V),_left),_converter(range(model.H),_right)
                embedding = {**left,**{k+model.V:v for k,v in right.items()}}
        elif topology_type == 'chimera':
            self.template_graph = dnx.chimera_graph(*shape)
            if embedding is None:
                self.template_graph = dnx.chimera_graph(*shape)
                embedder = processor(self.template_graph.edges(),M=16,N=16,L=4)
                left,right = embedder.tightestNativeBiClique(model.V,model.H,chain_imbalance=None)
                embedding = {v:chain for v,chain in enumerate(left+right)}
        else:
            raise RuntimeError("Sampler `topology_type` not compatible.")
        # Embedded structures have some useful methods
        embedding = dwave.embedding.EmbeddedStructure(self.template_graph.edges,embedding)
        self.embedding_orig = copy.deepcopy(embedding)

        # Create "sub-chains" to allow gaps in chains
        new_embedding = {}
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
        embedding = self.embedding
        bqm = self.to_binary_quadratic_model().copy()
        embed_kwargs = {**self.embed_kwargs,**kwargs}

        if auto_scale:
            # Same as target auto_scale but retains scalar
            norm_args = {'bias_range':self.properties['h_range'],
                         'quadratic_range':self.properties['j_range']}
            self.scalar = bqm.normalize(**norm_args)
        else:
            bqm.scale(1.0/float(self.beta))

        # Create new BQM including new subchains
        chain_strength = embed_kwargs['chain_strength']
        target_bqm = dimod.BinaryQuadraticModel.empty('BINARY')
        for v, bias in bqm.linear.items():
            if v in self.embedding:
                chain = self.embedding[v]
                b = bias / len(chain)
                target_bqm.add_variables_from({u: b for u in chain})
                # Values are adjusted as 'BINARY' type
                for p, q in self.embedding.chain_edges(v):
                    target_bqm.add_interaction(p, q, -4*chain_strength)
                    target_bqm.add_variable(p, 2*chain_strength)
                    target_bqm.add_variable(q, 2*chain_strength)
            elif v in self.disjoint_chains:
                disjoint_chain = []
                for i in range(self.disjoint_chains[v]):
                    v_sub = f'{v}_ADACHI_SUBCHAIN_{i}'
                    disjoint_chain += [q for q in self.embedding[v_sub]]
                    for p, q in self.embedding.chain_edges(v_sub):
                        target_bqm.add_interaction(p, q, -4*chain_strength)
                        target_bqm.add_variable(p, 2*chain_strength)
                        target_bqm.add_variable(q, 2*chain_strength)
                b = bias / len(disjoint_chain)
                target_bqm.add_variables_from({u: b for u in disjoint_chain})
            else:
                raise MissingChainError(v)

        target_bqm.add_offset(bqm.offset)

        for (u, v), bias in bqm.quadratic.items():
            if u in self.disjoint_chains:
                iter_u = [f'{u}_ADACHI_SUBCHAIN_{i}' for i in range(self.disjoint_chains[u])]
            else:
                iter_u = [u]
            if v in self.disjoint_chains:
                iter_v = [f'{v}_ADACHI_SUBCHAIN_{i}' for i in range(self.disjoint_chains[v])]
            else:
                iter_v = [v]

            interactions = []
            for i_u in iter_u:
                for i_v in iter_v:
                    interactions+=list(self.embedding.interaction_edges(i_u,i_v))

            if interactions:
                b = bias / len(interactions)
                target_bqm.add_interactions_from((u,v,b) for u,v in interactions)

        return target_bqm

    def unembed_sampleset(self, **kwargs):
        unembed_kwargs = {**self.unembed_kwargs,**kwargs}

        pruned_embedding = {v:(q for q in chain if q in self.networkx_graph)
                            for v,chain in self.embedding_orig.items()}

        sampleset = dwave.embedding.unembed_sampleset(self.target_sampleset,
                                                      pruned_embedding,
                                                      self.bqm,
                                                      **unembed_kwargs)

        return sampleset
