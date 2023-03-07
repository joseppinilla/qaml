import copy
import qaml
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

class GibbsNetworkSampler(NetworkSampler):
    """ Sampler for k-step Contrastive Divergence training of RBMs

        Args:
            model (torch.nn.Module): PyTorch `Module` with `forward` method.
            beta (float): inverse temperature for the sampler.
            return_prob (bool, default=True): if True, returns probabilities.
        Example for Contrastive Divergence (CD-k):
            >>> gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm,BATCH_SIZE)
            ...
            >>> # Positive Phase
            >>> v0, prob_h0 = gibbs_sampler(input_data,k=0)
            >>> # Negative Phase
            >>> vk, prob_hk = gibbs_sampler(input_data,k)

        Example for Persistent Contrastive Divergence (PCD):
            >>> gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm,BATCH_SIZE)
            >>> persistent_sampler = qaml.sampler.GibbsNetworkSampler(rbm,NUM_CHAINS)
            ...
            >>> # Positive Phase
            >>> v0, prob_h0 = gibbs_sampler(input_data,k=0)
            >>> # Negative Phase
            >>> vk, prob_hk = persist_sampler(k)

    """

    def __init__(self, model, num_chains, beta=1.0, return_prob=True):
        if 'Restricted' not in repr(model):
            raise RuntimeError("Not Supported")
        super(GibbsNetworkSampler, self).__init__(model,beta)
        self.return_prob = return_prob
        self.prob_v.data = torch.rand(num_chains,model.V)
        self.prob_h.data = torch.rand(num_chains,model.H)

    @torch.no_grad()
    def reconstruct(self, input_data, mask=None, k=1):
        beta = self.beta

        if mask is None: mask = torch.ones_like(self.prob_v)

        # Fill in with 0.5 if BINARY(0,1) or 0.0 if SPIN(-1,+1)
        mask_value = sum(self.model.vartype.value)/2.0
        clamp = torch.mul(input_data.detach().clone(),mask)
        prob_v = clamp.clone().masked_fill_((mask==0),mask_value)
        prob_h = self.model.forward(prob_v,scale=beta)
        for _ in range(k):
            recon = self.model.generate(self.sample(prob_h),scale=beta)
            prob_v.data = clamp + (mask==0)*recon
            prob_h.data = self.model.forward(self.sample(prob_v),scale=beta)

        return prob_v.clone(), prob_h.clone()

    @torch.no_grad()
    def forward(self, input_data=None, k=1):
        beta = self.beta
        model = self.model

        if input_data is not None:
            self.prob_v.data = input_data.clone()
            self.prob_h.data = model.forward(input_data.clone(),scale=beta)

        for _ in range(k):
            self.prob_v.data = model.generate(self.sample_h(),scale=beta)
            self.prob_h.data = model.forward(self.sample_v(),scale=beta)

        if self.return_prob:
            return self.prob_v.clone(), self.prob_h.clone()
        return self.sample_v(), self.sample_h()

""" The next samplers formulate the model as a Binary Quadratic Model (BQM) """
class BinaryQuadraticModelSampler(NetworkSampler):

    _qubo = None
    _ising = None
    _networkx_graph = None

    sampleset = None
    return_prob = False


    def __init__(self, model, beta=1.0):
        super(BinaryQuadraticModelSampler, self).__init__(model,beta)
        _ = self.to_bqm()

    def to_bqm(self, fixed_vars={}):
        """Obtain the Binary Quadratic Model of the network, from
        the full matrix representation. Edges with 0.0 biases are ignored."""
        model = self.model

        quadratic = model.matrix
        diag = np.diagonal(quadratic)
        offset = 0.0

        # The following could also be done for BINARY networks:
        # >>> self._bqm = dimod.BinaryQuadraticModel(model.matrix,dimod.BINARY)
        # but BQM(M,'SPIN') adds linear biases to offset instead of the diagonal

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

BQMSampler = BinaryQuadraticModelSampler

class SimulatedAnnealingNetworkSampler(BinaryQuadraticModelSampler):
    sa_kwargs = {"num_sweeps":1000}

    def __init__(self, model, beta=1.0, **kwargs):
        BinaryQuadraticModelSampler.__init__(self,model,beta)
        self.sampler = dimod.SimulatedAnnealingSampler(**kwargs)

    def forward(self, input_data=None, num_reads=100, **kwargs):
        bqm = self.to_bqm(input_data)
        bqm.scale(float(self.beta))
        sa_kwargs = {**self.sa_kwargs,**kwargs}
        sampleset = self.sampler.sample(bqm,num_reads=num_reads,**sa_kwargs)
        samples = sampleset.record.sample.copy()
        sampletensor = torch.tensor(samples,dtype=torch.float32)
        self.sampleset = sampleset

        if input_data is None:
            return sampletensor.split([self.model.V,self.model.H],1)
        else:
            return input_data.expand(num_reads,self.model.V), sampletensor

SASampler = SimulatedAnnealingNetworkSampler

class ExactNetworkSampler(BinaryQuadraticModelSampler):

    def __init__(self, model, beta=1.0):
        BinaryQuadraticModelSampler.__init__(self,model,beta)
        self.sampler = dimod.ExactSolver()

    @torch.no_grad()
    def get_energies(self, **ex_kwargs):
        if self.sampleset is None:
            bqm = self.to_bqm()
            solutions = self.sampler.sample(bqm,**ex_kwargs)
            self.sampleset = solutions
        else:
            solutions = self.sampleset

        return solutions.record['energy']

    @torch.no_grad()
    def get_probabilities(self, **ex_kwargs):
        beta = self.beta
        energies = self.get_energies(**ex_kwargs)
        Z = np.exp(-beta*energies).sum()
        P = np.exp(-beta*energies)/Z
        return energies, P, Z

    def forward(self, num_reads=None, **ex_kwargs):
        beta = self.beta
        bqm = self.to_bqm()

        solutions = self.sampler.sample(bqm,**ex_kwargs)
        energies = solutions.record['energy']
        Z = np.exp(-beta*energies).sum()
        P = torch.Tensor(np.exp(-beta*energies)/Z)

        if num_reads is None:
            tensorset = torch.Tensor(solutions.record.sample)
            prob = torch.matmul(P,tensorset).unsqueeze(0)
            vs,hs = prob.split([self.model.V,self.model.H],1)
        else:
            idx = torch.multinomial(P,num_reads,replacement=True)
            samples = solutions.record.sample[idx]
            tensorset = torch.Tensor(samples)
            vs,hs = tensorset.split([self.model.V,self.model.H],1)

        self.sampleset = solutions
        return vs, hs

class ExactEmbeddedNetworkSampler(BinaryQuadraticModelSampler):

    def __init__(self, model, beta=1.0, target_graph=None, embedding=None):
        BinaryQuadraticModelSampler.__init__(self,model,beta)

        if target_graph is None:
            target_graph = dnx.chimera_graph(16,16,4)
        self._networkx_graph = target_graph

        struct_sampler = dimod.StructureComposite(dimod.ExactSolver(),
                                                  list(target_graph.nodes),
                                                  list(target_graph.edges))

        if embedding is None:
            if 'Restricted' in repr(self.model):
                cache = minorminer.busclique.busgraph_cache(self.networkx_graph)
                embedding = cache.find_biclique_embedding(model.V,model.H)
            else:
                S = self.qubo.quadratic
                embedding = minorminer.find_embedding(S,self.networkx_graph)
            if not embedding:
                warnings.warn("Embedding not found")

        if not isinstance(embedding,dwave.embedding.EmbeddedStructure):
            edgelist = self.networkx_graph.edges
            embedding = dwave.embedding.EmbeddedStructure(edgelist,embedding)

        self.embedding = embedding
        self.sampler = dwave.system.FixedEmbeddingComposite(struct_sampler,
                                                            embedding)

    def forward(self, num_reads=None, **ex_kwargs):
        beta = self.beta
        bqm = self.to_bqm()

        solutions = self.sampler.sample(bqm,**ex_kwargs)
        energies = solutions.record['energy']
        Z = np.exp(-beta*energies).sum()
        P = torch.Tensor(np.exp(-beta*energies)/Z)

        if num_reads is None:
            tensorset = torch.Tensor(solutions.record.sample)
            prob = torch.matmul(P,tensorset).unsqueeze(0)
            vs,hs = prob.split([self.model.V,self.model.H],1)
        else:
            samples = [solutions.record.sample[i]
                       for i in torch.multinomial(P,num_reads,replacement=True)]
            tensorset = torch.Tensor(samples)
            vs,hs = tensorset.split([self.model.V,self.model.H],1)

        self.sampleset = solutions
        return vs, hs


class QuantumAnnealingNetworkSampler(BinaryQuadraticModelSampler):

    sample_kwargs = {"annealing_time":20.0,"label":"QARBM-DEV"}

    embed_kwargs = {"chain_strength":1.6}

    unembed_kwargs = {"chain_break_fraction":False,
                      "chain_break_method":dwave.embedding.chain_breaks.majority_vote}

    scalar = 1.0
    embedding = None
    auto_scale = True
    batch_mode = False
    batch_embeddings = None

    def __init__(self, model, embedding=None, beta=1.0, auto_scale=True,
                 chain_imbalance=0, failover=False, retry_interval=-1,
                 batch_mode=False, **conf):
        BinaryQuadraticModelSampler.__init__(self,model,beta=beta)

        sampler = dwave.system.DWaveSampler(failover,retry_interval,**conf)
        if embedding is None:
            embedding = qaml.minor.clique_from_cache(model,sampler)
            assert embedding, "Embedding not found"

        if not isinstance(embedding,dwave.embedding.EmbeddedStructure):
            target = sampler.to_networkx_graph()
            embedding = dwave.embedding.EmbeddedStructure(target.edges,embedding)

        if batch_mode:
            target = self.networkx_graph
            embeddings = qaml.minor.harvest_cliques(target,model.H)
            self.batch_embeddings = [{model.V+v:chain for v,chain in emb.items()} for emb in embeddings]
            self.batch_mode = len(self.batch_embeddings)

        self.sampler = sampler
        self.embedding = embedding
        self.auto_scale = auto_scale

    @classmethod
    def get_sampler(cls, failover=False, retry_interval=-1, **conf):
        return dwave.system.DWaveSampler(failover,retry_interval,**conf)

    def set_embedding(self, embedding):
        if not isinstance(embedding,dwave.embedding.EmbeddedStructure):
            target = self.sampler.to_networkx_graph()
            embedding = dwave.embedding.EmbeddedStructure(target.edges,embedding)
        self.embedding = embedding

    def to_networkx_graph(self):
        self._networkx_graph = self.sampler.to_networkx_graph()
        return self._networkx_graph

    def embed_bm(self, ising, embedding=None, flip_variables=[], **embed_kwargs):

        bqm = ising.copy()
        for v in flip_variables:
            bqm.flip_variable(v)

        embedding = self.embedding if embedding is None else embedding
        target_bqm = embedding.embed_bqm(bqm,**embed_kwargs)
        ignoring = [e for u in embedding for e in embedding.chain_edges(u)]
        scale_kwargs = {'ignored_interactions':ignoring}

        if self.auto_scale:
            # Same function as sampler's auto_scale but retains scalar
            scale_kwargs.update({'bias_range':self.sampler.properties['h_range'],
                               'quadratic_range':self.sampler.properties['j_range']})
            self.scalar = target_bqm.normalize(**scale_kwargs)
        else:
            target_bqm.scale(1.0/float(self.beta),**scale_kwargs)
            self.scalar = 1.0

        return target_bqm

    def sample_bm(self, fixed_vars=[], embed_kwargs={}, unembed_kwargs={}, **sample_kwargs):
        num_inputs = len(fixed_vars)
        batch_size = self.batch_mode
        edgelist = self.networkx_graph.edges
        if num_inputs == 0:
            ising = self.to_ising()
            embedding = self.embedding
        elif num_inputs == 1:
            ising = self.to_ising(*fixed_vars)
            clamped_emb = {v:self.embedding[v] for v in ising.variables}
            embedding = dwave.embedding.EmbeddedStructure(edgelist,clamped_emb)
        elif num_inputs <= batch_size:
            batch_embs = self.batch_embeddings
            batch_bqms = [self.to_ising(fix) for fix in fixed_vars]
            ising = dimod.BinaryQuadraticModel.empty(self.model.vartype)
            combined_emb = {}
            for i,(bqm,emb) in enumerate(zip(batch_bqms,batch_embs)):
                labels = {v:v+(i*len(bqm)) for v in bqm.variables}
                combined_emb.update({v+(i*len(bqm)):chain for v,chain in emb.items()})
                ising.update(bqm.relabel_variables(labels,inplace=False))
            embedding = dwave.embedding.EmbeddedStructure(edgelist,combined_emb)
        else:
            raise ValueError(f'Input ({num_inputs}) != batch ({batch_size})')

        embed_kwargs = {**self.embed_kwargs,**embed_kwargs}
        sample_kwargs = {**self.sample_kwargs,**sample_kwargs}
        unembed_kwargs = {**self.unembed_kwargs,**unembed_kwargs}

        num_reads = sample_kwargs.pop('num_reads',100)
        auto_scale = sample_kwargs.pop('auto_scale',self.auto_scale)
        num_spinrevs = sample_kwargs.pop('num_spin_reversal_transforms',0)

        if num_spinrevs > 1:
            reads_per_transform = num_reads//num_spinrevs
            iter_num_reads = [reads_per_transform]*(num_spinrevs-1)
            iter_num_reads += [reads_per_transform+(num_reads%num_spinrevs)]
        else:
            iter_num_reads = [num_reads]

        transform = []
        responses = []

        for num_reads in iter_num_reads:

            # Don't flip if num_spin_reversal_transforms is 0
            if num_spinrevs > 0:
                transform = [v for v in ising.variables if random() > .5]

            target_bqm = self.embed_bm(ising,embedding,flip_variables=transform,**embed_kwargs)
            target_response = self.sampler.sample(target_bqm,auto_scale=False,
                                          num_reads=num_reads,answer_mode='raw',
                                          num_spin_reversal_transforms=0,
                                          **sample_kwargs)
            target_response.resolve()
            target_response.change_vartype(self.model.vartype,inplace=True)

            flipped_response = dwave.embedding.unembed_sampleset(target_response,
                                                                 embedding,ising,
                                                                 **unembed_kwargs)
            tf_idxs = [flipped_response.variables.index(v) for v in transform]
            if self.model.vartype is dimod.BINARY:
                flipped_response.record.sample[:, tf_idxs] = 1 - flipped_response.record.sample[:, tf_idxs]
            elif self.model.vartype is dimod.SPIN:
                flipped_response.record.sample[:, tf_idxs] = -flipped_response.record.sample[:, tf_idxs]
            responses.append(flipped_response)

        sampleset = dimod.sampleset.concatenate(responses)
        samples = sampleset.record.sample.copy() # (num_reads,variables)

        if num_inputs == 0:
            self.sampleset = samples # (num_reads,V+H)
        elif num_inputs == 1:
            fixed, _ = dimod.as_samples(fixed_vars) # (1,V) -> (num_reads,V)
            self.sampleset = np.hstack((np.repeat(fixed,num_reads,0),samples))
        elif num_inputs <= batch_size:
            split_samples = np.split(samples,len(batch_bqms),axis=1) # [(num_reads,H)*BATCH_SIZE]
            mean_samples = np.mean(split_samples,axis=1) #(BATCH_SIZE,H)

            fixed = [f for f,_ in map(dimod.as_samples,fixed_vars)] #(BATCH_SIZE,V)
            self.sampleset = np.hstack((np.concatenate(fixed),mean_samples))

        sampletensor = torch.tensor(self.sampleset,dtype=torch.float32)
        return sampletensor.split([self.model.V,self.model.H],1)

    def reconstruct(self, input_data, mask, num_reads=100,
                    embed_kwargs={}, unembed_kwargs={}, **kwargs):

        kwargs = {**self.sample_kwargs,**kwargs,'num_reads':num_reads}
        fixed_vars = [{i:v.item() for i,v in enumerate(d) if mask[i]} for d in input_data]
        return self.sample_bm(fixed_vars,embed_kwargs,unembed_kwargs,**kwargs)

    def forward(self, input_data=[], num_reads=100,
                embed_kwargs={}, unembed_kwargs={}, **kwargs):

        kwargs = {**self.sample_kwargs,**kwargs,'num_reads':num_reads}
        fixed_vars = [{i:v.item() for i,v in enumerate(d)} for d in input_data]
        return self.sample_bm(fixed_vars,embed_kwargs,unembed_kwargs,**kwargs)

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

    embedding_orig = None

    def __init__(self, model, embedding=None, beta=1.0,
                 failover=False, retry_interval=-1, **config):
        QASampler.__init__(self,model,{},beta,failover,retry_interval,**config)

        topology_type = self.sampler.properties['topology']['type']
        shape = self.sampler.properties['topology']['shape']
        if topology_type == 'pegasus':
            self.template_graph = dnx.pegasus_graph(shape[0])
            if embedding is None:
                cache = minorminer.busclique.busgraph_cache(self.template_graph)
                embedding = cache.find_biclique_embedding(model.V,model.H)
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
                total_chain = sum(len(G) for G in chain_subgraphs)
                for i,G in enumerate(chain_subgraphs):
                    new_embedding[f'{x}_ADACHI_SUBCHAIN_{i}'] = tuple(G.nodes)
                self.disjoint_chains[x] = len(chain_subgraphs)
            elif len(chain_subgraphs)==1:
                new_embedding[x] = list(chain_graph.nodes)
            else:
                raise RuntimeError(f"No subgraphs were found for chain: {x}")
        self.embedding = dwave.embedding.EmbeddedStructure(self.networkx_graph.edges,new_embedding)

    def embed_bm(self, bqm, embedding, **embed_kwargs):

        embed_kwargs = {**self.embed_kwargs,**embed_kwargs}
        embedding = self.embedding if embedding is None else embedding

        # Create new BQM including new subchains
        chain_strength = embed_kwargs['chain_strength']
        target_bqm = dimod.BinaryQuadraticModel.empty('SPIN')
        target_bqm.offset += bqm.offset
        for v, bias in bqm.linear.items():
            if v in embedding:
                chain = embedding[v]
                b = bias / len(chain)
                target_bqm.add_variables_from({u: b for u in chain})
                for p, q in embedding.chain_edges(v):
                    target_bqm.add_interaction(p, q, -chain_strength)
                    target_bqm.offset += chain_strength
            elif v in self.disjoint_chains:
                disjoint_chain = []
                chains = self.disjoint_chains[v]
                for i in range(chains):
                    v_sub = f'{v}_ADACHI_SUBCHAIN_{i}'
                    disjoint_chain += [q for q in embedding[v_sub]]
                    for p, q in embedding.chain_edges(v_sub):
                        target_bqm.add_interaction(p, q, -chain_strength)
                        target_bqm.offset += chain_strength
                b = bias / len(disjoint_chain)
                target_bqm.add_variables_from({u: b for u in disjoint_chain})
            else:
                raise dimod.MissingChainError(v)

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
                    interactions+=list(embedding.interaction_edges(i_u,i_v))

            b = bias / len(interactions)
            target_bqm.add_interactions_from((u,v,b) for u,v in interactions)

        return target_bqm

    def unembed_sampleset(self, response, **unembed_kwargs):
        pruned_embedding = {v:(q for q in chain if q in self.networkx_graph)
                            for v,chain in self.embedding_orig.items()}

        sampleset = dwave.embedding.unembed_sampleset(response,pruned_embedding,
                                                     self.qubo,**unembed_kwargs)

        return sampleset

class AdaptiveQASampler(QASampler):

    def __init__(self, model, embedding=None, beta=1.0,
                 failover=False, retry_interval=-1, **config):
        QASampler.__init__(self,model,{},beta,failover,retry_interval,**config)

        topology_type = self.sampler.properties['topology']['type']
        shape = self.sampler.properties['topology']['shape']
        if topology_type == 'pegasus':
            self.template_graph = dnx.pegasus_graph(shape[0])
            if embedding is None:
                # TODO: Use option with chain_imbalance=None
                # helper = minorminer.utils.pegasus._pegasus_fragment_helper
                # _embedder, _converter = helper(16, self.template_graph)
                # _left,_right = _embedder.tightestNativeBiClique(model.V,model.H,chain_imbalance=None)
                # left,right = _converter(range(model.V),_left),_converter(range(model.H),_right)
                # embedding = {**left,**{k+model.V:v for k,v in right.items()}}
                cache = minorminer.busclique.busgraph_cache(self.template_graph)
                embedding = cache.find_biclique_embedding(model.V,model.H)
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

        # Find "best" subchain (i.e longest) and assign node to it.
        new_embedding = {}
        for x in embedding:
            emb_x = embedding[x]
            chain_edges = embedding._chain_edges[x]
            # Very inneficient but does the job of creating chain subgraphs
            chain_graph = nx.Graph()
            chain_graph.add_nodes_from([v for v in emb_x if self.networkx_graph.has_node(v)])
            chain_graph.add_edges_from([(emb_x[i],emb_x[j]) for i,j in chain_edges if self.networkx_graph.has_edge(emb_x[i],emb_x[j])])
            chain_subgraphs = [(len(c),chain_graph.subgraph(c)) for c in nx.connected_components(chain_graph)]

            if len(chain_subgraphs)>1:
                l,subgraph = max(chain_subgraphs,key=lambda l_chain: l_chain[0])
                new_embedding[x] = list(subgraph.nodes)
            elif len(chain_subgraphs)==1:
                new_embedding[x] = list(chain_graph.nodes)
            else:
                raise RuntimeError(f"No subgraphs were found for chain: {x}")

        self.embedding = dwave.embedding.EmbeddedStructure(self.networkx_graph.edges,new_embedding)

class RepurposeQASampler(QASampler):

    def __init__(self, model, embedding=None, beta=1.0,
                 failover=False, retry_interval=-1, **config):
        QASampler.__init__(self,model,{},beta,failover,retry_interval,**config)

        topology_type = self.sampler.properties['topology']['type']
        shape = self.sampler.properties['topology']['shape']
        if topology_type == 'pegasus':
            self.template_graph = dnx.pegasus_graph(shape[0])
            if embedding is None:
                # TODO: Use option with chain_imbalance=None
                # helper = minorminer.utils.pegasus._pegasus_fragment_helper
                # _embedder, _converter = helper(16, self.template_graph)
                # _left,_right = _embedder.tightestNativeBiClique(model.V,model.H,chain_imbalance=None)
                # left,right = _converter(range(model.V),_left),_converter(range(model.H),_right)
                # embedding = {**left,**{k+model.V:v for k,v in right.items()}}
                cache = minorminer.busclique.busgraph_cache(self.template_graph)
                embedding = cache.find_biclique_embedding(model.V,model.H)
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

        # Find all subchains and create new hidden units where possible
        model_size = model.V+model.H
        new_embedding = {}
        for x in embedding:
            emb_x = embedding[x]
            chain_edges = embedding._chain_edges[x]
            # Very inneficient but does the job of creating chain subgraphs
            chain_graph = nx.Graph()
            chain_graph.add_nodes_from([v for v in emb_x if self.networkx_graph.has_node(v)])
            chain_graph.add_edges_from([(emb_x[i],emb_x[j]) for i,j in chain_edges if self.networkx_graph.has_edge(emb_x[i],emb_x[j])])
            chain_subgraphs = [(len(c),chain_graph.subgraph(c)) for c in nx.connected_components(chain_graph)]

            if len(chain_subgraphs)>1:
                # Visible nodes
                if x<model.V:
                    l,subgraph = max(chain_subgraphs,key=lambda l_chain: l_chain[0])
                    new_embedding[x] = list(subgraph.nodes)
                # Hidden nodes
                else:
                    length,subgraph = chain_subgraphs[0]
                    new_embedding[x] = subgraph.nodes
                    for length,subgraph in chain_subgraphs[1:]:
                        new_x = model_size
                        new_embedding[new_x] = list(subgraph.nodes)
                        model_size+=1

            elif len(chain_subgraphs)==1:
                new_embedding[x] = list(chain_graph.nodes)
            else:
                raise RuntimeError(f"No subgraphs were found for chain: {x}")

        # Modify model to include new hidden nodes
        new_H = model_size-model.V
        if new_H>model.H:
            model.H = new_H
            model.c.data = torch.zeros(model.H)
            model.W.data = torch.randn(new_H,model.V)

        self.embedding = dwave.embedding.EmbeddedStructure(self.networkx_graph.edges,new_embedding)
