import qaml
import copy
import dimod
import torch
import dwave.system
import dwave.embedding
import dwave.preprocessing

import numpy as np
import networkx as nx
import dwave_networkx as dnx

from dwave.preprocessing import ScaleComposite
from dwave.system import FixedEmbeddingComposite

from qaml.sampler.base import BinaryQuadraticModelNetworkSampler
from qaml.composites import SpinReversalTransformComposite, LenientFixedEmbeddingComposite

def DummySampler(device):
    target = device.to_networkx_graph()
    nodelist = target.nodes
    edgelist = target.edges
    return dimod.StructureComposite(dimod.RandomSampler(),nodelist,edgelist)

class QuantumAnnealingNetworkSampler(BinaryQuadraticModelNetworkSampler):

    sample_kwargs = {
        # DWaveSampler
        "num_reads":10, "annealing_time":20.0, "label":"QAML",
        "answer_mode":"raw", "auto_scale":True,
        # ScaleComposite (Default values overwritten by get_device())
        "bias_range":[-5.0, 5.0], "quadratic_range":[-3.0, 3.0],
        # FixedEmbeddingComposite
        "chain_strength":1.6, "chain_break_fraction":False,
        "chain_break_method":dwave.embedding.chain_breaks.majority_vote,
        "return_embedding": True,
        # SpinReversalTransformComposite
        "num_spin_reversal_transforms":1}

    embedding = None

    def __init__(self, model, embedding=None, mask=None, auto_scale=True,
                 beta=1.0, failover=False, retry_interval=-1, test=False, **conf):
        BinaryQuadraticModelNetworkSampler.__init__(self,model,beta=beta)

        self.auto_scale = auto_scale
        self.device = self.get_device(failover,retry_interval,**conf)
        # If embedding not provided. Allows empty embedding {}
        if embedding is None:
            # Try biclique if RBM
            if 'Restricted' in str(model):
                embedding = qaml.minor.biclique_from_cache(model,self,mask)
            # Try clique otherwise and if biclique fails
            if not embedding:
                embedding = qaml.minor.clique_from_cache(model,self,mask)
            assert embedding, "Embedding not found"

        # Embedded structures have some useful methods
        edgelist = self.to_networkx_graph().edges
        self.embedding = dwave.embedding.EmbeddedStructure(edgelist,embedding)

        child = DummySampler(self.device) if test else self.device
        self.child = SpinReversalTransformComposite(
                        FixedEmbeddingComposite(
                            ScaleComposite(child),
                            self.embedding,scale_aware=True))

    @classmethod
    def get_device(cls, failover=False, retry_interval=-1, **conf):
        device = dwave.system.DWaveSampler(failover,retry_interval,**conf)
        cls.sample_kwargs['bias_range'] = device.properties['h_range']
        cls.sample_kwargs['quadratic_range'] = device.properties['j_range']
        return device

    def to_networkx_graph(self):
        self._networkx_graph = self.device.to_networkx_graph()
        return self._networkx_graph

QASampler = QuantumAnnealingNetworkSampler

class BatchQuantumAnnealingNetworkSampler(QuantumAnnealingNetworkSampler):

    harvest_method = None
    batch_embeddings = None

    def __init__(self, model, batch_embeddings=None, mask=None, auto_scale=True,
                 beta=1.0, failover=False, retry_interval=-1, test=False,
                 harvest_method=qaml.minor.harvest_cliques, **conf):
        BinaryQuadraticModelNetworkSampler.__init__(self,model,beta=beta)

        self.auto_scale = auto_scale
        self.device = self.get_device(failover,retry_interval,**conf)

        if batch_embeddings is None:
            batch_embeddings = list(harvest_method(model,self,mask))
            assert batch_embeddings, "Embedding not found"
        self.embedding = self.combine_embeddings(batch_embeddings)
        self.batch_embeddings = batch_embeddings

        child = DummySampler(self.device) if test else self.device
        self.child = SpinReversalTransformComposite(
                        FixedEmbeddingComposite(
                            ScaleComposite(child),
                            self.embedding,scale_aware=True))

    def combine_embeddings(self, batch_embeddings):
        combined_emb = {}
        offset = len(self.model)
        for i,emb in enumerate(batch_embeddings):
            emb_i = {x+(i*offset):chain for x,chain in emb.items()}
            combined_emb.update(emb_i)
        edgelist = self.networkx_graph.edges
        embedding = dwave.embedding.EmbeddedStructure(edgelist,combined_emb)
        return embedding

    def combine_bqms(self, fixed_vars):
        if len(fixed_vars) == 0:
            ising = self.to_ising()
            batch_bqms = [ising.copy() for _ in self.batch_embeddings]
        else:
            batch_bqms = [self.to_ising(fix).copy() for fix in fixed_vars]

        offset = len(self.model)
        vartype = self.model.vartype
        combined_bqm = dimod.BinaryQuadraticModel.empty(vartype)
        for i,bqm in enumerate(batch_bqms):
            labels = {x:x+(i*offset) for x in bqm.variables}
            relabeled = bqm.relabel_variables(labels,inplace=False)
            combined_bqm.update(relabeled.copy())
        return combined_bqm

    @property
    def batch_size(self):
        return len(self.batch_embeddings) if self.batch_embeddings else None

    def sample_bm(self, fixed_vars=[], **kwargs):
        scalar = None if self.auto_scale else self.beta.item()
        sample_kwargs = {**self.sample_kwargs,**kwargs}
        vartype = self.model.vartype
        if len(fixed_vars) > self.batch_size:
            raise RuntimeError("Input batch size larger than sampler size")

        bqm = self.combine_bqms(fixed_vars)
        response = self.child.sample(bqm,scalar=scalar,**sample_kwargs)
        response.resolve()
        sampleset = response.change_vartype(vartype)

        batch_size = len(fixed_vars) if fixed_vars else self.batch_size

        # (num_reads,VARS*batch_size)
        samples = sampleset.record.sample.copy()
        # (num_reads,VARS*batch_size)   ->   (num_reads,VARS)*batch_size
        split_samples = np.split(samples,batch_size,axis=1)

        info = sampleset.info.copy()
        variables = sampleset.variables.copy()
        # All samples belong to the same BQM. Concatenate and return.
        samplesets = []
        if len(fixed_vars) == 0:
            for i,split in enumerate(split_samples):
                split_set = dimod.SampleSet.from_samples(split,vartype,np.nan,info=info)
                split_set.relabel_variables({k:v for k,v in enumerate(variables)})
                samplesets.append(split_set)
            return dimod.concatenate(samplesets)

        # Or each sampleset is for a different input. Fill in and return.
        for fixed,split in zip(fixed_vars,split_samples):
            split_set = dimod.SampleSet.from_samples(split,vartype,np.nan,info=info)
            split_set.relabel_variables({k:v for k,v in enumerate(variables)})
            fixed_set = dimod.SampleSet.from_samples(fixed,vartype,np.nan)
            samplesets.append(dimod.append_variables(split_set,fixed_set))
        return samplesets

BatchQASampler = BatchQuantumAnnealingNetworkSampler

class AdachiQuantumAnnealingNetworkSampler(QuantumAnnealingNetworkSampler):
    """ Prune (currently unpruned) units in a tensor from D-Wave graph using the
    method in [1-2]. Prune only disconnected edges.

    [1] Adachi, S. H., & Henderson, M. P. (2015). Application of Quantum
    Annealing to Training of Deep Neural Networks.
    https://doi.org/10.1038/nature10012

    [2] Job, J., & Adachi, S. (2020). Systematic comparison of deep belief
    network training using quantum annealing vs. classical techniques.
    http://arxiv.org/abs/2009.00134

    """

    def __init__(self, model, embedding=None, mask=None, auto_scale=True,
                 beta=1.0, failover=False, retry_interval=-1, test=False, **config):
        BinaryQuadraticModelNetworkSampler.__init__(self,model,beta=beta)

        self.auto_scale = auto_scale
        self.device = self.get_device(failover,retry_interval,**config)

        template_graph = self.get_template_graph()

        if embedding is None:
            # Try biclique if RBM
            if 'Restricted' in str(model):
                embedding = qaml.minor.biclique_from_cache(model,template_graph,mask)
            # Try clique otherwise and if biclique fails
            if not embedding:
                embedding = qaml.minor.clique_from_cache(model,template_graph,mask)
            assert embedding, "Embedding not found"

        # Embedded structures have some useful methods
        edgelist = template_graph.edges
        embedding = dwave.embedding.EmbeddedStructure(edgelist,embedding)
        self.embedding_orig = embedding

        child = DummySampler(self.device) if test else self.device
        self.child = SpinReversalTransformComposite(
                        LenientFixedEmbeddingComposite(
                            ScaleComposite(child),
                            embedding,scale_aware=True))
        self.embedding = self.child.child.embedding

    def get_template_graph(self):
        topology_type = self.device.properties['topology']['type']
        shape = self.device.properties['topology']['shape']
        if topology_type == 'zephyr':
            return dnx.zephyr_graph(*shape)
        elif topology_type == 'pegasus':
            return dnx.pegasus_graph(*shape)
        elif topology_type == 'chimera':
            return dnx.chimera_graph(*shape)
        else:
            raise RuntimeError("Sampler `topology_type` not compatible.")

AdachiQASampler = AdachiQuantumAnnealingNetworkSampler

class AdaptiveQuantumAnnealingNetworkSampler(QuantumAnnealingNetworkSampler):
    def __init__(self, model, embedding=None, mask=None, auto_scale=True,
                 beta=1.0, failover=False, retry_interval=-1, test=False, **config):
        BinaryQuadraticModelNetworkSampler.__init__(self,model,beta=beta)

        self.auto_scale = auto_scale
        self.device = self.get_device(failover,retry_interval,**config)

        template_graph = self.get_template_graph()

        if embedding is None:
            # Try biclique if RBM
            if 'Restricted' in str(model):
                embedding = qaml.minor.biclique_from_cache(model,template_graph,mask)
            # Try clique otherwise and if biclique fails
            if not embedding:
                embedding = qaml.minor.clique_from_cache(model,template_graph,mask)
            assert embedding, "Embedding not found"

        # Embedded structures have some useful methods
        edgelist = template_graph.edges
        embedding = dwave.embedding.EmbeddedStructure(edgelist,embedding)

        ########################################################################

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

        ########################################################################

        child = DummySampler(self.device) if test else self.device
        self.child = SpinReversalTransformComposite(
                        FixedEmbeddingComposite(
                            ScaleComposite(child),
                            self.embedding,scale_aware=True))

    def get_template_graph(self):
        topology_type = self.device.properties['topology']['type']
        shape = self.device.properties['topology']['shape']
        if topology_type == 'zephyr':
            return dnx.zephyr_graph(*shape)
        elif topology_type == 'pegasus':
            return dnx.pegasus_graph(*shape)
        elif topology_type == 'chimera':
            return dnx.chimera_graph(*shape)
        else:
            raise RuntimeError("Sampler `topology_type` not compatible.")

AdaptiveQASampler = AdaptiveQuantumAnnealingNetworkSampler

class RepurposeQuantumAnnealingNetworkSampler(QuantumAnnealingNetworkSampler):
    def __init__(self, model, embedding=None, mask=None, auto_scale=True,
                 beta=1.0, failover=False, retry_interval=-1, test=False, **config):
        BinaryQuadraticModelNetworkSampler.__init__(self,model,beta=beta)

        self.auto_scale = auto_scale
        self.device = self.get_device(failover,retry_interval,**config)

        template_graph = self.get_template_graph()

        if embedding is None:
            # Try biclique if RBM
            if 'Restricted' in str(model):
                embedding = qaml.minor.biclique_from_cache(model,template_graph,mask)
            # Try clique otherwise and if biclique fails
            if not embedding:
                embedding = qaml.minor.clique_from_cache(model,template_graph,mask)
            assert embedding, "Embedding not found"

        # Embedded structures have some useful methods
        edgelist = template_graph.edges
        embedding = dwave.embedding.EmbeddedStructure(edgelist,embedding)

        ########################################################################

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
            print(f"Added {new_H-model.H} hidden nodes")
            model.H = new_H
            model.c.data = torch.zeros(model.H)
            model.W.data = torch.randn(new_H,model.V)

        self.embedding = dwave.embedding.EmbeddedStructure(self.networkx_graph.edges,new_embedding)

        ########################################################################

        child = DummySampler(self.device) if test else self.device
        self.child = SpinReversalTransformComposite(
                        FixedEmbeddingComposite(
                            ScaleComposite(child),
                            self.embedding,scale_aware=True))

    def get_template_graph(self):
        topology_type = self.device.properties['topology']['type']
        shape = self.device.properties['topology']['shape']
        if topology_type == 'zephyr':
            return dnx.zephyr_graph(*shape)
        elif topology_type == 'pegasus':
            return dnx.pegasus_graph(*shape)
        elif topology_type == 'chimera':
            return dnx.chimera_graph(*shape)
        else:
            raise RuntimeError("Sampler `topology_type` not compatible.")

RepurposeQASampler = RepurposeQuantumAnnealingNetworkSampler
