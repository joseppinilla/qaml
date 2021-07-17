import qaml
import torch
import dimod
import minorminer
import dwave_networkx as dnx

import matplotlib.pyplot as plt
from torch.nn.utils.prune import BasePruningMethod
from dwave.embedding import EmbeddedStructure
from minorminer.utils.polynomialembedder import processor

class AdachiUnstructured(BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor from D-Wave graph using the
    method in [1-2]. Prune only disconnected edges.

    [1] Adachi, S. H., & Henderson, M. P. (2015). Application of Quantum
    Annealing to Training of Deep Neural Networks.
    https://doi.org/10.1038/nature10012

    [2] Job, J., & Adachi, S. (2020). Systematic comparison of deep belief
    network training using quantum annealing vs. classical techniques.
    http://arxiv.org/abs/2009.00134

    Args:
        name (str): parameter name within ``module`` on which pruning
            will act.
        solver (str):

        shape (tuple(int,int)):

        embedding (dict):


    """
    PRUNING_TYPE = "unstructured"

    def __init__(self, sampler, shape):
        self.shape = shape
        self.sampler = sampler
        self.solver_graph = self.sampler.to_networkx_graph()
        self.template_graph = dnx.chimera_graph(16,16)
        self.embedder = processor(self.template_graph.edges(), M=16, N=16, L=4)

        # self.template_graph = dnx.pegasus_graph(16)

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone(memory_format=torch.contiguous_format)
        # cache = minorminer.busclique.busgraph_cache(self.template_graph)
        # self.embedding.update(cache.find_biclique_embedding(*self.shape))
        # embedding = dwave.embedding.EmbeddedStructure(self.template_graph.edges,self.embedding)
        left,right = self.embedder.tightestNativeBiClique(*self.shape,
                                                         chain_imbalance=None)
        emb_dict = {v:chain for v,chain in enumerate(left+right)}
        embedding = EmbeddedStructure(self.template_graph.edges,emb_dict)

        # Create new embedding object?
        missing = []
        V,H = self.shape
        for v in range(V):
            for h in range(V,V+H):
                emb_v = embedding[v]
                emb_h = embedding[h]
                int_v = embedding._interaction_edges[v,h]
                print(int_v)
                int_h = embedding._interaction_edges[h,v]
                print(int_h)
                for i,j in zip(int_v,int_h):
                    print(emb_v[i],emb_h[j])
                    if not self.solver_graph.has_edge(emb_v[i],emb_h[j]):
                        print('MISSING')

                for interaction in embedding.interaction_edges(v,h):
                    if not self.solver_graph.has_edge(*interaction):
                        missing.append((v,h))

        # To account for missing links in chains
        for x in embedding:
            emb_x = embedding[x]
            chain_edges = embedding._chain_edges[x]
            for i,j in chain_edges:
                if not self.solver_graph.has_edge(emb_x[i],emb_x[j]):
                    embedding._chain_edges[x].remove((i,j))

        for v,h in missing:
            mask[h-V][v] = 0

        self.sampler.embedding = embedding
        return mask

    @classmethod
    def apply(cls, module, name, sampler, shape):
        return super(AdachiUnstructured, cls).apply(module,name,sampler,shape)


def adachi_unstructured(module,name,sampler,shape):
    AdachiUnstructured.apply(module,name,sampler,shape)
    return module
