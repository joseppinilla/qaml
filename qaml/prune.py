import qaml
import torch
import dimod
import minorminer
import dwave_networkx as dnx

import matplotlib.pyplot as plt
from torch.nn.utils.prune import BasePruningMethod
from dwave.embedding import EmbeddedStructure
from minorminer.utils.polynomialembedder import processor

class AdaptiveUnstructured(BasePruningMethod):
    r""" Prune (currently unpruned) units in a tensor by adapting to existing
    D-Wave graph edges. Prune only disconnected chains.

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

        # D-Wave processor not available for Pegasus
        # self.template_graph = dnx.pegasus_graph(16)

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone(memory_format=torch.contiguous_format)
        # cache = minorminer.busclique.busgraph_cache(self.template_graph)
        # self.embedding.update(cache.find_biclique_embedding(*self.shape))
        # embedding = dwave.embedding.EmbeddedStructure(self.template_graph.edges,self.embedding)
        left,right = self.embedder.tightestNativeBiClique(*self.shape,
                                                         chain_imbalance=None)
        embedding = EmbeddedStructure(self.template_graph.edges,
                        {v:chain for v,chain in enumerate(left+right)})

        V,H = self.shape
        # Account for missing interactions between chains by using mask
        for v in range(V):
            for h in range(V,V+H):
                emb_v, emb_h = embedding[v], embedding[h]
                int_v = embedding._interaction_edges[v,h]
                int_h = embedding._interaction_edges[h,v]
                for interaction in embedding.interaction_edges(v,h):
                    if not self.solver_graph.has_edge(*interaction):
                        mask[h-V][v] = 0

        self.sampler.embedding = embedding
        return mask

    @classmethod
    def apply(cls, module, name, sampler, shape):
        return super(AdaptiveUnstructured, cls).apply(module,name,sampler,shape)

def adaptive_unstructured(module,name,sampler,shape):
    AdaptiveUnstructured.apply(module,name,sampler,shape)
    return module
