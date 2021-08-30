import qaml
import torch
import dimod
import minorminer
import dwave_networkx as dnx

import matplotlib.pyplot as plt
from torch.nn.utils.prune import BasePruningMethod
from dwave.embedding import EmbeddedStructure

class AdaptiveUnstructured(BasePruningMethod):
    r""" Prune (currently unpruned) units in a tensor by adapting to existing
    sampler graph edges. Prune only disconnected chains.

    Args:
        name (str): parameter name within ``module`` on which pruning
            will act.
        sampler (qaml.sampler.NetworkSampler): Network sampler object with
            underlying structure to prune from.

    """
    PRUNING_TYPE = "unstructured"

    def __init__(self, sampler):
        self.sampler = sampler

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone(memory_format=torch.contiguous_format)

        V,H = self.sampler.model.V, self.sampler.model.H
        try:
            embedding = self.sampler.embedding_orig
        except:
            embedding = self.sampler.embedding
        solver_graph = self.sampler.to_networkx_graph()
        # Account for missing interactions between chains by using mask
        for v in range(V):
            for h in range(V,V+H):
                interactions = embedding.interaction_edges(v,h)
                if not any(solver_graph.has_edge(q,t) for q,t in interactions):
                    mask[h-V][v] = 0
        return mask

    @classmethod
    def apply(cls, module, name, sampler):
        return super(AdaptiveUnstructured, cls).apply(module,name,sampler)

def adaptive_unstructured(module,name,sampler):
    AdaptiveUnstructured.apply(module,name,sampler)
    return module
