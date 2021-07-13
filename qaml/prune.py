import qaml
import torch
import dwave
import dimod
import minorminer
import dwave_networkx as dnx

from torch.nn.utils.prune import BasePruningMethod

class DWaveUnstructured(BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor from D-Wave graph.

    Args:
        name (str): parameter name within ``module`` on which pruning
            will act.
        solver (str):

    """
    PRUNING_TYPE = "unstructured"

    def __init__(self, solver, shape, embedding):
        self.shape = shape
        self.embedding = embedding
        self.sampler = dwave.system.DWaveSampler(solver=solver)
        self.solver_graph = self.sampler.to_networkx_graph()
        self.template_graph = dnx.chimera_graph(16,16)


    def compute_mask(self, t, default_mask):
        mask = default_mask.clone(memory_format=torch.contiguous_format)
        cache = minorminer.busclique.busgraph_cache(self.template_graph)
        self.embedding.update(cache.find_biclique_embedding(*self.shape))
        embedding = dwave.embedding.EmbeddedStructure(self.template_graph.edges,self.embedding)

        available = []
        missing = []
        V,H = self.shape
        for v in range(V):
            for h in range(V,V+H):
                for interaction in embedding.interaction_edges(v,h):
                    if self.solver_graph.has_edge(*interaction):
                        available.append((v,h))
                    else:
                        missing.append((v,h))
                        print("MISSING!")

        missing = [edge for edge in missing if edge not in available]

        return mask

    @classmethod
    def apply(cls, module, name, solver, shape, embedding):
        return super(DWaveUnstructured, cls).apply(module,name,solver,shape,embedding)


def dwave_structured(module,name,solver,shape,embedding):
    DWaveUnstructured.apply(module,name,solver,shape,embedding)
    return module
