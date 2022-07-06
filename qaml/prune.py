import dwave
import torch

import numpy as np
import networkx as nx

from torch.nn.utils.prune import BasePruningMethod

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
        return super(AdaptiveUnstructured,cls).apply(module,name,sampler)

def adaptive_unstructured(module, name, sampler):
    AdaptiveUnstructured.apply(module,name,sampler)
    return module


class PriorityEmbeddingUnstructured(BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, sampler, priority):
        self.sampler = sampler
        self.priority = priority

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone(memory_format=torch.contiguous_format)

        # Does not support embedding_orig, only works for AdaptiveQASampler
        priority = self.priority
        embedding = self.sampler.embedding
        V,H = self.sampler.model.V, self.sampler.model.H
        solver_graph = self.sampler.to_networkx_graph()

        # Account for missing interactions between chains by using mask
        temp_mask = mask.clone()
        for v in range(V):
            for h in range(V,V+H):
                interactions = embedding.interaction_edges(v,h)
                if not any(solver_graph.has_edge(q,t) for q,t in interactions):
                    temp_mask[h-V][v] = 0
        # Degree of each chain
        degree = {v:d.item() for v,d in zip(range(V),temp_mask.sum(0))}
        # Modify embedding of viible nodes to account for priority qubits
        temp_embedding = dict(embedding)
        v_sort = [v for _,v in sorted(zip(degree.values(),range(V)),reverse=True)]
        for p,v in zip(priority,v_sort):
            if degree[p] < degree[v]:
                v_chain = temp_embedding[v]
                p_chain = temp_embedding[p]
                temp_embedding[v] = p_chain
                temp_embedding[p] = v_chain

        edgelist = self.sampler.networkx_graph.edges
        embedding = dwave.embedding.EmbeddedStructure(edgelist,temp_embedding)

        # Account for missing interactions between chains by using mask
        for v in range(V):
            for h in range(V,V+H):
                interactions = embedding.interaction_edges(v,h)
                if not any(solver_graph.has_edge(q,t) for q,t in interactions):
                    mask[h-V][v] = 0

        self.sampler.embedding = embedding
        return mask

    @classmethod
    def apply(cls, module, name, sampler, priority):
        return super(PriorityEmbeddingUnstructured,cls).apply(module,name,sampler,priority)

def priority_embedding_unstructured(module, name, sampler, priority):
    PriorityEmbeddingUnstructured.apply(module,name,sampler,priority)
    return module

def trim_embedding(tgt, emb, src):
    """ This method takes in a target device graph, a minor-embeding and the
        source graph, and returns a trimmed version of the embedding. i.e.
        It removes qubits from the embedded graph if they are dangling and not
        joining two chains or forming a cyclical chain.
        Note: This may create imbalanced chains, which is not recommended."""

    if not isinstance(emb,dwave.embedding.EmbeddedStructure):
        emb = dwave.embedding.EmbeddedStructure(tgt.edges,emb)
    Eg = nx.Graph(e for u,v in src.edges for e in emb.interaction_edges(u,v))
    Eg.add_edges_from(e for v in src.nodes for e in emb.chain_edges(v))

    M = nx.to_numpy_array(Eg,nodelist=Eg.nodes)

    while np.any(idx := np.where(M.sum(1)==1)):
        M[idx,] = 0
        M[:,idx] = 0

    Tg = nx.from_numpy_array(M)
    nx.relabel_nodes(Tg,{k:v for k,v in enumerate(Eg.nodes)},copy=False)
    trimmed = {k:[v for v in emb[k] if Tg.degree[v]] for k in emb}
    return trimmed
