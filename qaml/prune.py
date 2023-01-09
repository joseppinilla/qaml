import dwave
import torch

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

    def __init__(self, sampler, verbose=False):
        self.verbose = verbose
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
        count = 0
        for v in range(V):
            for h in range(V,V+H):
                interactions = embedding.interaction_edges(v,h)
                if not any(solver_graph.has_edge(q,r) for q,r in interactions):
                    mask[h-V][v] = 0
                    count += 1
        if self.verbose: print(f"Edges pruned: {count}")
        return mask

    @classmethod
    def apply(cls, module, name, sampler, verbose=False):
        return super(AdaptiveUnstructured,cls).apply(module,name,sampler,verbose)

def adaptive_unstructured(module, name, sampler, verbose=False):
    AdaptiveUnstructured.apply(module,name,sampler,verbose)
    return module


class PriorityUnstructured(BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, sampler, priority, verbose=False):
        self.verbose = verbose
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
                if not any(solver_graph.has_edge(q,r) for q,r in interactions):
                    temp_mask[h-V][v] = 0
        # Degree of each chain
        degree = {v:d.item() for v,d in zip(range(V),temp_mask.sum(0))}
        # Modify embedding of viible nodes to account for priority qubits
        temp_embedding = dict(embedding)
        v_sort = [v for _,v in sorted(zip(degree.values(),range(V)),reverse=True)]
        v_prio = priority.nonzero().flatten().tolist()
        for p,v in zip(v_prio,v_sort):
            if degree[p] < degree[v]:
                v_chain = temp_embedding[v]
                p_chain = temp_embedding[p]
                temp_embedding[v] = p_chain
                temp_embedding[p] = v_chain

        edgelist = self.sampler.networkx_graph.edges
        embedding = dwave.embedding.EmbeddedStructure(edgelist,temp_embedding)
        self.sampler.embedding = embedding

        # Account for missing interactions between chains by using mask
        count = 0
        for v in range(V):
            for h in range(V,V+H):
                interactions = embedding.interaction_edges(v,h)
                if not any(solver_graph.has_edge(q,t) for q,t in interactions):
                    mask[h-V][v] = 0
                    count += 1
        if self.verbose: print(f"Edges pruned: {count}")
        return mask

    @classmethod
    def apply(cls, module, name, sampler, priority, verbose=False):
        return super(PriorityUnstructured,cls).apply(module,name,sampler,priority,verbose)

def priority_unstructured(module, name, sampler, priority, verbose):
    PriorityUnstructured.apply(module,name,sampler,priority,verbose)
    return module
