import minorminer
import numpy as np
import networkx as nx
import dwave.embedding


def biclique_from_cache(model, sampler, seed=None):
    """ Fetches the biclique embedding of the model onto the sampler """
    T = sampler if isinstance(sampler,nx.Graph) else sampler.to_networkx_graph()
    cache = minorminer.busclique.busgraph_cache(T,seed=seed)
    return cache.find_biclique_embedding(model.V,model.H)


def clique_from_cache(model, sampler, seed=None):
    """ Fetches the clique embedding of the model onto the sampler """
    S = model if isinstance(model,int) else len(model)
    T = sampler if isinstance(sampler,nx.Graph) else sampler.to_networkx_graph()
    cache = minorminer.busclique.busgraph_cache(T,seed=seed)
    return cache.find_clique_embedding(S)


def miner_heuristic(model, sampler, seed=None):
    """ Uses minorminer heuristic embedding """
    S =  model.networkx_graph
    T = sampler if isinstance(sampler,nx.Graph) else sampler.to_networkx_graph()
    return minorminer.find_embedding(S,T,random_seed=seed)


def harvest_cliques(N, sampler, seed=None):
    """ Yields as many embeddings of an N-clique as possible """
    T = sampler if isinstance(sampler,nx.Graph) else sampler.to_networkx_graph()
    Tcopy = T.copy()
    e_kwargs = {'seed':seed,'use_cache':False}
    while emb:=minorminer.busclique.find_clique_embedding(N,Tcopy,**e_kwargs):
        for v,chain in emb.items():
            Tcopy.remove_nodes_from(chain)
        yield emb

def trim_embedding(tgt, emb, src):
    """ This method takes in a target device graph, a minor-embeding and the
    source graph, and returns a trimmed version of the embedding. i.e.
    It removes qubits from the embedded graph if they are dangling and not
    joining two chains or forming a cyclical chain.
    Note: This may create imbalanced chains, which is not recommended.
    """

    if not isinstance(emb,dwave.embedding.EmbeddedStructure):
        emb = dwave.embedding.EmbeddedStructure(tgt.edges,emb)

    # Create interaction graph, i.e. the embedded qubits graph
    Eg = nx.Graph(e for u,v in src.edges for e in emb.interaction_edges(u,v))
    Eg.add_edges_from(e for v in src.nodes for e in emb.chain_edges(v))

    # Remove any nodes with only one edge
    # Easier in matrix representation
    M = nx.to_numpy_array(Eg,nodelist=Eg.nodes)
    count = 0
    while np.any(idx := np.where(M.sum(1)==1)):
        M[idx,] = 0
        M[:,idx] = 0
        count+=1

    # Create trimmed embedded graph and restore labels (matrix loses labels)
    Tg = nx.from_numpy_array(M)
    Tg = nx.relabel_nodes(Tg,{k:v for k,v in enumerate(Eg.nodes)},copy=True)
    # Create trimmed embedding
    trimmed = {k:[v for v in emb[k] if Tg.degree[v]] for k in emb}
    return trimmed, count
