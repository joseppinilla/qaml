import minorminer
import numpy as np
import networkx as nx
import dwave.embedding

def biclique_from_cache(model, sampler, mask=None, seed=None):
    """ Fetches the biclique embedding of the model onto the sampler """
    V = np.ma.masked_array(model.visible,mask=mask).compressed()
    H = model.hidden
    T = sampler if isinstance(sampler,nx.Graph) else sampler.to_networkx_graph()
    cache = minorminer.busclique.busgraph_cache(T,seed=seed)
    return cache.find_biclique_embedding(V.tolist(),H.tolist())


def clique_from_cache(model, sampler, mask=None, seed=None):
    """ Fetches the clique embedding of the model onto the sampler """
    V = np.ma.masked_array(model.visible,mask=mask).compressed()
    H = model.hidden
    T = sampler if isinstance(sampler,nx.Graph) else sampler.to_networkx_graph()
    cache = minorminer.busclique.busgraph_cache(T,seed=seed)
    return cache.find_clique_embedding(V.tolist() + H.tolist())


def miner_heuristic(model, sampler, mask=[], seed=None):
    """ Uses minorminer heuristic embedding """
    S = model if isinstance(sampler,nx.Graph) else model.to_networkx_graph()
    fixed_vars = [v for v,m in zip(model.visible,mask) if not m]
    S.remove_nodes_from(fixed_vars)
    T = sampler if isinstance(sampler,nx.Graph) else sampler.to_networkx_graph()
    return minorminer.find_embedding(S,T,random_seed=seed)


def harvest_cliques(model, sampler, mask=None, seed=None):
    """ Yields embeddings of an N-clique while found.
        Note: This method doesn't guarantee a maximum number of cliques nor that
        no other embeddings exist.
    """
    V = np.ma.masked_array(model.visible,mask=mask).compressed()
    H = model.hidden
    nodes = V.tolist() + H.tolist()
    T = sampler if isinstance(sampler,nx.Graph) else sampler.to_networkx_graph()
    Tg = T.copy()
    e_kwargs = {'seed':seed,'use_cache':False}
    while emb:=minorminer.busclique.find_clique_embedding(nodes,Tg,**e_kwargs):
        for v,chain in emb.items():
            Tg.remove_nodes_from(chain)
        yield emb

################################# MODIFIERS ####################################

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
