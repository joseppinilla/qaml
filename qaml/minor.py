import itertools
import minorminer
import numpy as np
import networkx as nx
import dwave.embedding
import dwave_networkx as dnx

def biclique_from_cache(model, sampler, mask=None, seed=None):
    """ Fetches the biclique embedding of the model onto the sampler """
    Vnodes = np.ma.masked_array(model.visible,mask=mask).compressed()
    Hnodes = model.hidden
    T = sampler if isinstance(sampler,nx.Graph) else sampler.to_networkx_graph()
    cache = minorminer.busclique.busgraph_cache(T,seed=seed)
    return cache.find_biclique_embedding(Vnodes.tolist(),Hnodes.tolist())


def clique_from_cache(model, sampler, mask=None, seed=None):
    """ Fetches the clique embedding of the model onto the sampler """
    Vnodes = np.ma.masked_array(model.visible,mask=mask).compressed()
    Hnodes = model.hidden
    T = sampler if isinstance(sampler,nx.Graph) else sampler.to_networkx_graph()
    cache = minorminer.busclique.busgraph_cache(T,seed=seed)
    return cache.find_clique_embedding(Vnodes.tolist() + Hnodes.tolist())


def miner_heuristic(model, sampler, mask=None, seed=None):
    """ Uses minorminer heuristic embedding """
    if mask is None: mask = []
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
    Vnodes = np.ma.masked_array(model.visible,mask=mask).compressed()
    Hnodes = model.hidden
    nodes = Vnodes.tolist() + Hnodes.tolist()
    T = sampler if isinstance(sampler,nx.Graph) else sampler.to_networkx_graph()
    Tg = T.copy()
    e_kwargs = {'seed':seed,'use_cache':False}
    while emb:=minorminer.busclique.find_clique_embedding(nodes,Tg,**e_kwargs):
        for v,chain in emb.items():
            Tg.remove_nodes_from(chain)
        yield emb

def _source_from_embedding(model, embedding, Vnodes, Hnodes):
    init = model.to_networkx_graph()

    source = nx.Graph()
    source.add_nodes_from(Vnodes,subset=0)
    source.add_nodes_from(Hnodes,subset=1)
    for u,v in itertools.combinations(list(embedding),2):
        if init.has_edge(u,v):
            interactions = list(embedding.interaction_edges(u,v))
        if interactions:
            source.add_edge(u,v)
    return source

#########################  Limited Boltzmann Machine ###########################

def limited_embedding(model, sampler, mask=None, seed=None):
    Vnodes = np.ma.masked_array(model.visible,mask=mask).compressed().tolist()
    Hnodes = model.hidden.tolist()
    T = sampler if isinstance(sampler,nx.Graph) else sampler.to_networkx_graph()
    cache = minorminer.busclique.busgraph_cache(T,seed=seed)
    bi_embed = cache.find_biclique_embedding(Vnodes,Hnodes)


    emb_dict = {n:[q for q in chain if q in T] for n,chain in bi_embed.items()}
    embedding = dwave.embedding.EmbeddedStructure(T.edges,emb_dict)

    source = _source_from_embedding(model,embedding,Vnodes,Hnodes)

    return source, embedding

###############################  BEST EFFORT ###################################

def bipartite_template(model, sampler, mask=None, seed=None):
    """ Start with bipartite on perfect yield and add more connections """
    V = np.ma.masked_array(model.visible,mask=mask).compressed()
    H = model.hidden

    T = sampler if isinstance(sampler,nx.Graph) else sampler.to_networkx_graph()
    V1,V2 = np.array_split(V,2)

    topology_type = sampler.device.properties['topology']['type']
    shape = sampler.device.properties['topology']['shape']
    assert topology_type == 'pegasus', 'Only Pegasus graph is supported'
    template_graph = dnx.pegasus_graph(shape[0])

    template_cache = minorminer.busclique.busgraph_cache(template_graph,seed=seed)

    clique_emb = template_cache.largest_clique()
    template_emb = template_cache.largest_balanced_biclique()
    Q1, Q2 = np.split(np.arange(len(template_emb)),2)

    emb1 = np.random.choice(Q1,len(V1),replace=False)
    emb2 = np.random.choice(Q2,len(V2),replace=False)

    embedding = {v:template_emb[e] for v,e in zip(V1,emb1)}
    embedding.update({v:template_emb[e] for v,e in zip(V2,emb2)})

    remaining_graph = template_graph.copy()
    for v,chain in embedding.items():
        remaining_graph.remove_nodes_from(chain)

    h_emb = {}
    for h,chain in clique_emb.items():
        emb = []
        for q in chain:
            if q in remaining_graph:
                emb.append(q)
        if emb:
            h_emb[h+len(V)] = emb

    embedding.update(h_emb)
    embedding = {n:[q for q in chain if q in T] for n,chain in embedding.items()}

    return embedding

def bipartite_effort(model, sampler, max_clique=None, mask=None, seed=None):
    """ Start with bipartite and add more connections """
    V = model.V
    Vnodes = np.ma.masked_array(model.visible,mask=mask).compressed()
    H = model.H
    Hnodes = model.hidden

    T = sampler if isinstance(sampler,nx.Graph) else sampler.to_networkx_graph()
    V1,V2 = np.array_split(Vnodes,2)

    cache = minorminer.busclique.busgraph_cache(T,seed=seed)

    # Bipartite Visible
    bi_emb = cache.largest_balanced_biclique()
    Q1, Q2 = np.split(np.arange(len(bi_emb)),2)

    emb1 = np.random.choice(Q1,len(V1),replace=False)
    emb2 = np.random.choice(Q2,len(V2),replace=False)

    embedding = {v:bi_emb[e] for v,e in zip(V1,emb1)}
    embedding.update({v:bi_emb[e] for v,e in zip(V2,emb2)})

    # Occupy with Visible
    remaining_graph = T.copy()
    for v,chain in embedding.items():
        remaining_graph.remove_nodes_from(chain)

    # Clique Hidden
    clique_emb = cache.largest_clique()
    if max_clique is None: max_clique = len(clique_emb)

    hidden = list(clique_emb)
    np.random.shuffle(hidden)

    h_emb = {}
    for h in hidden:

        chain = clique_emb[h]
        emb = []
        for q in chain:
            if q in remaining_graph:
                emb.append(q)
        if emb:
            h_emb[h+V] = emb

        if len(h_emb) == max_clique:
            break

    h_emb = {h:chain for h,chain in enumerate(h_emb.values(),start=len(V))}
    # Reset labels
    embedding.update(h_emb)
    embedding = {n:[q for q in chain if q in T] for n,chain in embedding.items()}

    embedding = dwave.embedding.EmbeddedStructure(T.edges,embedding)

    source = nx.Graph()
    source.add_nodes_from(Vnodes,subset=0)
    source.add_nodes_from(Hnodes,subset=1)
    for u,v in itertools.combinations(list(embedding),2):
        try:
            interactions = list(embedding.interaction_edges(u,v))
        except:
            continue
        if interactions:
            source.add_edge(u,v)

    return source, embedding

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
