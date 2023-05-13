%matplotlib qt
import qaml
import dimod
import torch
import minorminer
import itertools
import numpy as np
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt

VISIBLE_SIZE = 13*13
HIDDEN_SIZE = 6

bm = qaml.nn.BoltzmannMachine(VISIBLE_SIZE,HIDDEN_SIZE)

# For deterministic weights
SEED = 42
_ = torch.manual_seed(SEED)

# torch.nn.utils.prune.random_unstructured(bm,'W',0.2)
# torch.nn.utils.prune.random_unstructured(bm,'vv',0.2)
# torch.nn.utils.prune.random_unstructured(bm,'hh',0.2)

# Set up training mechanisms
solver_name = "Advantage_system6.1"
empty_sampler = qaml.sampler.QASampler(bm,solver=solver_name)


pruned,embedding = qaml.minor.bipartite_effort(bm,empty_sampler,HIDDEN_SIZE)

T = empty_sampler.networkx_graph
S = bm.networkx_graph


_ = plt.figure(figsize=(16,16))
dnx.draw_pegasus_embedding(T,embedding,pruned,node_size=12)


################################################################################
############################# PEGASUS(6) #######################################
################################################################################
T16 = dnx.pegasus_graph(16)
cache16 = minorminer.busclique.busgraph_cache(T16,seed=SEED)
bi_emb16 = cache16.largest_balanced_biclique()
clique_emb16 = cache16.largest_clique()



len(bi_emb16)/2
len(clique_emb16)

T6 = dnx.pegasus_graph(6)
cache6 = minorminer.busclique.busgraph_cache(T6,seed=SEED)
bi_emb6 = cache6.largest_balanced_biclique()
clique_emb6 = cache6.largest_clique()

_ = plt.figure(figsize=(16,16))
dnx.draw_pegasus_embedding(T6,bi_emb6,node_size=60)

_ = plt.figure(figsize=(16,16))
dnx.draw_pegasus_embedding(T6,clique_emb6,node_size=60)
len(clique_emb6)
VISIBLE = 100
HIDDEN = 64
bm6 = qaml.nn.BoltzmannMachine(VISIBLE,HIDDEN)
pruned6,effort6 = qaml.minor.bipartite_effort(bm6,T6,HIDDEN)

_ = plt.figure(figsize=(16,16))
dnx.draw_pegasus_embedding(T6,effort6,node_size=60)
print(len(pruned6) - VISIBLE)

_ = plt.figure(figsize=(16,16))
nx.draw(pruned6,pos=nx.multipartite_layout(pruned6),node_size=60)
