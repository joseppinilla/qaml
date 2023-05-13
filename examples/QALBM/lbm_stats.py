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


VISIBLE_SIZE = 16
HIDDEN_SIZE = 16

lbm = qaml.nn.LimitedBoltzmannMachine(VISIBLE_SIZE,HIDDEN_SIZE,'SPIN')
Glbm = lbm.to_networkx_graph()
len(Glbm.edges)

solver_name = "Advantage_system4.1"
sampler = qaml.sampler.QASampler(lbm,solver=solver_name)

T = sampler.to_networkx_graph()
S,embedding = qaml.minor.limited_embedding(lbm,sampler)

_ = plt.figure(figsize=(16,16))
dnx.draw_pegasus_embedding(T, embedding, S)


lbm.W.numel()
len(S.edges())
len(lbm.hh.nonzero())
len(lbm.hh.nonzero())



rbm = qaml.nn.RestrictedBoltzmannMachine(VISIBLE_SIZE,HIDDEN_SIZE,'SPIN')
Grbm = rbm.to_networkx_graph()
len(Grbm.edges)



solver_name = "Advantage_system4.1"
sampler = qaml.sampler.QASampler(rbm,solver=solver_name)

S.has_edge(0,1)
S = qaml.minor._source_from_embedding(rbm,sampler.embedding, rbm.visible.tolist(), rbm.hidden.tolist())

###############################################################################
model = lbm
model = rbm

import numpy as np
vv_mask = []
vi,vj = np.triu_indices(model.V,1)
for v,u in zip(vi,vj):
    if S.has_edge(v,u):
        vv_mask.append(1)
    else:
        vv_mask.append(0)
vv_mask = torch.tensor(vv_mask)

hh_mask = []
hi,hj = np.triu_indices(model.H,1)
for v,u in zip(hi,hj):
    if S.has_edge(v+model.V,u+model.V):
        hh_mask.append(1)
    else:
        hh_mask.append(0)
hh_mask = torch.tensor(hh_mask)

import itertools
W_mask = torch.ones_like(model.W)
for v,h in itertools.product(model.visible,model.hidden):
    if not S.has_edge(int(v),int(h)):
        W_mask[h-model.V][v] = 0

# torch.nn.utils.prune.custom_from_mask(model,'vv',vv_mask)
# torch.nn.utils.prune.custom_from_mask(model,'hh',hh_mask)
torch.nn.utils.prune.custom_from_mask(model,'W',W_mask)

rbm.W.numel()
len(lbm.hh.nonzero())
len(lbm.vv.nonzero())
