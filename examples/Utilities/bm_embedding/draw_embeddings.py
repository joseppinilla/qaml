import torch
import minorminer

import dwave_networkx as dnx
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 8)

S = torch.load('./source.pt')
T = torch.load('./target.pt')
miner = minorminer.miner(S,T)

heur_embedding = torch.load('./embedding_heur_176.pt')
sys_embedding = torch.load('./embedding_sys_176.pt')
batch_embeddings = torch.load('./embedding_sys_batch.pt')

############################ Systematic Embedding ##############################
_ = plt.figure()
dnx.draw_pegasus_embedding(T,sys_embedding,S,node_size=20)
_,_, sys_L = miner.quality_key(sys_embedding)
plt.title({sys_L[i]:sys_L[i+1] for i in range(0,len(sys_L),2)})

############################# Heuristic Embedding ##############################
_ = plt.figure()
dnx.draw_pegasus_embedding(T,heur_embedding,S,node_size=20)
_,_, heur_L = miner.quality_key(heur_embedding)
plt.title({heur_L[i]:heur_L[i+1] for i in range(0,len(heur_L),2)})

############################### Batch Embeddings ###############################
title = []
_ = plt.figure()
for i,emb in enumerate(batch_embeddings):
    kwargs = {'unused_color':(0.9,0.9,0.9,1//(1+i)),'node_size':20}
    dnx.draw_pegasus_embedding(T,emb,S,**kwargs)
    # _,_, batch_L = miner.quality_key(emb)
    # title.append({batch_L[i]:batch_L[i+1] for i in range(0,len(sys_L),2)})
plt.title(title)
