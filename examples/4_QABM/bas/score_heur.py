# # Classical BM training on the Bars-And-Stripes Dataset for Reconstruction
# Developed by: Jose Pinilla

# Required packages
import os
import qaml
import torch
SEED = 42
torch.manual_seed(SEED) # For deterministic weights

import matplotlib.pyplot as plt
import torchvision.transforms as torch_transforms

################################# Hyperparameters ##############################
EPOCHS = 50
M,N = SHAPE = (6,6)
DATA_SIZE = N*M
SUBCLASSES = [1,2]

# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

TRAIN_READS = 100

################################# Model Definition #############################
VISIBLE_SIZE = DATA_SIZE
HIDDEN_SIZE = 16

# Specify model with dimensions
bm = qaml.nn.BM(VISIBLE_SIZE, HIDDEN_SIZE, 'SPIN')

# Set up optimizer
optimizer = torch.optim.SGD(bm.parameters(), lr=learning_rate,
                            weight_decay=weight_decay,momentum=momentum)

# Set up training mechanisms
SOLVER_NAME = "Advantage_system4.1"
pos_sampler = qaml.sampler.BatchQASampler(bm,solver=SOLVER_NAME,mask=True,harvest_method=qaml.minor.harvest_heuristic)
POS_BATCH = len(pos_sampler.batch_embeddings)
neg_sampler = qaml.sampler.BatchQASampler(bm,solver=SOLVER_NAME,harvest_method=qaml.minor.harvest_heuristic)
NEG_BATCH = len(neg_sampler.batch_embeddings)

ML = qaml.autograd.MaximumLikelihood

#################################### Input Data ################################
train_dataset = qaml.datasets.BAS(*SHAPE,transform=qaml.datasets.ToSpinTensor())
set_label,get_label = qaml.datasets._embed_labels(train_dataset,
                                                  encoding='binary',
                                                  setter_getter=True)
qaml.datasets._subset_classes(train_dataset,SUBCLASSES)
train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=False)
train_loader = torch.utils.data.DataLoader(train_dataset,POS_BATCH,sampler=train_sampler)
TEST_READS = len(train_dataset)
# PLot all data
# fig,axs = plt.subplots(4,4)
# for batch_img,batch_label in train_loader:
#     for ax,img,label in zip(axs.flat,batch_img,batch_label):
#         ax.matshow(img.view(*SHAPE),vmin=0,vmax=1); ax.axis('off')
# plt.tight_layout()
# plt.savefig("dataset.svg")


################################### load Model #################################
# b_log = torch.load(f'{directory}/{SEED}/b_log.pt')
# c_log = torch.load(f'{directory}/{SEED}/c_log.pt')
# vv_log = torch.load(f'{directory}/{SEED}/vv_log.pt')
# hh_log = torch.load(f'{directory}/{SEED}/hh_log.pt')
# W_log = torch.load(f'{directory}/{SEED}/W_log.pt')
# err_log = torch.load(f'{directory}/{SEED}/err_log.pt')
# p_log = torch.load(f'{directory}/{SEED}/p_log.pt')
# r_log = torch.load(f'{directory}/{SEED}/r_log.pt')
# score_log = torch.load(f'{directory}/{SEED}/score_log.pt')
# epoch_err_log = torch.load(f'{directory}/{SEED}/epoch_err_log.pt')

################################## Pre-Training ################################
# Set the model to training mode
bm.train()
p_log = []
r_log = []
err_log = []
score_log = []
epoch_err_log = []
b_log = [bm.b.detach().clone().numpy()]
c_log = [bm.c.detach().clone().numpy()]
W_log = [bm.W.detach().clone().numpy().flatten()]
vv_log = [bm.vv.detach().clone().numpy().flatten()]
hh_log = [bm.hh.detach().clone().numpy().flatten()]

# BAS score
vk,_ = neg_sampler(num_reads=TEST_READS)
precision, recall, score = train_dataset.score(((vk+1)/2).view(-1,*SHAPE))
p_log.append(precision); r_log.append(recall); score_log.append(score)
print(f"Precision {precision:.2} Recall {recall:.2} Score {score:.2}")

################################## Model Training ##############################
for t in range(10):
    kl_div = torch.Tensor([0.])
    epoch_error = torch.Tensor([0.])
    for img_batch,labels_batch in train_loader:
        input_data = img_batch.view(1,-1)

        # Positive Phase
        v0, h0 = pos_sampler(input_data.detach(),num_reads=TRAIN_READS)
        # Negative Phase
        vk, hk = neg_sampler(num_reads=TRAIN_READS)

        # Reconstruction error from Contrastive Divergence
        err = ML.apply(neg_sampler,(v0,h0),(vk,hk), *bm.parameters())

        # Do not accumulate gradients
        optimizer.zero_grad()

        # Compute gradients
        err.backward()

        # Update parameters
        optimizer.step()

        #Accumulate error for this epoch
        epoch_error  += err
        err_log.append(err.item())

    # Error Log
    b_log.append(bm.b.detach().clone().numpy())
    c_log.append(bm.c.detach().clone().numpy())
    vv_log.append(bm.vv.detach().clone().numpy())
    hh_log.append(bm.hh.detach().clone().numpy())
    W_log.append(bm.W.detach().clone().numpy().flatten())
    # Error Log
    epoch_err_log.append(epoch_error.item())
    print(f"Epoch {t} Reconstruction Error = {epoch_error.item()}")
    # BAS score
    vk,_ = neg_sampler(num_reads=TEST_READS)
    precision, recall, score = train_dataset.score(((vk+1)/2).view(-1,*SHAPE))
    p_log.append(precision); r_log.append(recall); score_log.append(score)
    print(f"Precision {precision:.2} Recall {recall:.2} Score {score:.2}")

directory = f"bm{VISIBLE_SIZE}_{HIDDEN_SIZE}-{TRAIN_READS}/heur"
os.makedirs(f'{directory}/{SEED}',exist_ok=True)

torch.save(b_log,f'{directory}/{SEED}/b_log.pt')
torch.save(c_log,f'{directory}/{SEED}/c_log.pt')
torch.save(vv_log,f'{directory}/{SEED}/vv_log.pt')
torch.save(hh_log,f'{directory}/{SEED}/hh_log.pt')
torch.save(W_log,f'{directory}/{SEED}/W_log.pt')
torch.save(err_log,f'{directory}/{SEED}/err_log.pt')
torch.save(p_log,f'{directory}/{SEED}/p_log.pt')
torch.save(r_log,f'{directory}/{SEED}/r_log.pt')
torch.save(score_log,f'{directory}/{SEED}/score_log.pt')
torch.save(epoch_err_log,f'{directory}/{SEED}/epoch_err_log.pt')

# Samples
fig,axs = plt.subplots(4,4)
for ax,img in zip(axs.flat,vk):
    ax.matshow(img.view(*SHAPE),vmin=0,vmax=1); ax.axis('off')
plt.tight_layout()
plt.savefig(f"{directory}/{SEED}/sample_vk.svg")

# Precision graph
fig, ax = plt.subplots()
ax.plot(p_log)
plt.ylabel("Precision")
plt.xlabel("Epoch")
plt.savefig(f"{directory}/{SEED}/precision.svg")

# Recall graph
fig, ax = plt.subplots()
ax.plot(r_log)
plt.ylabel("Recall")
plt.xlabel("Epoch")
plt.savefig(f"{directory}/{SEED}/recall.svg")

# Score graph
fig, ax = plt.subplots()
ax.plot(score_log)
plt.ylabel("Score")
plt.xlabel("Epoch")
plt.savefig(f"{directory}/{SEED}/score.svg")

# Iteration Error
fig, ax = plt.subplots()
ax.plot(err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")
plt.savefig(f"{directory}/{SEED}/err.svg")

# Epoch Error
fig, ax = plt.subplots()
ax.plot(epoch_err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")
plt.savefig(f"{directory}/{SEED}/epoch_err.svg")

# PLot batch_embeddings
import matplotlib.cm
import dwave_networkx as dnx
from dwave_networkx.drawing.distinguishable_colors import distinguishable_color_map

def plot_batch_embeddings(batch_embeddings,solver_graph):

    batches = {v:list(x for chain in embedding.values() for x in chain) for v,embedding in enumerate(batch_embeddings)}
    # n = len(batches)
    # color = distinguishable_color_map(int(n))
    # chain_color = {v: color(i/n) for i, v in enumerate(batches)}
    _ = plt.figure(figsize=(16,16))
    dnx.draw_pegasus_embedding(solver_graph,batches,node_size=10)


plot_batch_embeddings(pos_sampler.batch_embeddings,pos_sampler.to_networkx_graph())
plt.savefig("./embedding_heur_pos_98.svg")

plot_batch_embeddings(neg_sampler.batch_embeddings,neg_sampler.to_networkx_graph())
plt.savefig("./embedding_heur_neg_09.svg")



import dimod
import minorminer

S_pos = dimod.to_networkx_graph(pos_sampler.to_qubo({v.item():0 for v in bm.visible}))
T_pos = pos_sampler.to_networkx_graph()
miner_pos = minorminer.miner(S_pos,T_pos)
QK_pos = map(miner_pos.quality_key,pos_sampler.batch_embeddings)
_,_,CL_pos = zip(*QK_pos)


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
%matplotlib qt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.computed_zorder=True
offset = 0
for QK in CL_pos:
    ax.bar(QK[0::2], QK[1::2],alpha=0.8,ec='k',zs=offset,zdir='y')
    offset+=1


for QK in CL_pos:
    plt.bar(QK[0::2], QK[1::2],alpha=0.5)





S_neg = bm.to_networkx_graph()
T_neg = pos_sampler.to_networkx_graph()
miner_neg = minorminer.miner(S_neg,T_neg)
QK_neg = list(map(miner_neg.quality_key,neg_sampler.batch_embeddings))
_,_,CL_neg = zip(*QK_neg)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
%matplotlib qt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
offset = 0
for QK in CL_neg:
    ax.bar(QK[0::2], QK[1::2],alpha=0.8,ec='k',zs=offset,zdir='y')
    offset+=1
