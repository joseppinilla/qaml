# # Classical BM training on the Bars-And-Stripes Dataset for Reconstruction
# Developed by: Jose Pinilla

# Required packages
import os
import qaml
import torch
SEED = 2
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
HIDDEN_SIZE = 8

# Specify model with dimensions
bm = qaml.nn.BM(VISIBLE_SIZE, HIDDEN_SIZE,'SPIN',lin_range=[-4,4],quad_range=[-1,1])

prune = 'full'
# prune = 0.5
# torch.nn.utils.prune.random_unstructured(bm,'vv',prune)
# torch.nn.utils.prune.random_unstructured(bm,'hh',prune)
# torch.nn.utils.prune.random_unstructured(bm,'W',prune)

# Set up optimizer
optimizer = torch.optim.SGD(bm.parameters(), lr=learning_rate,
                            weight_decay=weight_decay,momentum=momentum)

# Set up training mechanisms
SOLVER_NAME = "Advantage_system4.1"
pos_sampler = qaml.sampler.BatchQASampler(bm,solver=SOLVER_NAME,mask=True)
POS_BATCH = len(pos_sampler.batch_embeddings)
neg_sampler = qaml.sampler.BatchQASampler(bm,solver=SOLVER_NAME)
NEG_BATCH = len(neg_sampler.batch_embeddings)
print(POS_BATCH,NEG_BATCH)

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
fig,axs = plt.subplots(4,4)
for batch_img,batch_label in train_loader:
    for ax,img,label in zip(axs.flat,batch_img,batch_label):
        ax.matshow(img.view(*SHAPE),vmin=0,vmax=1); ax.axis('off')
plt.tight_layout()
plt.savefig("dataset.svg")

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
vs,hs = neg_sampler(num_reads=TEST_READS)
precision, recall, score = train_dataset.score(((vs+1)/2).view(-1,*SHAPE))
p_log.append(precision); r_log.append(recall); score_log.append(score)
print(f"Precision {precision:.2} Recall {recall:.2} Score {score:.2}")

energy = bm.energy(vs,hs)
neg_sampler.sampleset.record.energy = energy
plot_joint_samplesets(neg_sampler.sampleset,(bm.V,bm.H))
plt.savefig(f"./joint/{prune}/plt_0.png")

################################## Model Training ##############################
for t in range(1,5):
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
    vs,hs = neg_sampler(num_reads=TEST_READS)
    precision, recall, score = train_dataset.score(((vs+1)/2).view(-1,*SHAPE))
    p_log.append(precision); r_log.append(recall); score_log.append(score)
    print(f"Precision {precision:.2} Recall {recall:.2} Score {score:.2}")

    energy = bm.energy(vs,hs)
    neg_sampler.sampleset.record.energy = energy
    plot_joint_samplesets(neg_sampler.sampleset,(36,8))
    plt.savefig(f"./joint/{prune}/plt_{t}.png")

directory = f"bm{VISIBLE_SIZE}_{HIDDEN_SIZE}-{TRAIN_READS}bcW/{prune}"
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
for ax,img in zip(axs.flat,vs):
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
    dnx.draw_pegasus_embedding(solver_graph,batches,node_size=10,interaction_edges=[])


plot_batch_embeddings(neg_sampler.batch_embeddings,pos_sampler.to_networkx_graph())
plt.savefig("./embedding_10.svg")
plt.savefig("./embedding_10.png")

plot_batch_embeddings(pos_sampler.batch_embeddings,pos_sampler.to_networkx_graph())
plt.savefig("./embedding_84.svg")
plt.savefig("./embedding_84.png")


import dimod
import minorminer
%matplotlib qt

S_pos = dimod.to_networkx_graph(pos_sampler.to_qubo({v.item():0 for v in bm.visible}))
T_pos = pos_sampler.to_networkx_graph()
miner_pos = minorminer.miner(S_pos,T_pos)

QK_pos = map(miner_pos.quality_key,pos_sampler.batch_embeddings)
_,_,CL_pos = zip(*QK_pos)

fig = plt.figure()
ax = plt.gca()
QK_dict = {}
for QK in CL_pos:
    keys = QK[0::2]
    vals = QK[1::2]
    for k,v in zip(keys,vals):
        QK_dict[k] = QK_dict.get(k,[v]) + [v]
K,V = zip(*sorted(zip(QK_dict.keys(),QK_dict.values())))
bplot = ax.boxplot(V,labels=K,patch_artist=True,showmeans=True,whis=1.0)
ax.set_xlabel('Chain length')
ax.set_ylabel('Count')
ax.yaxis.grid(True)
for line in bplot['medians']:
    line.set_linewidth(2)
    line.set_color('#7f7f7f')
for line in bplot['means']:
    line.set_markeredgecolor('#7f7f7f')
    line.set_markerfacecolor('#7f7f7f')

S_neg = bm.to_networkx_graph()
T_neg = pos_sampler.to_networkx_graph()
miner_neg = minorminer.miner(S_neg,T_neg)

trimmed = [qaml.prune.trim_embedding(T_neg,emb,S_neg) for emb in neg_sampler.batch_embeddings]

trimmed,_ = zip(*trimmed)

# QK_neg = list(map(miner_neg.quality_key,neg_sampler.batch_embeddings))
QK_neg = list(map(miner_neg.quality_key,trimmed))
_,_,CL_neg = zip(*QK_neg)

fig = plt.figure()
ax = plt.gca()
QK_dict = {}
for QK in CL_neg:
    keys = QK[0::2]
    vals = QK[1::2]
    for k,v in zip(keys,vals):
        QK_dict[k] = QK_dict.get(k,[v]) + [v]
K,V = zip(*sorted(zip(QK_dict.keys(),QK_dict.values())))
bplot = ax.boxplot(V,labels=K,patch_artist=True,showmeans=True,whis=1.0)
ax.set_xlabel('Chain length')
ax.set_ylabel('Count')
ax.yaxis.grid(True)
for line in bplot['medians']:
    line.set_linewidth(2)
    line.set_color('#7f7f7f')
for line in bplot['means']:
    line.set_markeredgecolor('#7f7f7f')
    line.set_markerfacecolor('#7f7f7f')



################################################################################
################################################################################
################################################################################
import matplotlib
import numpy as np

np.set_printoptions(precision=2)
torch.set_printoptions(precision=2)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

def plot_joint_samplesets(sampleset,shape):
    # Use shape to find size of the graph
    horizontal,vertical = shape
    maxX, maxY = 2**(horizontal)-1, 2**(vertical)-1

    # Treat the numbers as Gray code to plot similar numbers close to each other
    # e.g. '0110' -> 4, '0111' -> 5, '0101' -> 6, ...
    def bin_to_gray_index(n):
        mask = n = int(n,2)
        while mask != 0:
            mask >>= 1; n ^= mask
        return n

    # Takes a dimod.SampleSet and returns a string of the sample values
    # e.g. sample={0: 0, 1: 0, 2: 1, 3: 0, 4: 1, 5: 1} -> "00101"
    def sample_to_string(datum):
        key_sorted = sorted(datum.sample)
        return ''.join(str(int(1+datum.sample[k])//2) for k in key_sorted)

    # Initilize results
    minE, maxE = None, None
    x = []; y = []; E = []; c = []
    # Reverse iteration plots lower energy samples on top if there is overlap
    for datum in sampleset.data(sorted_by='energy',reverse=True):
        value = sample_to_string(datum)
        x_point = bin_to_gray_index(value[0:horizontal])
        y_point = bin_to_gray_index(value[horizontal:])
        x.append(x_point/maxX); y.append(y_point/maxY)
        c.append(datum.num_occurrences); E.append(datum.energy)
        if (minE is None) or (datum.energy < minE): minE = datum.energy
        if (maxE is None) or (datum.energy > maxE): maxE = datum.energy


    fig = plt.figure(figsize=(9,8)) # (width,height)
    ax = plt.gca()
    # Scatter points
    sct = ax.scatter(x,y,c=E,cmap="jet",alpha=0.5,marker='s')
    ax.set_xlabel('VISIBLE')
    ax.set_ylabel('HIDDEN')
    # Point at lowest value (or values) with an 'X'
    minXY = [(x[ie],y[ie]) for ie,e in enumerate(E) if e==minE]
    ax.scatter(*zip(*minXY),s=100,linewidths=1,c='silver',marker='x')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    # Plot colorbar
    # plt.subplots_adjust(top=1,bottom=0.25,left=.10,right=.90,hspace=0,wspace=0)
    # cax = fig.add_axes([0.5,0.15,0.5,0.02]) # [left,bottom,width,height]
    plt.subplots_adjust(top=.9,bottom=0.1,left=0.1,right=.9,hspace=0,wspace=0)
    cax = fig.add_axes([0.93,0.2,0.01,0.6]) # [left,bottom,width,height]
    cbar = plt.colorbar(sct,orientation='vertical',cax=cax, extend='neither')


# import dimod
# energy = bm.energy(vs,hs)
# neg_sampler.sampleset.record.energy = energy
# plot_joint_samplesets(neg_sampler.sampleset,(36,16))
# plt.savefig(f"./joint/plt_{t}.png")
#
#
# sa_sampler = qaml.sampler.SimulatedAnnealingNetworkSampler(bm)
# vsa,hsa = sa_sampler(num_reads=10000,num_sweeps=1000)
# plot_joint_samplesets(sa_sampler.sampleset,(36,16))
# plt.savefig(f"./joint/W{prune}/SA_plt.png")
