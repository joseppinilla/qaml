# # Quantum-Assisted BM training on the BAS Dataset for Reconstruction
# This is an example on quantum-assisted training of an BM on the BAS(8,8)
# dataset.
# Developed by: Jose Pinilla

# Required packages
import os
import qaml
import torch

import matplotlib.pyplot as plt
import torchvision.transforms as torch_transforms

################################# Hyperparameters ##############################
EPOCHS = 5
M,N = SHAPE = (8,8)
DATA_SIZE = M*N
TRAIN, TEST = SPLIT = (360,150) #(8,8)
TEST_SAMPLES = 20
# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

################################# Model Definition #############################
VISIBLE_SIZE = DATA_SIZE
HIDDEN_SIZE = 16

bm = qaml.nn.BoltzmannMachine(VISIBLE_SIZE,HIDDEN_SIZE,'SPIN')

# For deterministic weights
SEED = 42
torch.manual_seed(SEED)

# Initialize biases
_ = torch.nn.init.uniform_(bm.b,-4.0,4.0)
_ = torch.nn.init.uniform_(bm.c,-4.0,4.0)
_ = torch.nn.init.uniform_(bm.vv,-1.0,1.0)
_ = torch.nn.init.uniform_(bm.hh,-1.0,1.0)
_ = torch.nn.init.uniform_(bm.W,-1.0,1.0)

# torch.nn.utils.prune.random_unstructured(bm,'W',0.2)
# torch.nn.utils.prune.random_unstructured(bm,'vv',0.2)
# torch.nn.utils.prune.random_unstructured(bm,'hh',0.2)

import numpy as np

rbm = qaml.nn.RestrictedBoltzmannMachine(3,4,'SPIN')
bas_dataset = qaml.datasets.BAS(*SHAPE,transform=qaml.datasets.ToSpinTensor())
set_label,get_label = qaml.datasets._embed_labels(bas_dataset,setter_getter=True)

mask = set_label(torch.ones(1,*SHAPE),0).flatten()

mask



solver_name = "Advantage_system6.1"
sampler = qaml.sampler.QASampler.get_sampler(solver=solver_name)

embeddings = qaml.minor.harvest_cliques(bm,sampler,mask)
list(embeddings)

embedding = qaml.minor.biclique_from_cache(bm,sampler,mask)
embedding

embedding = qaml.minor.clique_from_cache(bm,sampler,mask)
embedding

embedding = qaml.minor.miner_heuristic(bm,sampler,mask=mask)
embedding

# Set up optimizers
optimizer = torch.optim.SGD(bm.parameters(), lr=learning_rate,
                            weight_decay=weight_decay,momentum=momentum)

# Set up training mechanisms
solver_name = "Advantage_system6.1"


qa_sampler = qaml.sampler.QASampler(bm,solver=solver_name,batch_mode=True)

pos_emb = qaml.minor.clique_from_cache()
pos_sampler = qaml.sampler.BatchQASampler(bm,solver=solver_name)

neg_sampler = qaml.sampler.QASampler(bm,solver=solver_name)

recon_emb = qaml.minor.
recon_sampler = qaml.sampler.BatchQASampler(bm,fixed_vars=,solver=solver_name)


import numpy as np
A = np.asarray([1,0,1,0])

idx = A.nonzero()

idx[]

np.nonzero

neg_sampler =

# Loss and autograd
ML = qaml.autograd.MaximumLikelihood

#################################### Input Data ################################
bas_dataset = qaml.datasets.BAS(*SHAPE,transform=qaml.datasets.ToSpinTensor())
set_label,get_label = qaml.datasets._embed_labels(bas_dataset,setter_getter=True)
train_dataset,test_dataset = torch.utils.data.random_split(bas_dataset,[*SPLIT])

BATCH_SIZE = qa_sampler.batch_mode
train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=False)
train_loader = torch.utils.data.DataLoader(train_dataset,BATCH_SIZE,sampler=train_sampler)

test_sampler = torch.utils.data.RandomSampler(test_dataset,False,TEST_SAMPLES)
test_loader = torch.utils.data.DataLoader(test_dataset,sampler=test_sampler)

# Visualize
fig,axs = plt.subplots(4,5)
for ax,(img,label) in zip(axs.flat,test_loader):
    ax.matshow(img.squeeze())
    ax.set_title(int(label))
    ax.axis('off')
plt.tight_layout()

################################## Model Training ##############################
# Set the model to training mode
bm.train()
p_log = []
r_log = []
score_log = []
err_log = []
accuracy_log = []
batch_err_log = []
b_log = [bm.b.detach().clone().numpy()]
c_log = [bm.c.detach().clone().numpy()]
vv_log = [bm.vv.detach().clone().numpy().flatten()]
hh_log = [bm.hh.detach().clone().numpy().flatten()]
W_log = [bm.W.detach().clone().numpy().flatten()]
for t in range(5):
    epoch_error = 0

    for img_batch, labels_batch in train_loader:

        # Positive Phase
        v0, h0 = qa_sampler(img_batch.flatten(1),num_reads=100)
        # Negative Phase
        vk, hk = qa_sampler(num_reads=1000,num_spin_reversal_transforms=4)

        # Reconstruction error from Contrastive Divergence
        err = ML.apply(qa_sampler,(v0,h0),(vk,hk),*bm.parameters())

        # Do not accumulate gradients
        optimizer.zero_grad()

        # Compute gradients
        err.backward()

        # Update parameters
        optimizer.step()

        #Accumulate error for this epoch
        epoch_error  += err.item()
        batch_err_log.append(err.item())

        # Error Log
        b_log.append(bm.b.detach().clone().numpy())
        c_log.append(bm.c.detach().clone().numpy())
        vv_log.append(bm.vv.detach().clone().numpy())
        hh_log.append(bm.hh.detach().clone().numpy())
        W_log.append(bm.W.detach().clone().numpy().flatten())

    err_log.append(epoch_error)
    print(f"Epoch {t} Reconstruction Error = {epoch_error}")

precision, recall, score = bas_dataset.score(((vk+1)/2).view(-1,*SHAPE))
p_log.append(precision); r_log.append(recall); score_log.append(score)
print(f"Precision {precision:.2} Recall {recall:.2} Score {score:.2}")

############################# CLASSIFICATION ##################################

count = 0
for test_data, test_label in test_loader:
    input_data = test_data.flatten(1)
    mask = set_label(torch.ones_like(test_data),0).flatten()
    v_recon,h_recon = qa_sampler.reconstruct(input_data,mask=mask,num_reads=5)
    label_pred = get_label(v_recon.view(-1,*SHAPE))
    if (label_pred.mode(0)[0] == get_label(test_data)).all():
        count+=1
accuracy_log.append(count/TEST_SAMPLES)
print(f"Testing accuracy: {count}/{TEST_SAMPLES} ({count/TEST_SAMPLES:.2f})")

%matplotlib qt
fig,axs = plt.subplots(10,5)
for ax,img in zip(axs.flat,vk):
    ax.matshow(img.view(*SHAPE))
    ax.axis('off')
plt.tight_layout()


import dwave.preprocessing
import torch
import numpy as np





# Precision graph
fig, ax = plt.subplots()
ax.plot(p_log)
plt.ylabel("Precision")
plt.xlabel("Epoch")

# Recall graph
fig, ax = plt.subplots()
ax.plot(r_log)
plt.ylabel("Recall")
plt.xlabel("Epoch")

# Score graph
fig, ax = plt.subplots()
ax.plot(score_log)
plt.ylabel("Score")
plt.xlabel("Epoch")

# L1 error graph
fig, ax = plt.subplots()
ax.plot(err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")


# Visible bias graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',DATA_SIZE).colors))
lc_v = ax.plot(b_log)
plt.legend(iter(lc_v),[f'b{i}' for i in range(DATA_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Visible Biases")
plt.xlabel("Epoch")

# Hidden bias graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',HIDDEN_SIZE).colors))
lc_h = plt.plot(c_log)
plt.legend(lc_h,[f'c{i}' for i in range(HIDDEN_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Hidden Biases")
plt.xlabel("Epoch")

# Visible-Visible bias graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',len(bm.vv)).colors))
lc_vv = ax.plot(vv_log)
plt.legend(iter(lc_vv),[f'b{i}' for i in range(len(bm.vv))],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Visible-Visible Biases")
plt.xlabel("Epoch")

# Hidden-Hidden bias graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',len(bm.hh)).colors))
lc_hh = ax.plot(vv_log)
plt.legend(iter(lc_hh),[f'b{i}' for i in range(len(bm.hh))],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Hidden-Hidden Biases")
plt.xlabel("Epoch")

# Weights graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',HIDDEN_SIZE*DATA_SIZE).colors))
lc_w = plt.plot(W_log)
plt.ylabel("Weights")
plt.xlabel("Epoch")
