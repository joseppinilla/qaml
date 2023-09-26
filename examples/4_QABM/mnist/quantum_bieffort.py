# # Quantum-Assisted BM training on the BAS Dataset for Reconstruction
# This is an example on quantum-assisted training of an BM on the BAS(8,8)
# dataset.
# Developed by: Jose Pinilla

# Required packages
import os
import qaml
import torch

import matplotlib.pyplot as plt
import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

from torch.nn.functional import max_pool2d

################################# Hyperparameters ##############################
EPOCHS = 75
M,N = SHAPE = (13,13)
DATA_SIZE = M*N

# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

batch_mode = True
solver_name = "Advantage_system6.1"

SEED = 42
torch.manual_seed(SEED)

################################# Model Definition #############################
VISIBLE_SIZE = DATA_SIZE
HIDDEN_SIZE = 8

bm = qaml.nn.BoltzmannMachine(VISIBLE_SIZE,1,'SPIN')

# For deterministic weights
SEED = 42
_ = torch.manual_seed(SEED)

# Initialize biases
_ = torch.nn.init.uniform_(bm.b,-4.0,4.0)
_ = torch.nn.init.uniform_(bm.c,-4.0,4.0)
_ = torch.nn.init.uniform_(bm.vv,-1.0,1.0)
_ = torch.nn.init.uniform_(bm.hh,-1.0,1.0)
_ = torch.nn.init.uniform_(bm.W,-1.0,1.0)

# Set up training mechanisms
empty_sampler = qaml.sampler.QASampler(bm,solver=solver_name)
pruned,embedding = qaml.minor.bipartite_effort(bm,empty_sampler,HIDDEN_SIZE)
bm = qaml.nn.BoltzmannMachine(VISIBLE_SIZE,len(embedding)-VISIBLE_SIZE,'SPIN')

# import minorminer
# miner = minorminer.miner(pruned,neg_sampler.to_networkx_graph())
# len([q for chain in embedding.values() for q in chain])
# embedding = miner.improve_embeddings([embedding])[0]
# len([q for chain in embedding.values() for q in chain])
# %matplotlib qt
# import dwave_networkx as dnx
# fig = plt.figure(figsize=(16,16))
# dnx.draw_pegasus_embedding(neg_sampler.networkx_graph,embedding,node_size=30)

import numpy as np
vv_mask = []
vi,vj = np.triu_indices(bm.V,1)
for v,u in zip(vi,vj):
    if pruned.has_edge(v,u):
        vv_mask.append(1)
    else:
        vv_mask.append(0)
vv_mask = torch.tensor(vv_mask)

hh_mask = []
hi,hj = np.triu_indices(bm.H,1)
for v,u in zip(hi,hj):
    if pruned.has_edge(v+bm.V,u+bm.V):
        hh_mask.append(1)
    else:
        hh_mask.append(0)
hh_mask = torch.tensor(hh_mask)

import itertools
W_mask = torch.ones_like(bm.W)
for v,h in itertools.product(bm.visible,bm.hidden):
    if not pruned.has_edge(int(v),int(h)):
        W_mask[h-bm.V][v] = 0

# torch.nn.utils.prune.custom_from_mask(bm,'vv',vv_mask)
# torch.nn.utils.prune.custom_from_mask(bm,'hh',hh_mask)
# torch.nn.utils.prune.custom_from_mask(bm,'W',W_mask)

qaml.prune.custom_from_mask(bm,'vv',vv_mask)
qaml.prune.custom_from_mask(bm,'hh',hh_mask)
qaml.prune.custom_from_mask(bm,'W',W_mask)



neg_sampler = qaml.sampler.QASampler(bm,solver=solver_name,embedding=embedding)
# neg_sampler = qaml.sampler.QASampler(bm,solver=solver_name)
pos_sampler = qaml.sampler.QASampler(bm,solver=solver_name,embedding={},batch_mode=True,mask=True)


# Set up optimizers
optimizer = torch.optim.SGD(bm.parameters(), lr=learning_rate,
                            weight_decay=weight_decay,momentum=momentum)

# Loss and autograd
ML = qaml.autograd.MaximumLikelihood

BATCH_SIZE = pos_sampler.batch_mode
print(BATCH_SIZE)
#################################### Input Data ################################
mnist_train = torch_datasets.MNIST(root='./data/', train=True, download=True,
                                   transform=qaml.datasets.ToSpinTensor())

mnist_train.data = torch_transforms.functional.crop(mnist_train.data.float(),1,1,26,26).byte()
mnist_train.data = max_pool2d(mnist_train.data.float(),(2,2)).byte()
set_label, get_label = qaml.datasets._embed_labels(mnist_train,axis=1,scale=255,
                                                   encoding='one_hot',setter_getter=True)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE)
print(len(train_loader))
print(len(mnist_train.data))
# Reconstruction sampler from label
mask = set_label(torch.ones(1,*SHAPE),0).flatten()
recon_sampler = qaml.sampler.QASampler(bm,solver=solver_name,embedding={},batch_mode=True,mask=mask)
RECON_SIZE = recon_sampler.batch_mode
print(RECON_SIZE)
mnist_test = torch_datasets.MNIST(root='./data/', train=False, download=True,
                                  transform=qaml.datasets.ToSpinTensor())
mnist_test.data = torch_transforms.functional.crop(mnist_test.data.float(),1,1,26,26).byte()
mnist_test.data = max_pool2d(mnist_test.data.float(),(2,2)).byte()
qaml.datasets._embed_labels(mnist_test,encoding='one_hot',
                                                  scale=255)
test_loader = torch.utils.data.DataLoader(mnist_test,RECON_SIZE)
print(len(mnist_test.data))
print(len(test_loader))
################################## Model Training ##############################
# Set the model to training mode
bm.train()
err_log = []
accuracy_log = []
batch_err_log = []
b_log = [bm.b.detach().clone().numpy()]
c_log = [bm.c.detach().clone().numpy()]
vv_log = [(bm.vv_orig*bm.vv_mask).detach().clone().numpy().flatten()]
hh_log = [(bm.hh_orig*bm.hh_mask).detach().clone().numpy().flatten()]
W_log =  [(bm.W_orig*bm.W_mask).detach().clone().numpy().flatten()]


for t in range(1):
    epoch_error = 0

    for i, (img_batch, labels_batch) in enumerate(train_loader):
        print(f"Batch {i}")
        # Positive Phase
        v0, h0 = pos_sampler(img_batch.flatten(1),num_reads=100)

        # Negative Phase
        vk, hk = neg_sampler(num_reads=1000,num_spin_reversal_transforms=4)

        # Reconstruction error from Contrastive Divergence
        err = ML.apply(neg_sampler,(v0,h0),(vk,hk),*bm.parameters())

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
        vv_log.append((bm.vv_orig*bm.vv_mask).detach().clone().numpy().flatten())
        hh_log.append((bm.hh_orig*bm.hh_mask).detach().clone().numpy().flatten())
        W_log.append((bm.W_orig*bm.W_mask).detach().clone().numpy().flatten())

    # Error report
    err_log.append(epoch_error)
    print(f"Epoch {t} Reconstruction Error = {epoch_error}")

    ############################# CLASSIFICATION ##################################
    count = 0
    for test_data, test_labels in test_loader:
        input_data = test_data.flatten(1)
        mask = set_label(torch.ones(1,*SHAPE),0).flatten()
        v_batch,h_batch = recon_sampler.reconstruct(input_data,mask=mask,num_reads=5)
        for v_recon,v_test in zip(v_batch,test_data):
            label_pred = get_label(v_recon.view(1,*SHAPE))
            label_test = get_label(v_test.view(1,*SHAPE))

            if (torch.argmax(label_pred) == torch.argmax(label_test)):
                count+=1

    TEST_SAMPLES = len(mnist_test.data)
    accuracy_log.append(count/TEST_SAMPLES)
    print(f"Testing accuracy: {count}/{TEST_SAMPLES} ({count/TEST_SAMPLES:.2f})")


epoch_error*210

directory = f"{HIDDEN_SIZE}_batch{BATCH_SIZE}"
directory = directory.replace('.','')

if not os.path.exists(directory):
        os.makedirs(directory)
if not os.path.exists(f'{directory}/{SEED}'):
        os.makedirs(f'{directory}/{SEED}')


accuracy = torch.load(f"./{directory}/{SEED}/accuracy.pt")
err = torch.load(f"./{directory}/{SEED}/err.pt")

err_log = [epoch_error*210]+err
err_log
 = [accuracy_log]+accuracy
accuracy_log

# torch.save(b_log,f"./{directory}/{SEED}/b.pt")
# torch.save(c_log,f"./{directory}/{SEED}/c.pt")
# torch.save(W_log,f"./{directory}/{SEED}/W.pt")
# torch.save(hh_log,f"./{directory}/{SEED}/hh.pt")
# torch.save(vv_log,f"./{directory}/{SEED}/vv.pt")
# torch.save(err_log,f"./{directory}/{SEED}/err.pt")
# torch.save(batch_err_log,f"./{directory}/{SEED}/batch_err.pt")
# torch.save(accuracy_log,f"./{directory}/{SEED}/accuracy.pt")


len(bm.hh.nonzero())

import dwave_networkx as dnx

fig, ax = plt.subplots(figsize=(16,16))
dnx.draw_pegasus_embedding(neg_sampler.networkx_graph,neg_sampler.embedding,node_size=10)

len([q in chain for chain in neg_sampler.embedding.values() for q in chain])

len([q in chain for embedding in pos_sampler.batch_embeddings for chain in embedding.values() for q in chain])
len(neg_sampler.networkx_graph)

%matplotlib qt
fig,axs = plt.subplots(10,5)
for ax,img in zip(axs.flat,mnist_test.data):
    ax.matshow(img.view(*SHAPE))
    ax.axis('off')
plt.tight_layout()

# Batch Err
fig, ax = plt.subplots()
ax.plot(batch_err_log)
plt.ylabel("Batch Err")
plt.xlabel("Batch")


# Accuracy graph
fig, ax = plt.subplots()
ax.plot(accuracy_log)
plt.ylabel("Test Accuracy")
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
# plt.legend(iter(lc_v),[f'b{i}' for i in range(DATA_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Visible Biases")
plt.xlabel("Epoch")

# Hidden bias graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',HIDDEN_SIZE).colors))
lc_h = plt.plot(c_log)
# plt.legend(lc_h,[f'c{i}' for i in range(HIDDEN_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Hidden Biases")
plt.xlabel("Epoch")

# Visible-Visible bias graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',len(bm.vv)).colors))
lc_vv = ax.plot(vv_log)
# plt.legend(iter(lc_vv),[f'b{i}' for i in range(len(bm.vv))],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Visible-Visible Biases")
plt.xlabel("Epoch")

# Hidden-Hidden bias graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',len(bm.hh)).colors))
lc_hh = ax.plot(vv_log)
# plt.legend(iter(lc_hh),[f'b{i}' for i in range(len(bm.hh))],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Hidden-Hidden Biases")
plt.xlabel("Epoch")

# Weights graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',HIDDEN_SIZE*DATA_SIZE).colors))
lc_w = plt.plot(W_log)
plt.ylabel("Weights")
plt.xlabel("Epoch")
