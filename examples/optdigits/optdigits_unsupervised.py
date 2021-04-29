import qaml
import torch
import itertools

import matplotlib.pyplot as plt

import torchvision.transforms as torch_transforms
################################# Hyperparameters ##############################
SHAPE = (8,8)
EPOCHS = 5
BATCH_SIZE = 64
# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

#################################### Input Data ################################
train_dataset = qaml.datasets.OptDigits(root='./data/', train=True,
                                     transform=torch_transforms.ToTensor(),
                                     download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           shuffle=True)

test_dataset = qaml.datasets.OptDigits(root='./data/', train=False,
                                    transform=torch_transforms.ToTensor(),
                                    download=True)
test_loader = torch.utils.data.DataLoader(test_dataset)

DATASET_SIZE = len(train_dataset)

################################# Model Definition #############################
DATA_SIZE = len(train_dataset.data[0].flatten())
HIDDEN_SIZE = 64

# Specify model with dimensions
rbm = qaml.nn.RBM(DATA_SIZE,HIDDEN_SIZE,beta=2.0)

# Set up optimizer
optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate,
                                              weight_decay=weight_decay,
                                              momentum=momentum)

# Set up training mechanisms
# ex_sampler = qaml.sampler.ExactNetworkSampler(rbm)
gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm)
# sa_sampler = qaml.sampler.SimulatedAnnealingNetworkSampler(rbm)
# qa_sampler = qaml.sampler.QuantumAnnealingNetworkSampler(rbm, solver="Advantage_system1.1")
# CD = qaml.autograd.ConstrastiveDivergence()
CD = qaml.autograd.SampleBasedConstrastiveDivergence()
################################## Model Training ##############################
# Set the model to training mode
rbm.train()
err_log = []
bv_log = [rbm.bv.detach().clone().numpy()]
bh_log = [rbm.bh.detach().clone().numpy()]
W_log = [rbm.W.detach().clone().numpy().flatten()]
for t in range(EPOCHS):
    epoch_error = torch.Tensor([0.])
    for img_batch, labels_batch in train_loader:
        input_data = img_batch.flatten(1)/16

        # Positive Phase
        v0, prob_h0 = input_data, rbm(input_data)
        # Negative Phase

        # Anneal schedule
        # forward_20us = [(0.0,0.0),(20.0,1.0)]
        # paused_anneal = [(0.0,0.0),(0.5,0.5),(100.5,0.5),(101.0,1.0)]
        # reverse_anneal = [[0.0, 1.0], [2.75, 0.35], [82.75, 0.35], [100, 1.0]]

        # INITIAL STATE
        # init_v = {v:val.item() for v,val in enumerate(v0.mean(dim=0).bernoulli())}
        # init_h = {h:val.item() for h,val in enumerate(prob_h0.mean(dim=0).bernoulli())}
        # initial_state = {**init_v,**{rbm.V+j:h for j,h in init_h.items()}}
        # emb_init = {k:val for i,val in initial_state.items() for k in qa_sampler.embedding[i] }

        # vk, prob_hk = qa_sampler(num_reads=500,anneal_schedule=reverse_anneal,initial_state=emb_init)
        # vk, prob_hk = qa_sampler(num_reads=BATCH_SIZE,anneal_schedule=forward_20us)
        # vk, prob_hk = qa_sampler(num_reads=100)
        # vk, prob_hk = sa_sampler(num_reads=BATCH_SIZE,num_sweeps=500)
        vk, prob_hk = gibbs_sampler(v0,k=1)
        # vk, prob_hk = ex_sampler(num_reads=100)

        # Reconstruction error from Contrastive Divergence
        err = CD.apply((v0,prob_h0), (vk,prob_hk), *rbm.parameters())

        # Do not accumulated gradients
        optimizer.zero_grad()
        # Compute gradients. Save compute graph at last epoch
        err.backward(retain_graph=(t == EPOCHS-1))

        # Update parameters
        optimizer.step()
        epoch_error  += err
    # Error Log
    bv_log.append(rbm.bv.detach().clone().numpy())
    bh_log.append(rbm.bh.detach().clone().numpy())
    W_log.append(rbm.W.detach().clone().numpy().flatten())
    err_log.append(epoch_error)
    print(f"Epoch {t} Reconstruction Error = {epoch_error.item()}")

# Set the model to evaluation mode
# rbm.eval()
torch.save(rbm,"qbas_unsupervised.pt")
rbm =  torch.load("qbas_unsupervised.pt")
# Error graph
plt.plot(err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")
plt.savefig("err_log.png")

# Visible bias graph
ax = plt.gca()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',DATA_SIZE).colors))
lc_v = ax.plot(bv_log)
plt.legend(iter(lc_v),[f'bv{i}' for i in range(DATA_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Visible Biases")
plt.xlabel("Epoch")
plt.savefig("visible_bias_log.png")

# Hidden bias graph
ax = plt.gca()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',HIDDEN_SIZE).colors))
lc_h = plt.plot(bh_log)
plt.legend(lc_h,[f'bh{i}' for i in range(HIDDEN_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Hidden Biases")
plt.xlabel("Epoch")
plt.savefig("hidden_bias_log.png")

# Weights graph
ax = plt.gca()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',HIDDEN_SIZE*DATA_SIZE).colors))
lc_w = plt.plot(W_log)
plt.legend(lc_w,[f'w{i},{j}' for j in range(HIDDEN_SIZE) for i in range(DATA_SIZE)],ncol=4,bbox_to_anchor=(1,1))
plt.ylabel("Weights")
plt.xlabel("Epoch")
plt.savefig("weights_log.png")

################################## ENERGY ######################################

data_energies = []
for img,_ in train_dataset:
    data_energies.append(rbm.energy(img.view(rbm.V).bernoulli(),rbm(img.view(rbm.V)).bernoulli()).item())
    # data_energies.append(rbm.free_energy(img.view(rbm.V)).item())

rand_energies = []
for _ in range(len(train_dataset)*10):
    rand_energies.append(rbm.energy(torch.rand(rbm.V).bernoulli(),torch.rand(rbm.H).bernoulli()).item())
    # rand_energies.append(rbm.free_energy(torch.rand(rbm.V).bernoulli()).item())

model_energies = []
for v in itertools.product([0, 1], repeat=DATA_SIZE):
    for h in itertools.product([0, 1], repeat=HIDDEN_SIZE):
        model_energies.append(rbm.energy(torch.tensor(v,dtype=torch.float),torch.tensor(h,dtype=torch.float)).item())
    # model_energies.append(rbm.free_energy(torch.tensor(v,dtype=torch.float)).item())

gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm)
gibbs_energies = []
for _ in range(100):
    for img,_ in train_dataset:
        prob_v,prob_h = gibbs_sampler(img.float().view(rbm.V),k=1)
        gibbs_energies.append(rbm.energy(prob_v,prob_h).item())
        # gibbs_energies.append(rbm.free_energy(prob_v).item())

sa_energies = []
sa_sampler = qaml.sampler.SimulatedAnnealingNetworkSampler(rbm)
sa_sampleset = sa_sampler(num_reads=1000, num_sweeps=1000)
for s_v,s_h in zip(*sa_sampleset):
    sa_energies.append(rbm.energy(s_v.detach(),s_h.detach()).item())
    # sa_energies.append(rbm.free_energy(s_v.detach()).item())

ex_energies = []
ex_sampler = qaml.sampler.ExactNetworkSampler(rbm)
ex_sampleset = ex_sampler(num_reads=1000,beta=1.0)
for s_v,s_h in zip(*ex_sampleset):
    ex_energies.append(rbm.energy(s_v.detach(),s_h.detach()).item())
    # ex_energies.append(rbm.free_energy(s_v.detach()).item())

qa_energies = []
qa_sampler = qaml.sampler.QuantumAnnealingNetworkSampler(rbm,solver="Advantage_system1.1")
qa_sampleset = qa_sampler(num_reads=1000)
for s_v,s_h in zip(*qa_sampleset):
    qa_energies.append(rbm.energy(s_v.detach(),s_h.detach()).item())
    # qa_energies.append(rbm.free_energy(s_v.detach()).item())

import matplotlib
import numpy as np
%matplotlib qt
hist_kwargs = {'ec':'k','lw':2.0,'alpha':0.5,'histtype':'stepfilled','bins':100}
matplotlib.rcParams.update({'font.size': 22})
weights = lambda data: np.ones_like(data)/len(data)

plt.hist(rand_energies,weights=weights(rand_energies),label="Random",color='r',**hist_kwargs)
plt.hist(data_energies,weights=weights(data_energies), label="Data", color='b', **hist_kwargs)
# plt.hist(ex_energies,weights=weights(ex_energies),label="Exact", color='w',**hist_kwargs)
# plt.hist(model_energies, weights=weights(model_energies), label="Model", color='r', **hist_kwargs)
plt.hist(gibbs_energies,weights=weights(gibbs_energies),label="Gibbs-1",color='g',**hist_kwargs)
plt.hist(qa_energies,weights=weights(qa_energies),label="QA",color='orange', **hist_kwargs)
# plt.hist(sa_energies,weights=weights(sa_energies),label="SA",color='c',**hist_kwargs)
plt.legend(loc='upper right')
plt.ylim(0.0,0.05)
plt.ylabel("Count/Total")
plt.xlabel("Energy")

plt.savefig("energies.pdf")


################################## VISUALIZE ###################################
plt.matshow(rbm.bv.detach().view(*SHAPE), cmap='viridis', vmin=-1, vmax=1)
plt.colorbar()

fig,axs = plt.subplots(HIDDEN_SIZE//4,4)
for i,ax in enumerate(axs.flat):
    weight_matrix = rbm.W[i].detach().view(*SHAPE)
    ms = ax.matshow(weight_matrix, cmap='viridis', vmin=-1, vmax=1)
    ax.axis('off')
fig.subplots_adjust(wspace=0.0, hspace=0.0)
cbar = fig.colorbar(ms, ax=axs.ravel().tolist(), shrink=0.95)
plt.savefig("weights.png")

################################# qBAS Score ###################################
N = 1000
sample_v,sample_h = qa_sampler(num_reads=N)
plt.matshow(torch.mean(sample_v,dim=0).view(*SHAPE).detach())
train_dataset.score(sample_v.view(N,*SHAPE),score_only=False)
