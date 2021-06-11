import qaml
import torch
import itertools

import matplotlib.pyplot as plt

import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

################################# Hyperparameters ##############################
SHAPE = (7,6)
EPOCHS = 50
SAMPLES = 1000
BATCH_SIZE = 500
# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

#################################### Input Data ################################
train_dataset = qaml.datasets.BAS(*SHAPE,transform=torch_transforms.ToTensor())
train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=True,
                                               num_samples=SAMPLES)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           sampler=train_sampler,
                                           batch_size=BATCH_SIZE)

DATASET_SIZE = SAMPLES*len(train_dataset)

################################# Model Definition #############################
DATA_SIZE = len(train_dataset.data[0].flatten())
HIDDEN_SIZE = 8

# Specify model with dimensions
beta = torch.nn.Parameter(torch.tensor(2.0), requires_grad=True)
rbm = qaml.nn.RBM(DATA_SIZE,HIDDEN_SIZE,beta=beta.item())

# Set up optimizer
optimizer = torch.optim.SGD(rbm.parameters(),
                            lr=learning_rate,
                            weight_decay=weight_decay,
                            momentum=momentum)

beta_optimizer = torch.optim.SGD([beta],lr=0.025)

# Set up training mechanisms
qa_sampler = qaml.sampler.QuantumAnnealingNetworkSampler(rbm, solver="Advantage_system1.1")
CD = qaml.autograd.SampleBasedConstrastiveDivergence()
betaGrad = qaml.autograd.AdaptiveBeta()
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
        input_data = img_batch.flatten(1)

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
        # vk, prob_hk = gibbs_sampler(v0.detach(), k=5)
        vk, prob_hk = qa_sampler(num_reads=BATCH_SIZE)

        # Reconstruction error from Contrastive Divergence
        err = CD.apply((v0,prob_h0), (vk,prob_hk), *rbm.parameters())

        # Do not accumulated gradients
        optimizer.zero_grad()
        # Compute gradients. Save compute graph at last epoch
        err.backward(retain_graph=(t == EPOCHS-1))

        # Update parameters
        optimizer.step()
        epoch_error  += err

        # Beta update
        err_beta = betaGrad.apply(rbm.energy(v0,prob_h0),rbm.energy(vk,prob_hk),beta)
        err_beta = betaGrad.apply(rbm.energy(v0,prob_h0),rbm.energy(vk,prob_hk),beta)
        beta_optimizer.zero_grad()
        err_beta.backward()
        beta_optimizer.step()
        rbm.beta = beta.item()


    # Error Log
    bv_log.append(rbm.bv.detach().clone().numpy())
    bh_log.append(rbm.bh.detach().clone().numpy())
    W_log.append(rbm.W.detach().clone().numpy().flatten())
    err_log.append(epoch_error.item())
    print(f"Epoch {t} Reconstruction Error = {epoch_error.item()}")
    print(f"Beta = {rbm.beta}")

################################# qBAS Score ###################################
N = 1000 # CLASSICAL
gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm)
prob_v,prob_h = gibbs_sampler(torch.rand(N,DATA_SIZE),k=2)
plt.matshow(prob_v[42].view(*SHAPE)); plt.colorbar()
p,r,score = train_dataset.score(prob_v.view(N,*SHAPE).bernoulli())
print(f"qBAS : Precision = {p:.02} Recall = {r:.02} Score = {score:.02}")

############################## RECONSTRUCTION ##################################
k = 100
count = 0
mask = torch_transforms.F.erase(torch.ones(SHAPE),1,1,5,4,0).flatten()
for img, label in train_dataset:

    clamped = mask*img.flatten(1)
    prob_hk = rbm(clamped)
    prob_vk = rbm.generate(prob_hk).detach()
    for _ in range(k):
        masked = clamped + (1-mask)*prob_vk.data
        prob_hk.data = rbm(masked)
        prob_vk.data = rbm.generate(prob_hk)
    recon = (clamped + (1-mask)*prob_vk>0.2).view(SHAPE)
    if recon in train_dataset:
        count+=1

print(f"Dataset Reconstruction: {count/len(train_dataset):.02}")

# Set the model to evaluation mode
rbm.eval()
torch.save(rbm,"quantum_bas.pt")
# rbm =  torch.load("qbas_unsupervised.pt")

# Error graph
plt.plot(err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")
plt.savefig("quantum_err_log.png")

# Visible bias graph
ax = plt.gca()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',DATA_SIZE).colors))
lc_v = ax.plot(bv_log)
plt.legend(iter(lc_v),[f'bv{i}' for i in range(DATA_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Visible Biases")
plt.xlabel("Epoch")
plt.savefig("quantum_bv_log.png")

# Hidden bias graph
ax = plt.gca()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',HIDDEN_SIZE).colors))
lc_h = plt.plot(bh_log)
plt.legend(lc_h,[f'bh{i}' for i in range(HIDDEN_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Hidden Biases")
plt.xlabel("Epoch")
plt.savefig("quantum_bh_log.png")

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
# plt.hist(gibbs_energies,weights=weights(gibbs_energies),label="Gibbs-1",color='g',**hist_kwargs)
plt.hist(qa_energies,weights=weights(qa_energies),label="QA",color='orange', **hist_kwargs)
# plt.hist(sa_energies,weights=weights(sa_energies),label="SA",color='c',**hist_kwargs)
plt.legend(loc='upper right')
plt.ylim(0.0,0.05)
plt.ylabel("Count/Total")
plt.xlabel("Energy")
# SAMPLERS
sax = plt.gca().twinx()
plt.legend(loc='upper right')
plt.ylabel("Count/Total")
plt.xlabel("Energy")
plt.ylim(0.0,0.11)
plt.savefig("energies.pdf")


ax2.hist(qa_energies, label="QA", **hist_kwargs)
ax2.plot(t, s2, 'r.')
plt.ylabel('sin', color='r')
plt.show()

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
plt.savefig("quantum_weights.png")
