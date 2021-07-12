# %% markdown
# # Quantum-Assisted RBM training on the BAS Dataset for Reconstruction
# This is an example on quantum-assisted training of an RBM on the BAS(4,4)
# dataset.
# Developed by: Jose Pinilla
# %%
# Required packages
import qaml
import torch

import matplotlib.pyplot as plt
import torchvision.transforms as torch_transforms

# %%
################################# Hyperparameters ##############################
SHAPE = (4,4)
DATA_SIZE = 4*4
HIDDEN_SIZE = 16
EPOCHS = 10
SAMPLES = 1000
BATCH_SIZE = 500
# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

# %%
#################################### Input Data ################################
train_dataset = qaml.datasets.BAS(*SHAPE,transform=torch_transforms.ToTensor())
train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=True,
                                               num_samples=SAMPLES)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           sampler=train_sampler,
                                           batch_size=BATCH_SIZE)

# PLot all data
fig,axs = plt.subplots(6,5)
for ax,(img,label) in zip(axs.flat,train_dataset):
    ax.matshow(img.view(*SHAPE),vmin=0,vmax=1); ax.axis('off')
plt.tight_layout()

# %%
################################# Model Definition #############################
# Specify model with dimensions
beta = torch.nn.Parameter(torch.tensor(1.5), requires_grad=True)
rbm = qaml.nn.RBM(DATA_SIZE,HIDDEN_SIZE,beta=beta)

# Initialize biases
torch.nn.init.uniform_(rbm.b,-0.1,0.1)
torch.nn.init.uniform_(rbm.c,-0.1,0.1)
torch.nn.init.uniform_(rbm.W,-0.1,0.1)


# Set up optimizer
optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate,
                            weight_decay=weight_decay, momentum=momentum)

beta_optimizer = torch.optim.SGD([beta],lr=0.01)

# Set up training mechanisms
solver_name = "Advantage_system1.1"
qa_sampler = qaml.sampler.QuantumAnnealingNetworkSampler(rbm,solver=solver_name)
CD = qaml.autograd.SampleBasedConstrastiveDivergence()
betaGrad = qaml.autograd.AdaptiveBeta()

# %%
################################## Model Training ##############################
# Set the model to training mode
rbm.train()
err_log = []
err_beta_log = []
b_log = [rbm.b.detach().clone().numpy()]
c_log = [rbm.c.detach().clone().numpy()]
W_log = [rbm.W.detach().clone().numpy().flatten()]
for t in range(EPOCHS):
    epoch_error = torch.Tensor([0.])
    epoch_error_beta = torch.Tensor([0.])

    for img_batch, labels_batch in train_loader:
        input_data = img_batch.flatten(1)

        # Positive Phase
        v0, prob_h0 = input_data, rbm(input_data)
        # Negative Phase
        vk, prob_hk = qa_sampler(1000,auto_scale=False)

        # Reconstruction error from Contrastive Divergence
        err = CD.apply((v0,prob_h0), (vk,prob_hk), *rbm.parameters())
        err_beta = betaGrad.apply(rbm.energy(v0,prob_h0),rbm.energy(vk,prob_hk),beta) #TODO: use sampleset Energies?

        # Do not accumulated gradients
        optimizer.zero_grad()
        beta_optimizer.zero_grad()

        # Compute gradients. Save compute graph at last epoch
        err.backward()
        err_beta.backward()

        # Update parameters
        optimizer.step()
        beta_optimizer.step()

        #Accumulate error for this epoch
        epoch_error  += err
        epoch_error_beta  += err_beta

    # Error Log
    b_log.append(rbm.b.detach().clone().numpy())
    c_log.append(rbm.c.detach().clone().numpy())
    W_log.append(rbm.W.detach().clone().numpy().flatten())
    err_log.append(epoch_error.item())
    err_beta_log.append(epoch_error_beta.item())
    print(f"Epoch {t} Reconstruction Error = {epoch_error.item()}")
    print(f"Beta = {rbm.beta}")
    print(f"Alpha = {qa_sampler.scalar}")
    print(f"Effective Beta = {rbm.beta*qa_sampler.scalar}")

# %%
################################# qBAS Score ###################################
N = 1000 # CLASSICAL
gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm)
prob_v,_ = gibbs_sampler(torch.rand(N,DATA_SIZE),k=10)
img_samples = prob_v.view(N,*SHAPE).bernoulli()
# PLot some samples
fig,axs = plt.subplots(4,5)
for ax,img in zip(axs.flat,img_samples):
    ax.matshow(img.view(*SHAPE),vmin=0,vmax=1); ax.axis('off')
plt.tight_layout()
# Get and print score
p,r,score = train_dataset.score(img_samples)
print(f"qBAS : Precision = {p:.02} Recall = {r:.02} Score = {score:.02}")

############################## RECONSTRUCTION ##################################
k = 10
count = 0
mask = torch_transforms.functional.erase(torch.ones(SHAPE),1,1,2,2,0).flatten()
for img, label in train_dataset:

    clamped = mask*img.flatten(1)
    prob_hk = rbm.forward(clamped + (1-mask)*0.5)
    prob_vk = rbm.generate(prob_hk).detach()
    for _ in range(k):
        masked = clamped + (1-mask)*prob_vk.data
        prob_hk.data = rbm.forward(masked).data
        prob_vk.data = rbm.generate(prob_hk).data
    recon = (clamped + (1-mask)*prob_vk).bernoulli().view(img.shape)

    if recon.equal(img):
        count+=1

print(f"Dataset Reconstruction: {count/len(train_dataset):.02}")

############################ MODEL VISUALIZATION ###############################

# Beta error graph
plt.plot(err_beta_log)
plt.ylabel("Beta Error")
plt.xlabel("Epoch")

# Error graph
plt.plot(err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")
plt.savefig("quantum_err_log.pdf")

# Visible bias graph
ax = plt.gca()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',DATA_SIZE).colors))
lc_v = ax.plot(b_log)
plt.legend(iter(lc_v),[f'b{i}' for i in range(DATA_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Visible Biases")
plt.xlabel("Epoch")
plt.savefig("quantum_b_log.pdf")

# Hidden bias graph
ax = plt.gca()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',HIDDEN_SIZE).colors))
lc_h = plt.plot(c_log)
plt.legend(lc_h,[f'c{i}' for i in range(HIDDEN_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Hidden Biases")
plt.xlabel("Epoch")
plt.savefig("quantum_c_log.pdf")

################################## ENERGY ######################################
rand_data = torch.rand(len(train_dataset)*10,rbm.V)
rand_energies = rbm.free_energy(rand_data.bernoulli()).detach().numpy()

data_energies = []
for img,label in train_dataset:
    data = img.flatten(1)
    data_energies.append(rbm.free_energy(data).item())

gibbs_energies = []
gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm)
for img,label in train_dataset:
    data = img.flatten(1)
    prob_v,prob_h = gibbs_sampler(data,k=5)
    gibbs_energies.append(rbm.free_energy(prob_v.bernoulli()).item())

qa_energies = []
qa_sampler = qaml.sampler.QuantumAnnealingNetworkSampler(rbm,solver="Advantage_system1.1")
qa_sampleset = qa_sampler(num_reads=1000)
for s_v,s_h in zip(*qa_sampleset):
    qa_energies.append(rbm.free_energy(s_v.detach()).item())


import matplotlib
import numpy as np
hist_kwargs = {'ec':'k','lw':2.0,'alpha':0.5,'histtype':'stepfilled','bins':100}
matplotlib.rcParams.update({'font.size': 22})
weights = lambda data: np.ones_like(data)/len(data)

plt.hist(rand_energies,weights=weights(rand_energies),label="Random",color='r',**hist_kwargs)
plt.hist(data_energies,weights=weights(data_energies), label="Data", color='b', **hist_kwargs)
plt.hist(gibbs_energies,weights=weights(gibbs_energies),label="Gibbs-1",color='g',**hist_kwargs)
plt.hist(qa_energies,weights=weights(qa_energies),label="QA",color='orange', **hist_kwargs)

plt.legend(loc='upper right')
plt.ylim(0.0,0.05)
plt.ylabel("Count/Total")
plt.xlabel("Energy")
plt.savefig("quantum_energies.pdf")

################################## VISUALIZE ###################################
plt.matshow(rbm.bv.detach().view(*SHAPE), cmap='viridis')
plt.colorbar()

fig,axs = plt.subplots(HIDDEN_SIZE//4,4)
for i,ax in enumerate(axs.flat):
    weight_matrix = rbm.W[i].detach().view(*SHAPE)
    ms = ax.matshow(weight_matrix, cmap='viridis', vmin=-1, vmax=1)
    ax.axis('off')
fig.subplots_adjust(wspace=0.1, hspace=0.1)
cbar = fig.colorbar(ms, ax=axs.ravel().tolist(), shrink=0.95)
plt.savefig("quantum_weights.pdf")