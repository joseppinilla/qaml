import qaml
import torch
import itertools

import matplotlib.pyplot as plt

import torchvision.transforms as torch_transforms
################################# Hyperparameters ##############################
SHAPE = (8,8)
EPOCHS = 10
BATCH_SIZE = 1024
# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

#################################### Input Data ################################
train_dataset = qaml.datasets.OptDigits(root='./data/', train=True,
                                     transform=torch_transforms.Compose([
                                     torch_transforms.ToTensor(),#]),
                                     lambda x:(x>0.6).to(x.dtype)]), #Binarize
                                     target_transform=torch_transforms.Compose([
                                     lambda x:torch.LongTensor([x]),
                                     lambda x:torch.nn.functional.one_hot(x-1,4)]),
                                     download=True)

train_dataset.classes = train_dataset.classes[1:5]
idx = (train_dataset.targets==1) | (train_dataset.targets==2) | (train_dataset.targets==3) | (train_dataset.targets==4)
train_dataset.targets = train_dataset.targets[idx]
train_dataset.data = train_dataset.data.reshape(-1,1,8,8)[idx]
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           shuffle=True)

test_dataset = qaml.datasets.OptDigits(root='./data/', train=False,
                                    transform=torch_transforms.Compose([
                                    torch_transforms.ToTensor(),#]),
                                    lambda x:(x>0.6).to(x.dtype)]), #Binarize
                                    target_transform=torch_transforms.Compose([
                                    lambda x:torch.LongTensor([x]),
                                    lambda x:torch.nn.functional.one_hot(x-1,4)]),
                                    download=True)
test_dataset.classes = test_dataset.classes[1:5]
idx = (test_dataset.targets==1) | (test_dataset.targets==2) | (test_dataset.targets==3) | (test_dataset.targets==4)
test_dataset.targets = test_dataset.targets[idx]
test_dataset.data = test_dataset.data.reshape(-1,1,8,8)[idx]
test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=True)

DATASET_SIZE = len(train_dataset)
################################# Model Definition #############################
DATA_SIZE = len(train_dataset.data[0].flatten())
LABEL_SIZE = len(train_dataset.classes)

VISIBLE_SIZE = DATA_SIZE + LABEL_SIZE
HIDDEN_SIZE = 8

# Specify model with dimensions
beta = torch.nn.Parameter(torch.tensor(2.0), requires_grad=True)
rbm = qaml.nn.RBM(VISIBLE_SIZE,HIDDEN_SIZE,beta=beta.item())

# Set up optimizer
optimizer = torch.optim.SGD(rbm.parameters(),
                            lr=learning_rate,
                            weight_decay=weight_decay,
                            momentum=momentum)

beta_optimizer = torch.optim.SGD([beta],lr=0.025)

# Set up training mechanisms
qa_sampler = qaml.sampler.QuantumAnnealingNetworkSampler(rbm, solver="DW_2000Q_6") #"Advantage_system1.1"
CD = qaml.autograd.SampleBasedConstrastiveDivergence()
betaGrad = qaml.autograd.AdaptiveBeta()
################################## Model Training ##############################
# Set the model to training mode
rbm.train()
err_log = []
beta_err_log = []
bv_log = [rbm.bv.detach().clone().numpy()]
bh_log = [rbm.bh.detach().clone().numpy()]
W_log = [rbm.W.detach().clone().numpy().flatten()]
for t in range(50):
    epoch_error = torch.Tensor([0.])
    for img_batch, labels_batch in train_loader:
        input_data = torch.cat((img_batch.flatten(1),labels_batch.flatten(1)),1)

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
        vk, prob_hk = qa_sampler(num_reads=100)

        # Reconstruction error from Contrastive Divergence
        err = CD.apply((v0,prob_h0), (vk,prob_hk), *rbm.parameters())

        # Do not accumulated gradients
        optimizer.zero_grad()
        # Compute gradients. Save compute graph at last epoch
        err.backward(retain_graph = (t==EPOCHS-1))

        # Update parameters
        optimizer.step()
        epoch_error  += err

        # Beta update
        err_beta = betaGrad.apply(rbm.energy(v0,prob_h0),rbm.energy(vk,prob_hk),beta)
        beta_optimizer.zero_grad()
        err_beta.backward()
        beta_optimizer.step()
        rbm.beta = beta.item()

    # Error Log
    bv_log.append(rbm.bv.detach().clone().numpy())
    bh_log.append(rbm.bh.detach().clone().numpy())
    W_log.append(rbm.W.detach().clone().numpy().flatten())
    err_log.append(epoch_error)
    beta_err_log.append(err_beta)
    print(f"Epoch {t} Reconstruction Error = {epoch_error.item()}")

    print(f"Beta = {rbm.beta}")
    ############################## CLASSIFICATION ##################################
    count = 0
    for i,(test_data, test_label) in enumerate(test_loader):
        prob_hk = rbm(torch.cat((test_data.flatten(1),torch.zeros(1,LABEL_SIZE)),dim=1))
        data_pred,label_pred = rbm.generate(prob_hk).split((DATA_SIZE,LABEL_SIZE),dim=1)
        if label_pred.argmax() == test_label.argmax():
            count+=1
    print(f"Testing accuracy: {count}/{len(test_dataset)} ({count/len(test_dataset):.2f})")


%matplotlib qt
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
fig,axs = plt.subplots(4,5)
for ax,(img,label) in zip(axs.flat,train_dataset):
    ax.matshow(img.view(8, 8))
    ax.set_title(str(label.argmax().item()))
    ax.axis('off')



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
rand_data = torch.rand(len(train_dataset)*10,rbm.V)
rand_energies = rbm.free_energy(rand_data.bernoulli()).detach().numpy()

data_energies = []
for img,label in train_dataset:
    data = torch.cat((img.flatten(1),label.flatten(1)),1)
    data_energies.append(rbm.free_energy(data).item())

gibbs_energies = []
gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm)
for img,label in train_dataset:
    data = torch.cat((img.flatten(1),label.flatten(1)),1)
    prob_v,prob_h = gibbs_sampler(data,k=5)
    gibbs_energies.append(rbm.free_energy(prob_v.bernoulli()).item())

qa_energies = []
qa_sampler = qaml.sampler.QuantumAnnealingNetworkSampler(rbm,solver="Advantage_system1.1")
qa_sampleset = qa_sampler(num_reads=1000)
for s_v,s_h in zip(*qa_sampleset):
    qa_energies.append(rbm.free_energy(s_v.detach()).item())

pcd_energies = []
pcd_sampler = qaml.sampler.PersistentGibbsNetworkSampler(rbm,BATCH_SIZE)
pcd_sampleset = pcd_sampler(BATCH_SIZE,k=1)
for s_v,s_h in zip(*pcd_sampleset):
    pcd_energies.append(rbm.free_energy(s_v).item())


import matplotlib
import numpy as np
%matplotlib qt
hist_kwargs = {'ec':'k','lw':2.0,'alpha':0.5,'histtype':'stepfilled','bins':100}
matplotlib.rcParams.update({'font.size': 22})
weights = lambda data: np.ones_like(data)/len(data)

plt.hist(rand_energies,weights=weights(rand_energies),label="Random",color='r',**hist_kwargs)
plt.hist(data_energies,weights=weights(data_energies), label="Data", color='b', **hist_kwargs)
plt.hist(gibbs_energies,weights=weights(gibbs_energies),label="Gibbs-1",color='g',**hist_kwargs)
plt.hist(pcd_energies,weights=weights(pcd_energies),label="PCD-1",color='purple',**hist_kwargs)
plt.hist(qa_energies,weights=weights(qa_energies),label="QA",color='orange', **hist_kwargs)

plt.legend(loc='upper right')
plt.ylim(0.0,0.05)
plt.ylabel("Count/Total")
plt.xlabel("Energy")

plt.savefig("energies.pdf")


################################## VISUALIZE ###################################
plt.matshow(rbm.bv.detach()[:DATA_SIZE].view(*SHAPE), cmap='viridis', vmin=-1, vmax=1)
plt.colorbar()

fig,axs = plt.subplots(HIDDEN_SIZE//4,4)
for i,ax in enumerate(axs.flat):
    weight_matrix = rbm.W[i].detach()[:DATA_SIZE].view(*SHAPE)
    ms = ax.matshow(weight_matrix, cmap='viridis', vmin=-1, vmax=1)
    ax.axis('off')
cbar = fig.colorbar(ms, ax=axs.ravel().tolist(), shrink=0.95)
plt.savefig("quantum_weights.png")
