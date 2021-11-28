# %% markdown
# # Classical RBM training on the Bars-And-Stripes Dataset for Reconstruction
# This is an example on classical Gibbs training of an RBM on the BAS(4,4)
# dataset.
# Developed by: Jose Pinilla
# %%
# Required packages
import qaml
import torch
torch.manual_seed(0) # For deterministic weights

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as torch_transforms

# %%
################################# Hyperparameters ##############################
M,N = SHAPE = (8,8)
DATA_SIZE = N*M
HIDDEN_SIZE = 64
EPOCHS = 200
SAMPLES = None
BATCH_SIZE = 400
TRAIN,TEST = SPLIT = 400,110
# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 0.001
momentum = 0.5

# %%
#################################### Input Data ################################
bas_dataset = qaml.datasets.BAS(*SHAPE,embed_label=True,transform=torch.Tensor)
train_dataset,test_dataset = torch.utils.data.random_split(bas_dataset,[*SPLIT])
train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=False,
                                               num_samples=SAMPLES)
train_loader = torch.utils.data.DataLoader(train_dataset,sampler=train_sampler,
                                           batch_size=BATCH_SIZE)

# PLot all data
fig,axs = plt.subplots(6,5)
for ax,(img,label) in zip(axs.flat,train_dataset):
    ax.matshow(img.view(*SHAPE),vmin=0,vmax=1); ax.axis('off')
plt.tight_layout()

# %%
################################# Model Definition #############################
# Specify model with dimensions
rbm = qaml.nn.RBM(DATA_SIZE, HIDDEN_SIZE)

# Initialize biases
weight_init = 0.5
_ = torch.nn.init.constant_(rbm.b,0.5)
_ = torch.nn.init.zeros_(rbm.c)
_ = torch.nn.init.uniform_(rbm.W,-weight_init,weight_init)

# Set up optimizer
optimizer = torch.optim.SGD(rbm.parameters(),lr=learning_rate,
                            weight_decay=weight_decay,momentum=momentum)

# Set up training mechanisms
beta = 1.0
gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm,beta=beta)
CD = qaml.autograd.SampleBasedConstrastiveDivergence()

# %%
################################## Model Training ##############################
# Set the model to training mode
rbm.train()
err_log = []
accuracy_log = []
b_log = [rbm.b.detach().clone().numpy()]
c_log = [rbm.c.detach().clone().numpy()]
W_log = [rbm.W.detach().clone().numpy().flatten()]
for t in range(EPOCHS):
    epoch_error = torch.Tensor([0.])
    for img_batch, labels_batch in train_loader:
        input_data = img_batch.flatten(1)

        # Positive Phase
        v0, prob_h0 = input_data, rbm(input_data,scale=beta)
        # Negative Phase
        vk, prob_hk = gibbs_sampler(v0.detach(), k=5)

        # Reconstruction error from Contrastive Divergence
        err = CD.apply((v0,prob_h0), (vk,prob_hk), *rbm.parameters())

        # Do not accumulate gradients
        optimizer.zero_grad()

        # Compute gradients
        err.backward()

        # Update parameters
        optimizer.step()

        #Accumulate error for this epoch
        epoch_error  += err

    # Error Log
    b_log.append(rbm.b.detach().clone().numpy())
    c_log.append(rbm.c.detach().clone().numpy())
    W_log.append(rbm.W.detach().clone().numpy().flatten())
    err_log.append(epoch_error.item())
    print(f"Epoch {t} Reconstruction Error = {epoch_error.item()}")
    ############################## CLASSIFICATION ##################################
    count = 0
    for test_data, test_label in test_dataset:
        test_data[-2:,-1] = 0.5
        prob_hk = rbm(test_data.flatten(),scale=beta)
        label_pred = rbm.generate(prob_hk,scale=beta).view(*SHAPE)[-2:,-1]
        if label_pred.argmax() == test_label.argmax():
            count+=1
    accuracy_log.append(count/TEST)
    print(f"Testing accuracy: {count}/{TEST} ({count/TEST:.2f})")

# %%
############################## RECONSTRUCTION ##################################
bce = []
mask = torch_transforms.functional.erase(torch.ones(1,M,N),4,4,4,4,0).flatten()
for img, label in train_dataset:
    input_data = img.flatten()
    prob_vk,prob_hk = gibbs_sampler.reconstruct(input_data,k=5,mask=mask)
    bce.append(F.binary_cross_entropy(input_data,prob_vk).item())
_ = plt.hist(bce,bins=100)

# %%
############################ MODEL VISUALIZATION ###############################
# Testing accuracy graph
fig, ax = plt.subplots()
plt.plot(accuracy_log)
plt.ylabel("Testing Accuracy")
plt.xlabel("Epoch")
plt.savefig("classical_accuracy.pdf")

# L1 error graph
fig, ax = plt.subplots()
plt.plot(err_log)
plt.ylabel("Reconstruction Error (L1)")
plt.xlabel("Epoch")
plt.savefig("classical_err_log.pdf")

# Visible bias graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',DATA_SIZE).colors))
lc_v = ax.plot(b_log)
plt.legend(lc_v,[f'b{i}' for i in range(DATA_SIZE)],ncol=4,loc=(0,1))
plt.ylabel("Visible Biases")
plt.xlabel("Epoch")
plt.savefig("classival_b_log.pdf")

# Hidden bias graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',HIDDEN_SIZE).colors))
lc_h = plt.plot(c_log)
plt.legend(lc_h,[f'c{i}' for i in range(HIDDEN_SIZE)],ncol=4,loc=(0,1))
plt.ylabel("Hidden Biases")
plt.xlabel("Epoch")
plt.savefig("classical_c_log.pdf")

# Weights graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',rbm.V*rbm.H).colors))
lc_w = plt.plot(W_log)
plt.ylabel("Weights")
plt.xlabel("Epoch")

# %%
################################## ENERGY ######################################
data_energies = []
for img,label in bas_dataset:
    data = img.flatten()
    data_energies.append(rbm.free_energy(data).item())

rand_energies = []
rand_data = torch.rand(len(bas_dataset)*10,rbm.V)
for img in rand_data:
    rand_energies.append(rbm.free_energy(img.bernoulli()).item())

gibbs_energies = []
for img,label in bas_dataset:
    data = img.flatten()
    prob_v,prob_h = gibbs_sampler(data,k=5)
    gibbs_energies.append(rbm.free_energy(prob_v.bernoulli()).item())

qa_energies = []
solver_name = "Advantage_system4.1"
qa_sampler = qaml.sampler.QASampler(rbm,solver=solver_name)
qa_sampleset = qa_sampler(num_reads=1000,auto_scale=True)
for s_v,s_h in zip(*qa_sampleset):
    qa_energies.append(rbm.free_energy(s_v.detach()).item())

plot_data = [(data_energies,  'Data',    'blue'),
             (rand_energies,  'Random',  'red'),
             (gibbs_energies, 'Gibbs-5', 'green'),
             (qa_energies,    'Quantum', 'orange')]

hist_kwargs = {'ec':'k','lw':2.0,'alpha':0.5,'histtype':'stepfilled','bins':100}
weights = lambda data: [1./len(data) for _ in data]

fig, ax = plt.subplots(figsize=(15,10))
for data,name,color in plot_data:
    ax.hist(data,weights=weights(data),label=name,color=color,**hist_kwargs)

plt.xlabel("Energy")
plt.ylabel("Count/Total")
plt.legend(loc='upper right')
plt.savefig("classical_energies.pdf")

# %%
################################## VISUALIZE ###################################
plt.matshow(rbm.b.detach().view(*SHAPE))
plt.colorbar()
plt.savefig("classical_b.pdf")
plt.matshow(rbm.c.detach().view(1,HIDDEN_SIZE))
plt.yticks([])
plt.colorbar()
plt.savefig("classical_c.pdf")

fig,axs = plt.subplots(HIDDEN_SIZE//4,4)
for i,ax in enumerate(axs.flat):
    weight_matrix = rbm.W[i].detach().view(*SHAPE)
    ms = ax.matshow(weight_matrix)
    ax.axis('off')
fig.subplots_adjust(wspace=0.1, hspace=0.1)
cbar = fig.colorbar(ms, ax=axs.ravel().tolist(), shrink=0.95)
plt.savefig("classical_weights.pdf")
