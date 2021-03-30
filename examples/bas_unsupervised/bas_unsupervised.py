
import qaml
import torch
import itertools

import matplotlib.pyplot as plt

import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

################################# Hyperparameters ##############################
SHAPE = (3,3)
EPOCHS = 300
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
HIDDEN_SIZE = 15

# Specify model with dimensions
rbm = qaml.nn.RBM(DATA_SIZE, HIDDEN_SIZE)

# Set up optimizer
optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate,
                                              weight_decay=weight_decay,
                                              momentum=momentum)

# Set up training mechanisms
sampler = qaml.sampler.GibbsNetworkSampler(rbm)
CD = qaml.autograd.ConstrastiveDivergence()

################################## Model Training ##############################
# Set the model to training mode
rbm.train()
err_log = []
bv_log = []
bh_log = []
for t in range(EPOCHS):
    epoch_error = torch.Tensor([0.])
    for img_batch, labels_batch in train_loader:
        input_data = img_batch.flatten(1)

        # Positive Phase
        v0, prob_h0 = input_data, rbm(input_data)
        # Negative Phase
        vk, prob_hk = sampler(v0.detach(), k=1)

        # Reconstruction error from Contrastive Divergence
        err = CD.apply((v0,prob_h0), (vk,prob_hk), *rbm.parameters())

        # Do not accumulated gradients
        optimizer.zero_grad()
        # Compute gradients. Save compute graph at last epoch
        err.backward(retain_graph=(t == EPOCHS-1))

        # Update parameters
        optimizer.step()
        epoch_error  += err
    # Log
    bv_log.append(rbm.bv.detach().clone().numpy())
    bh_log.append(rbm.bh.detach().clone().numpy())
    err_log.append(epoch_error)
    print(f"Epoch {t} Reconstruction Error = {epoch_error.item()}")

# Error graph
plt.plot(err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")
plt.savefig("err_log.png")

# Visible bias graph
ax = plt.gca()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',DATA_SIZE).colors))
lc_v = ax.plot(bv_log)
plt.legend(iter(lc_v),[f'bv{i}' for i in range(DATA_SIZE)],ncol=4,bbox_to_anchor=(1,1))
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

# Set the model to evaluation mode
rbm.eval()
torch.save(rbm,"bas_unsupervised.pt")

################################## ENERGY ######################################
data_energies = []
for img,_ in train_dataset:
    data_energies.append(rbm.free_energy(img.float().view(rbm.V)).item())

rand_energies = []
for _ in range(len(train_dataset)*10):
    rand_energies.append(rbm.free_energy(torch.rand(rbm.V).bernoulli()).item())


hist_kwargs = {'alpha':0.5,'density':False,'bins':100}
plt.hist(rand_energies,label="Random", **hist_kwargs)
plt.hist(data_energies, label="Data", **hist_kwargs)
plt.ylabel("Count")
plt.xlabel("Energy")
plt.legend()
plt.savefig("energies.png")

Z = 0.0
energies = []
for v in itertools.product([0, 1], repeat=DATA_SIZE):
    E = rbm.free_energy(torch.tensor(v,dtype=torch.float))
    energies.append(E)
    Z+=torch.exp(-E)
plt.bar(range(2**DATA_SIZE),[torch.exp(-E).item()/Z.item() for E in energies])


################################## VISUALIZE ###################################
plt.matshow(rbm.bv.detach().view(*SHAPE), cmap='viridis', vmin=-1, vmax=1)
plt.colorbar()

fig,axs = plt.subplots(HIDDEN_SIZE//4,4)
for i,ax in enumerate(axs.flat):
    weight_matrix = rbm.W[i].detach().view(*SHAPE)
    ms = ax.matshow(weight_matrix, cmap='viridis', vmin=-1, vmax=1)
    ax.axis('off')
fig.subplots_adjust(wspace=0.1, hspace=0.1)
cbar = fig.colorbar(ms, ax=axs.ravel().tolist(), shrink=0.95)
plt.savefig("weights.png")

#################################### SAMPLE ####################################
sample_v,sample_h = sampler(torch.rand(DATA_SIZE),k=5)
plt.matshow(sample_v.view(*SHAPE).bernoulli().detach())
