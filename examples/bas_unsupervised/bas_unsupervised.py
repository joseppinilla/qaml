
import qaml
import torch

import matplotlib.pyplot as plt

import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

################################# Hyperparameters ##############################
SHAPE = (4,4)
EPOCHS = 20
SAMPLES = 250
BATCH_SIZE = 1
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
sa_sampler = qaml.sampler.SimulatedAnnealingNetworkSampler(rbm)
# CD = qaml.autograd.SampleBasedConstrastiveDivergence()

################################## Model Training ##############################
# Set the model to training mode
rbm.train()
err_log = []
for t in range(EPOCHS):
    epoch_error = torch.Tensor([0.])
    for img_batch, labels_batch in train_loader:
        input_data = img_batch.flatten(1)

        # Positive Phase
        v0, prob_h0 = input_data, rbm(input_data)
        # Negative Phase
        vk, prob_hk = sampler(v0.detach(), k=1)
        # vk, prob_hk = sa_sampler(num_reads=BATCH_SIZE, num_sweeps=10)

        # Reconstruction error from Contrastive Divergence
        err = CD.apply((v0,prob_h0), (vk,prob_hk), *rbm.parameters())

        # Do not accumulated gradients
        optimizer.zero_grad()
        # Compute gradients. Save compute graph at last epoch
        err.backward(retain_graph=(t == EPOCHS-1))

        # Update parameters
        optimizer.step()
        epoch_error  += err
    err_log.append(epoch_error)
    print(f"Epoch {t} Reconstruction Error = {epoch_error.item()}")


# Error graph
plt.plot(err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")
plt.savefig("gibbs_err_log.png")

# Set the model to evaluation mode
rbm.eval()
torch.save(rbm,"bas_unsupervised.pt")
# rbm = torch.load("bas_unsupervised.pt")
################################## ENERGY ######################################

data_energies = []
for img,_ in train_dataset:
    data_energies.append(rbm.free_energy(img.float().view(rbm.V)).item())

noise_energies = []
for _ in range(len(train_dataset)):
    noise_energies.append(rbm.free_energy(torch.rand(rbm.V).bernoulli()).item())

model_energies = []
for _ in range(len(train_dataset)):
    v,h = sa_sampler(num_reads=1)
    model_energies.append(rbm.energy(v.flatten(),h.flatten()).item())

hist_kwargs = {'alpha':0.5,'density':True,'bins':100}
plt.hist(noise_energies,label="Random", **hist_kwargs)
plt.hist(data_energies, label="Data", **hist_kwargs)
plt.hist(model_energies, label="Model", **hist_kwargs)
plt.ylabel("Count")
plt.xlabel("Energy")
plt.legend()
plt.savefig("gibbs_energies.png")

################################## VISUALIZE ###################################
fig,axs = plt.subplots(3,5)
for i,ax in enumerate(axs.flat):
    weight_matrix = rbm.W[i].detach().view(*SHAPE)
    ms = ax.matshow(weight_matrix, cmap='viridis', vmin=-1, vmax=1)
    ax.axis('off')
fig.subplots_adjust(wspace=0.0, hspace=0.0)
cbar = fig.colorbar(ms, ax=axs.ravel().tolist(), shrink=0.95)
plt.savefig("weights.png")

#################################### SAMPLE ####################################
sample_v,sample_h = sa_sampler(num_sweeps=1000)
plt.matshow(sample_v.view(*SHAPE))
