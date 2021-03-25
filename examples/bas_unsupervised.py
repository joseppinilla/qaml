
import qaml
import torch

import matplotlib.pyplot as plt

import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

################################# Hyperparameters ##############################
EPOCHS = 300
SAMPLES = 1000
BATCH_SIZE = 500
# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

#################################### Input Data ################################
train_dataset = qaml.datasets.BAS(7,7,transform=torch_transforms.ToTensor())
train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=True,
                                               num_samples=SAMPLES)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           sampler=train_sampler,
                                           batch_size=BATCH_SIZE)

DATASET_SIZE = SAMPLES*len(train_dataset)

################################# Model Definition #############################
DATA_SIZE = len(train_dataset.data[0].flatten())
HIDDEN_SIZE = 25

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
    err_log.append(epoch_error)
    print(f"Epoch {t} Reconstruction Error = {epoch_error.item()}")

# Error graph
plt.plot(err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")

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

plt.hist(noise_energies,label="Random",bins=100)
plt.hist(data_energies, label="Data",bins=100)
plt.ylabel("Count")
plt.xlabel("Energy")
plt.legend()

################################## VISUALIZE ###################################
fig,axs = plt.subplots(5,5)
for i,ax in enumerate(axs.flat):
    weight_matrix = rbm.W[i].detach().view(7, 7)
    ms = ax.matshow(weight_matrix, cmap='viridis', vmin=-1, vmax=1)
    ax.axis('off')
fig.subplots_adjust(wspace=0.0, hspace=0.0)
cbar = fig.colorbar(ms, ax=axs.ravel().tolist(), shrink=0.95)
