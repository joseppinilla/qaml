
import qaml
import torch

import matplotlib.pyplot as plt

import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

################################# Hyperparameters ##############################
SHAPE = (7,7)
EPOCHS = 10
SAMPLES = 1000
BATCH_SIZE = 250
# Stochastic Gradient Descent
learning_rate = 0.7
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
HIDDEN_SIZE = 25

# Specify model with dimensions
rbm = qaml.nn.RBM(DATA_SIZE, HIDDEN_SIZE)

# Set up optimizer
optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate)
                                              # weight_decay=weight_decay,
                                              # momentum=momentum)

# Set up training mechanisms
import embera
import minorminer


sa_sampler = qaml.sampler.SimulatedAnnealingNetworkSampler(rbm)
bqm = sa_sampler.binary_quadratic_model
H = embera.architectures.dwave_collection(chip_id='DW_2000Q_6')[0]
embedding = minorminer.find_embedding(bqm.quadratic,H)
embera.draw_architecture_embedding(H,embedding,node_size=10)

sampler = qaml.sampler.QuantumAssistedNetworkSampler(rbm, embedding,solver="DW_2000Q_6")
CD = qaml.autograd.SampleBasedConstrastiveDivergence()

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
        vk, prob_hk = sampler(num_reads=1000)

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
plt.savefig("err_log.png")

################################## ENERGY ######################################

data_energies = []
for img,_ in train_dataset:
    data_energies.append(rbm.free_energy(img.float().view(rbm.V)).item())

noise_energies = []
for _ in range(len(train_dataset)):
    noise_energies.append(rbm.free_energy(torch.rand(rbm.V).bernoulli()).item())

hist_kwargs = {'alpha':0.5,'density':True,'bins':100}
plt.hist(noise_energies,label="Random", **hist_kwargs)
plt.hist(data_energies, label="Data", **hist_kwargs)
plt.ylabel("Count")
plt.xlabel("Energy")
plt.legend()
plt.savefig("energies.png")

################################## VISUALIZE ###################################
fig,axs = plt.subplots(5,5)
for i,ax in enumerate(axs.flat):
    weight_matrix = rbm.W[i].detach().view(*SHAPE)
    ms = ax.matshow(weight_matrix, cmap='viridis', vmin=-1, vmax=1)
    ax.axis('off')
fig.subplots_adjust(wspace=0.0, hspace=0.0)
cbar = fig.colorbar(ms, ax=axs.ravel().tolist(), shrink=0.95)
plt.savefig("weights.png")

#################################### SAMPLE ####################################
sample_v,sample_h = sampler(num_reads=1, num_spin_reversal_transforms=0)
plt.matshow(sample_v.view(*SHAPE).detach())
