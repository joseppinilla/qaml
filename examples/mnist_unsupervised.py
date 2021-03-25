
import qaml
import torch

import matplotlib.pyplot as plt

import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

################################# Hyperparameters ##############################
EPOCHS = 10
BATCH_SIZE = 64
# Stochastic Gradient Descent
learning_rate = 1e-3
weight_decay = 1e-4
momentum = 0.5

#################################### Input Data ################################
train_dataset = torch_datasets.MNIST(root='./data/', train=True,
                                     transform=torch_transforms.ToTensor(),
                                     download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           shuffle=True)

test_dataset = torch_datasets.MNIST(root='./data/', train=False,
                                    transform=torch_transforms.ToTensor(),
                                    download=True)
test_loader = torch.utils.data.DataLoader(test_dataset)

################################# Model Definition #############################
DATA_SIZE = len(train_dataset.data[0].flatten())
HIDDEN_SIZE = 128

# Specify model with dimensions
rbm = qaml.nn.RBM(DATA_SIZE, HIDDEN_SIZE)

# Set up optimizer
optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate,
                                              weight_decay=weight_decay,
                                              momentum=momentum)
# Set up training mechanisms
sampler = qaml.sampler.PersistentGibbsNetworkSampler(rbm, BATCH_SIZE)
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
        vk, prob_hk = sampler(len(v0), k=1)

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

plt.plot(err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")

# Set the model to evaluation mode
rbm.eval()
torch.save(rbm,"mnist_unsupervised.pt")
# rbm = torch.load("mnist_unsupervised.pt")
################################# VISUALIZE ####################################

# Computation Graph
from torchviz import make_dot
make_dot(err)

################################## ENERGY ######################################
data_energies = []
for img,_ in train_dataset:
    data_energies.append(rbm.free_energy(img.float().view(rbm.V)).item())

test_energies = []
for img,_ in test_dataset:
    test_energies.append(rbm.free_energy(img.float().view(rbm.V)).item())

plt.hist(data_energies, label="Data",bins=100)
plt.hist(test_energies, label="Test",bins=100)
plt.ylabel("Count")
plt.xlabel("Energy")
plt.legend()

################################# FEATURES #####################################
plt.matshow(rbm.bv.detach().view(28, 28))
plt.colorbar()

fig,axs = plt.subplots(16,8)
for i,ax in enumerate(axs.flat):
    weight_matrix = rbm.W[i].detach().view(28, 28)
    ms = ax.matshow(weight_matrix, cmap='viridis', vmin=-1, vmax=1)
    ax.axis('off')
fig.subplots_adjust(wspace=0.0, hspace=0.0)
cbar = fig.colorbar(ms, ax=axs.ravel().tolist(), shrink=0.95)

########## SAMPLE ##########
import torch.nn.functional as F
k=5
input = torch.randn(784)
for _ in range(k):
    pH_v = torch.sigmoid(F.linear(input, rbm.W, rbm.bh))
    pV_h = torch.sigmoid(F.linear(pH_v, rbm.W.t(), rbm.bv))
    input.data = pV_h.bernoulli()

plt.matshow(pV_h.detach().view(28, 28))

########## NOISE RECONSTRUCTION ##########
input_data, label = train_loader.dataset[22]
corrupt_data = (input_data + torch.randn_like(input_data)*0.2).view(784)

plt.matshow(corrupt_data.view(28, 28))
pV_h = corrupt_data
for _ in range(1):
    pH_v = torch.sigmoid(F.linear(pV_h, rbm.W, rbm.bh))
    pV_h = torch.sigmoid(F.linear(pH_v, rbm.W.t(), rbm.bv))
    input.data = pV_h.bernoulli()

plt.matshow(pV_h.detach().view(28, 28))

########## RECONSTRUCTION ##########
input_data, label = train_loader.dataset[4]
mask = torch.ones_like(input_data)
for i in range(0,15):
    for j in range(0,15):
        mask[0][j][i] = 0
plt.matshow(input_data.view(28, 28))
corrupt_data = (input_data*mask).view(1,784)
plt.matshow(corrupt_data.view(28, 28))

pV_h = corrupt_data
for _ in range(3):
    pH_v = torch.sigmoid(F.linear(pV_h, rbm.W, rbm.bh))
    pV_h = torch.sigmoid(F.linear(pH_v, rbm.W.t(), rbm.bv))
    input.data = pV_h.bernoulli()
plt.matshow(pV_h.detach().view(28, 28))

######################## CLASSIFIER
LABEL_SIZE = len(train_dataset.classes)

model = torch.nn.Sequential(rbm,
                            torch.nn.Linear(HIDDEN_SIZE,LABEL_SIZE),)
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for t in range(10):
    for v_batch, labels_batch in train_loader:
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(v_batch.view(len(v_batch),DATA_SIZE))

        # Compute and print loss.
        loss = loss_fn(y_pred, torch.nn.functional.one_hot(labels_batch,10)*1.0)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the Tensors it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
    print(f"Epoch {t} Loss = {loss.item()}")

count = 0
for test_data, test_label in test_loader:
    label_pred = model(test_data.view(1,DATA_SIZE)).argmax()
    if label_pred == test_label:
        count+=1
print(f"{count}/{len(test_dataset)}")
