# # Quantum RBM training on the MNIST Dataset for reconstruction and classification
# This is an example on quantum annealing training of an RBM on the MNIST
# dataset.
# Developed by: Jose Pinilla

# Required packages
import os
import qaml
import torch

import matplotlib.pyplot as plt
import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

from torch.nn.functional import max_pool2d

# Deterministic results
SEED = 8
_ = torch.manual_seed(SEED)


################################# Hyperparameters ##############################
EPOCHS = 1
M,N = SHAPE = (13,13)
BATCH_SIZE = 1024
DATA_SIZE = M*N

# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

# Set up training mechanisms
auto_scale = True
solver_name = "Advantage_system4.1"

############################ Dataset and Transformations #######################
mnist_train = torch_datasets.MNIST(root='./data/', train=True, download=True,
                                   transform=qaml.datasets.ToSpinTensor())

mnist_train.data = torch_transforms.functional.crop(mnist_train.data.float(),1,1,26,26).byte()
mnist_train.data = max_pool2d(mnist_train.data.float(),(2,2)).byte()
qaml.datasets._embed_labels(mnist_train,axis=1,encoding='one_hot',scale=255)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE)

mnist_test = torch_datasets.MNIST(root='./data/', train=False, download=True,
                                  transform=qaml.datasets.ToSpinTensor())
mnist_test.data = torch_transforms.functional.crop(mnist_test.data.float(),1,1,26,26).byte()
mnist_test.data = max_pool2d(mnist_test.data.float(),(2,2)).byte()
set_label, get_labe = qaml.datasets._embed_labels(mnist_test,encoding='one_hot',
                                                  scale=255,setter_getter=True)
test_loader = torch.utils.data.DataLoader(mnist_test)

# Visualize
fig, axs = plt.subplots(4, 5)
for ax, (img, label) in zip(axs.flat, test_loader):
    ax.matshow(img.squeeze())
    ax.set_title(int(label))
    ax.axis('off')
plt.tight_layout()

################################# Model Definition #############################
VISIBLE_SIZE = DATA_SIZE
HIDDEN_SIZE = 8

# Specify model with dimensions
rbm = qaml.nn.RestrictedBoltzmannMachine(VISIBLE_SIZE,HIDDEN_SIZE,'SPIN')

# Initialize biases
_ = torch.nn.init.uniform_(rbm.b,-4.0,4.0)
_ = torch.nn.init.uniform_(rbm.c,-4.0,4.0)
_ = torch.nn.init.uniform_(rbm.W,-1.0,1.0)

# Set up optimizer
beta = 2.5
optimizer = torch.optim.SGD(rbm.parameters(),lr=learning_rate/beta,
                            weight_decay=weight_decay,
                            momentum=momentum)

# Loss and autograd
CD = qaml.autograd.ContrastiveDivergence

# Set up training mechanisms
qa_sampler = qaml.sampler.QASampler(rbm,solver=solver_name,beta=beta)

# TODO
