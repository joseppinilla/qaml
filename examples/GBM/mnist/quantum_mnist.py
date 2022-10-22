# # Quantum BM training on the MNIST Dataset for reconstruction and classification
# This is an example on quantum annealing training of an BM on the MNIST
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

################################# Hyperparameters ##############################
EPOCHS = 1
M,N = SHAPE = (13,13)
DATA_SIZE = M*N

# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

################################# Model Definition #############################
VISIBLE_SIZE = DATA_SIZE
HIDDEN_SIZE = 6

# Specify model with dimensions
bm = qaml.nn.BoltzmannMachine(VISIBLE_SIZE,HIDDEN_SIZE,'SPIN')

# Initialize biases
SEED = 8
torch.manual_seed(SEED)
# Initialize biases
_ = torch.nn.init.uniform_(bm.b,-4.0,4.0)
_ = torch.nn.init.uniform_(bm.c,-4.0,4.0)
_ = torch.nn.init.uniform_(bm.vv,-1.0,1.0)
_ = torch.nn.init.uniform_(bm.hh,-1.0,1.0)
_ = torch.nn.init.uniform_(bm.W,-1.0,1.0)

# torch.nn.utils.prune.random_unstructured(bm,'W',0.9)
# torch.nn.utils.prune.random_unstructured(bm,'vv',0.9)
# torch.nn.utils.prune.random_unstructured(bm,'hh',0.9)

# Set up optimizer
optimizer = torch.optim.SGD(bm.parameters(),lr=learning_rate,
                            weight_decay=weight_decay,
                            momentum=momentum)

# Set up training mechanisms
solver_name = "Advantage_system6.1"
qa_sampler = qaml.sampler.QASampler(bm,solver=solver_name,batch_mode=True)

# Loss and autograd
ML = qaml.autograd.MaximumLikelihood

############################ Dataset and Transformations #######################
mnist_train = torch_datasets.MNIST(root='./data/', train=True, download=True,
                                     transform=qaml.datasets.ToSpinTensor())
mnist_train.data = max_pool2d(mnist_train.data.float(),(2,2),padding=-1).byte()
qaml.datasets._embed_labels(mnist_train,axis=1,encoding='one_hot',scale=255)
BATCH_SIZE = qa_sampler.batch_mode
train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=BATCH_SIZE)

mnist_test = torch_datasets.MNIST(root='./data/', train=False, download=True,
                                  transform=qaml.datasets.ToSpinTensor())
mnist_test.data = max_pool2d(mnist_test.data.float(),(2,2),padding=-1).byte()
set_label,get_labe = qaml.datasets._embed_labels(mnist_test,encoding='one_hot',
                                                 scale=255,setter_getter=True)
test_loader = torch.utils.data.DataLoader(mnist_test)


# Visualize
fig,axs = plt.subplots(4,5)
for ax,(img,label) in zip(axs.flat,test_loader):
    ax.matshow(img.squeeze())
    ax.set_title(int(label))
    ax.axis('off')
plt.tight_layout()
