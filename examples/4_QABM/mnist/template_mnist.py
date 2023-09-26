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
EPOCHS = 75
M,N = SHAPE = (13,13)
DATA_SIZE = M*N

# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

batch_mode = True
solver_name = "Advantage_system6.1"

SEED = 5
torch.manual_seed(SEED)

################################# Model Definition #############################
VISIBLE_SIZE = DATA_SIZE
HIDDEN_SIZE = 7

# Specify model with dimensions
bm = qaml.nn.BoltzmannMachine(VISIBLE_SIZE,HIDDEN_SIZE,'SPIN')

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
qa_sampler = qaml.sampler.QASampler(bm,solver=solver_name,batch_mode=batch_mode,mask=True)

# Loss and autograd
ML = qaml.autograd.MaximumLikelihood

BATCH_SIZE = qa_sampler.batch_mode
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
fig,axs = plt.subplots(4,5)
for ax,(img,label) in zip(axs.flat,test_loader):
    ax.matshow(img.squeeze())
    ax.set_title(int(label))
    ax.axis('off')
plt.tight_layout()

len(mnist_train.data)

for img,target in train_loader:
    break
len(train_loader)
img.shape

################################## Training Log ################################

directory = f"{method}/{solver_name}/{SEED}"
os.makedirs(f'{directory}',exist_ok=True)

torch.save(bm.to_networkx_graph(),f"{directory}/graph.pt")
torch.save(dict(qa_sampler.embedding),f"{directory}/embedding.pt")
if batch_mode:
    torch.save([dict(e) for e in qa_sampler.batch_embeddings],f"{directory}/batch_embeddings.pt")


err_log = []
kl_div_log = []
scalar_log = []
accuracy_log = []
batch_err_log = []
b_log = [bm.b.detach().clone().numpy()]
c_log = [bm.c.detach().clone().numpy()]
W_log = [bm.W.detach().clone().numpy()]
vv_log = [bm.vv.detach().clone().numpy().flatten()]
hh_log = [bm.hh.detach().clone().numpy().flatten()]

################################## Model Training ##############################
for t in range(5):
    epoch_error = torch.Tensor([0.])

    for img_batch, labels_batch in train_loader:

        # Positive Phase
        v0, prob_h0 = qa_sampler(img_batch.flatten(1),num_reads=100)
        # Negative Phase
        vk, prob_hk = qa_sampler(num_reads=1000,num_spin_reversal_transforms=4)

        # Reconstruction error from Contrastive Divergence
        err = ML.apply(qa_sampler,(v0,prob_h0),(vk,prob_hk),*bm.parameters())

        # Do not accumulate gradients
        optimizer.zero_grad()

        # Compute gradients
        err.backward()

        # Update parameters
        optimizer.step()

        #Accumulate error for this epoch
        epoch_error  += err.item()
        batch_err_log.append(err.item())

        # Error Log
        b_log.append(bm.b.detach().clone().numpy())
        c_log.append(bm.c.detach().clone().numpy())
        vv_log.append(bm.vv.detach().clone().numpy())
        hh_log.append(bm.hh.detach().clone().numpy())
        W_log.append(bm.W.detach().clone().numpy().flatten())

    err_log.append(epoch_error)
    print(f"Epoch {t} Reconstruction Error = {epoch_error}")
