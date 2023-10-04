############################## OptDigits RBM Example ############################
# Quantum BM training on the OptDigits Dataset for reconstruction and classification
# This is an example on quantum annealing training of an BM on the OptDigits
# dataset.
# Developed by: Jose Pinilla

# Required packages
import os
import qaml
import torch
# For deterministic weights
SEED = 52
torch.manual_seed(SEED)

import matplotlib.pyplot as plt


################################# Hyperparameters ##############################
EPOCHS = 75
M,N = SHAPE = (8,8)
SUBCLASSES = [0,1,2,3,5,6,7,8]


# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

POS_READS =10
NEG_READS = 10000
TEST_READS = 100

################################# Model Definition #############################
VISIBLE_SIZE = M*N
HIDDEN_SIZE = 16
BATCH_SIZE = 32

# Specify model with dimensions
bm = qaml.nn.BoltzmannMachine(VISIBLE_SIZE,HIDDEN_SIZE,'SPIN')

# Set up optimizer
optimizer = torch.optim.SGD(bm.parameters(),lr=learning_rate,
                            weight_decay=weight_decay,
                            momentum=momentum)

# Set up training mechanisms
sa_sampler =  qaml.sampler.SimulatedAnnealingNetworkSampler(bm)

# Loss and autograd
ML = qaml.autograd.MaximumLikelihood

#################################### Input Data ################################
opt_train = qaml.datasets.OptDigits(root='./data/',train=True,download=True,
                                        transform=qaml.datasets.ToSpinTensor())
qaml.datasets._subset_classes(opt_train,SUBCLASSES)
qaml.datasets._embed_labels(opt_train,encoding='one_hot',scale=255)
train_sampler = torch.utils.data.RandomSampler(opt_train,replacement=False)
train_loader = torch.utils.data.DataLoader(opt_train,BATCH_SIZE,sampler=train_sampler)


opt_test = qaml.datasets.OptDigits(root='./data/',train=False,download=True,
                                       transform=qaml.datasets.ToSpinTensor())
qaml.datasets._subset_classes(opt_test,SUBCLASSES)
set_label,get_label = qaml.datasets._embed_labels(opt_test,encoding='one_hot',
                                                  scale=255,setter_getter=True)
test_sampler = torch.utils.data.RandomSampler(opt_test,False)
test_loader = torch.utils.data.DataLoader(opt_test,BATCH_SIZE,sampler=test_sampler)

# Visualize
fig,axs = plt.subplots(4,5)
for ax,(img_batch,labels_batch) in zip(axs.flat,test_loader):
    for img, label in zip(img_batch,labels_batch):
        ax.matshow(img.squeeze()); ax.axis('off')
plt.tight_layout()
plt.savefig("dataset.svg")

################################## Training Log ################################
err_log = []
scalar_log = []
accuracy_log = []
batch_err_log = []
b_log = [bm.b.detach().clone().numpy()]
c_log = [bm.c.detach().clone().numpy()]
W_log = [bm.W.detach().clone().numpy().flatten()]
vv_log = [bm.vv.detach().clone().numpy().flatten()]
hh_log = [bm.hh.detach().clone().numpy().flatten()]

################################## Model Training ##############################
for t in range(1):
    print(f"Epoch {t}")
    epoch_error = torch.Tensor([0.])
    for img_batch, labels_batch in train_loader:

        # Positive Phase
        v0, h0 = sa_sampler(img_batch.flatten(1),num_reads=POS_READS)
        # Negative Phase
        vk, hk = sa_sampler(num_reads=NEG_READS)

        # Reconstruction error from Contrastive Divergence
        err = ML.apply(sa_sampler,(v0,h0), (vk,hk), *bm.parameters())

        # Do not accumulate gradients
        optimizer.zero_grad()

        # Compute gradients
        err.backward()

        # Update parameters
        optimizer.step()

        #Accumulate error for this epoch
        epoch_error  += err
        batch_err_log.append(err.item())

    # Parameter log
    b_log.append(bm.b.detach().clone().numpy())
    c_log.append(bm.c.detach().clone().numpy())
    vv_log.append(bm.vv.detach().clone().numpy())
    hh_log.append(bm.hh.detach().clone().numpy())
    W_log.append(bm.W.detach().clone().numpy().flatten())

    # Error Log
    err_log.append(epoch_error.item())
    print(f"Reconstruction Error = {epoch_error.item()}")

# VISIBLE = 64
# HIDDEN = 16
# Epoch 0
# Reconstruction Error = 3143.8515625
# Wall time: 21min 20s

# Samples
vk,_ = sa_sampler(num_reads=TEST_READS)
fig,axs = plt.subplots(10,5)
for ax,img in zip(axs.flat,vk):
    ax.matshow(img.view(*SHAPE))
    ax.axis('off')
plt.tight_layout()

# Classification
count = 0
tests = len(opt_test)
for test_data, test_label in opt_test:
    input_data =  test_data.flatten(1)
    mask = set_label(torch.ones_like(test_data),0).flatten()
    v_recon,h_recon = sa_sampler(input_data,mask=mask,k=5)
    label_pred = get_label(v_recon.view(-1,*SHAPE))
    if label_pred.argmax() == get_label(test_data).argmax():
        count+=1
accuracy_log.append(count/tests)
print(f"Testing accuracy: {count}/{tests} ({count/tests:.2f})")
