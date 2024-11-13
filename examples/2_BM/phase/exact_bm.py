# # Quantum-Assisted BM training on the Phase State Dataset for Reconstruction
# Developed by: Jose Pinilla

# Required packages
import os
import qaml
import torch
SEED = 42
torch.manual_seed(SEED) # For deterministic weights

import matplotlib.pyplot as plt
import torchvision.transforms as torch_transforms

################################# Hyperparameters ##############################
EPOCHS = 100
SHAPE = (1,9)
DATA_SIZE = 8
LABEL_SIZE = 1
# Stochastic Gradient Descent
learning_rate = 0.01
weight_decay = 1e-4
momentum = 0.5

TRAIN_READS = 10
TEST_READS = 100

#################################### Input Data ################################
phase_dataset = qaml.datasets.PhaseState(DATA_SIZE,labeled=True,
                                         transform=qaml.datasets.ToSpinTensor(),
                                         target_transform=qaml.datasets.ToSpinTensor())
TRAIN = len(phase_dataset)*2//3
SPLIT = TRAIN,len(phase_dataset)-TRAIN
train_dataset,test_dataset = torch.utils.data.random_split(phase_dataset,[*SPLIT])

train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=False)
train_loader = torch.utils.data.DataLoader(train_dataset,sampler=train_sampler)
test_sampler = torch.utils.data.RandomSampler(test_dataset,replacement=False)
test_loader = torch.utils.data.DataLoader(test_dataset,sampler=test_sampler)

# Visualize
fig,axs = plt.subplots(4,5)
for ax,(img,label) in zip(axs.flat,train_loader):
    ax.matshow(img.view(1,-1))
    ax.set_title(int(label))
    ax.axis('off')
plt.tight_layout()
plt.savefig("dataset.svg")

################################# Model Definition #############################
VISIBLE_SIZE = DATA_SIZE + LABEL_SIZE
HIDDEN_SIZE = 8
bm = qaml.nn.BoltzmannMachine(VISIBLE_SIZE,HIDDEN_SIZE,'SPIN')

# Set up training mechanisms
pos_sampler = qaml.sampler.ExactNetworkSampler(bm)
neg_sampler = qaml.sampler.ExactNetworkSampler(bm)

# Loss and autograd
ML = qaml.autograd.MaximumLikelihood
# Set up optimizers
optimizer = torch.optim.SGD(bm.parameters(),lr=learning_rate,
                            weight_decay=weight_decay,momentum=momentum)

################################## Model Training ##############################
bm.train()
p_log = []
r_log = []
err_log = []
score_log = []
epoch_err_log = []
b_log = [bm.b.detach().clone().numpy()]
c_log = [bm.c.detach().clone().numpy()]
vv_log = [bm.vv.detach().clone().numpy().flatten()]
hh_log = [bm.hh.detach().clone().numpy().flatten()]
W_log = [bm.W.detach().clone().numpy().flatten()]

for t in range(EPOCHS):
    epoch_error = torch.Tensor([0.])
    for train_data, train_label in train_loader:

        input_data = torch.cat((train_data,train_label),axis=1)
        # Positive Phase
        v0, h0 = pos_sampler(input_data.flatten(),num_reads=TRAIN_READS)

        # Negative Phase
        vk, hk = neg_sampler(num_reads=TRAIN_READS)

        # Compute error
        err = ML.apply(neg_sampler,(v0,h0),(vk,hk),*bm.parameters())
        optimizer.zero_grad()

        # Compute gradients
        err.backward()

        # Update parameters
        optimizer.step()

        epoch_error += err
        err_log.append(err.item())

    # Parameter log
    b_log.append(bm.b.detach().clone().numpy())
    c_log.append(bm.c.detach().clone().numpy())
    vv_log.append(bm.vv.detach().clone().numpy())
    hh_log.append(bm.hh.detach().clone().numpy())
    W_log.append(bm.W.detach().clone().numpy().flatten())
    # Error Log
    epoch_err_log.append(epoch_error.item())
    print(f"Epoch {t} Reconstruction Error = {epoch_error.item()}")
    # Samples score
    vk,_ = neg_sampler(num_reads=TEST_READS)
    precision, recall, score = phase_dataset.score((vk[:,:DATA_SIZE]+1)/2)
    p_log.append(precision); r_log.append(recall); score_log.append(score)
    print(f"Precision {precision:.2} Recall {recall:.2} Score {score:.2}")


if not os.path.exists(f'{SEED}'):
        os.makedirs(f'{SEED}')

torch.save(b_log,f"{SEED}/b.pt")
torch.save(c_log,f"{SEED}/c.pt")
torch.save(vv_log,f"{SEED}/vv.pt")
torch.save(hh_log,f"{SEED}/hh.pt")
torch.save(p_log,f'{SEED}/p_log.pt')
torch.save(r_log,f'{SEED}/r_log.pt')
torch.save(err_log,f'{SEED}/err_log.pt')
torch.save(score_log,f'{SEED}/score_log.pt')
torch.save(epoch_err_log,f'{SEED}/epoch_err_log.pt')

# Samples
fig,axs = plt.subplots(4,4)
for ax,img in zip(axs.flat,vk):
    ax.matshow(img.view(*SHAPE),vmin=0,vmax=1); ax.axis('off')
plt.tight_layout()
plt.savefig(f"sample_vk_{TRAIN_READS}.svg")

# Iteration Error graph
fig, ax = plt.subplots()
ax.plot(err_log)
plt.ylabel("Iteration Error")
plt.xlabel("Epoch")

# Precision graph
fig, ax = plt.subplots()
ax.plot(p_log)
plt.ylabel("Precision")
plt.xlabel("Epoch")

# Recall graph
fig, ax = plt.subplots()
ax.plot(r_log)
plt.ylabel("Recall")
plt.xlabel("Epoch")

# Score graph
fig, ax = plt.subplots()
ax.plot(score_log)
plt.ylabel("Score")
plt.xlabel("Epoch")

# Epoch error graph
fig, ax = plt.subplots()
ax.plot(epoch_err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")
