# # Classical BM training on the Bars-And-Stripes Dataset for Reconstruction
# Developed by: Jose Pinilla

# Required packages
import qaml
import torch
SEED = 0
torch.manual_seed(SEED) # For deterministic weights

import matplotlib.pyplot as plt
import torchvision.transforms as torch_transforms

################################# Hyperparameters ##############################
EPOCHS = 50
M,N = SHAPE = (6,6)
DATA_SIZE = N*M
SUBCLASSES = [1,2]

# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

TRAIN_READS = 10
TEST_READS = 100

#################################### Input Data ################################
train_dataset = qaml.datasets.BAS(*SHAPE,transform=qaml.datasets.ToSpinTensor())
set_label,get_label = qaml.datasets._embed_labels(train_dataset,
                                                  encoding='binary',
                                                  setter_getter=True)
qaml.datasets._subset_classes(train_dataset,SUBCLASSES)
train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=False)
train_loader = torch.utils.data.DataLoader(train_dataset,sampler=train_sampler)

# PLot all data
fig,axs = plt.subplots(4,4)
for ax,(img,label) in zip(axs.flat,train_loader):
    ax.matshow(img.view(*SHAPE),vmin=0,vmax=1); ax.axis('off')
plt.tight_layout()
plt.savefig("dataset.svg")

################################# Model Definition #############################
VISIBLE_SIZE = DATA_SIZE
HIDDEN_SIZE = 16

# Specify model with dimensions
bm = qaml.nn.BM(VISIBLE_SIZE, HIDDEN_SIZE, 'SPIN')

# Set up optimizer
optimizer = torch.optim.SGD(bm.parameters(), lr=learning_rate,
                            weight_decay=weight_decay,momentum=momentum)

# Set up training mechanisms
sa_sampler = qaml.sampler.SimulatedAnnealingNetworkSampler(bm)
ML = qaml.autograd.MaximumLikelihood

################################## Model Training ##############################
# Set the model to training mode
bm.train()
p_log = []
r_log = []
err_log = []
score_log = []
epoch_err_log = []
b_log = [bm.b.detach().clone().numpy()]
c_log = [bm.c.detach().clone().numpy()]
W_log = [bm.W.detach().clone().numpy().flatten()]
vv_log = [bm.vv.detach().clone().numpy().flatten()]
hh_log = [bm.hh.detach().clone().numpy().flatten()]

for t in range(EPOCHS):
    kl_div = torch.Tensor([0.])
    epoch_error = torch.Tensor([0.])
    for img_batch,labels_batch in train_loader:
        input_data = img_batch.view(1,-1)

        # Positive Phase
        v0, h0 = sa_sampler(input_data.detach(),num_reads=TRAIN_READS)
        # Negative Phase
        vk, hk = sa_sampler(num_reads=TRAIN_READS)

        # Reconstruction error from Contrastive Divergence
        err = ML.apply(sa_sampler,(v0,h0),(vk,hk), *bm.parameters())

        # Do not accumulate gradients
        optimizer.zero_grad()

        # Compute gradients
        err.backward()

        # Update parameters
        optimizer.step()

        #Accumulate error for this epoch
        epoch_error  += err
        err_log.append(err.item())

    # Error Log
    b_log.append(bm.b.detach().clone().numpy())
    c_log.append(bm.c.detach().clone().numpy())
    vv_log.append(bm.vv.detach().clone().numpy())
    hh_log.append(bm.hh.detach().clone().numpy())
    W_log.append(bm.W.detach().clone().numpy().flatten())
    # Error Log
    epoch_err_log.append(epoch_error.item())
    print(f"Epoch {t} Reconstruction Error = {epoch_error.item()}")
    # BAS score
    vk,_ = sa_sampler(num_reads=TEST_READS)
    precision, recall, score = train_dataset.score(((vk+1)/2).view(-1,*SHAPE))
    p_log.append(precision); r_log.append(recall); score_log.append(score)
    print(f"Precision {precision:.2} Recall {recall:.2} Score {score:.2}")

torch.save(err_log,f'err_log_{TRAIN_READS}.pt')
torch.save(p_log,f'p_log_{TRAIN_READS}.pt')
torch.save(r_log,f'r_log_{TRAIN_READS}.pt')
torch.save(score_log,f'score_log_{TRAIN_READS}.pt')
torch.save(epoch_err_log,f'epoch_err_log_{TRAIN_READS}.pt')

# Samples
fig,axs = plt.subplots(4,4)
for ax,img in zip(axs.flat,vk):
    ax.matshow(img.view(*SHAPE),vmin=0,vmax=1); ax.axis('off')
plt.tight_layout()
plt.savefig(f"sample_vk_{TRAIN_READS}.svg")

# Precision graph
fig, ax = plt.subplots()
ax.plot(p_log)
plt.ylabel("Precision")
plt.xlabel("Epoch")
plt.savefig(f"precision_{TRAIN_READS}.svg")

# Recall graph
fig, ax = plt.subplots()
ax.plot(r_log)
plt.ylabel("Recall")
plt.xlabel("Epoch")
plt.savefig(f"recall_{TRAIN_READS}.svg")

# Score graph
fig, ax = plt.subplots()
ax.plot(score_log)
plt.ylabel("Score")
plt.xlabel("Epoch")
plt.savefig(f"score_{TRAIN_READS}.svg")

# Iteration Error
fig, ax = plt.subplots()
ax.plot(err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")
plt.savefig(f"sample_vk_{TRAIN_READS}.svg")

# Epoch Error
fig, ax = plt.subplots()
ax.plot(epoch_err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")
plt.savefig(f"sample_vk_{TRAIN_READS}.svg")