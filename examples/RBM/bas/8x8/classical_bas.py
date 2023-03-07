# # Classical RBM training on the Bars-And-Stripes Dataset for Reconstruction
# This is an example on classical Gibbs training of an RBM on the BAS(4,4)
# dataset.
# Developed by: Jose Pinilla

# Required packages
import qaml
import torch
torch.manual_seed(0) # For deterministic weights

import matplotlib.pyplot as plt
import torchvision.transforms as torch_transforms

################################# Hyperparameters ##############################
M,N = SHAPE = (8,8)
DATA_SIZE = N*M
EPOCHS = 35
BATCH_SIZE = 64
SUBCLASSES = [1,2]

# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

#################################### Input Data ################################
train_dataset = qaml.datasets.BAS(*SHAPE,transform=qaml.datasets.ToSpinTensor())
set_label,get_label = qaml.datasets._embed_labels(train_dataset,
                                                  encoding='binary',
                                                  setter_getter=True)
qaml.datasets._subset_classes(train_dataset,SUBCLASSES)
train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=False)
train_loader = torch.utils.data.DataLoader(train_dataset,BATCH_SIZE,sampler=train_sampler)

# PLot all data
fig,axs = plt.subplots(8,8)
for img_batch,labels_batch in train_loader:
    for ax,img,label in zip(axs.flat,img_batch,labels_batch):
        ax.matshow(img.view(*SHAPE),vmin=0,vmax=1); ax.axis('off')
    break
plt.tight_layout()

################################# Model Definition #############################
VISIBLE_SIZE = DATA_SIZE
HIDDEN_SIZE = 64

# Specify model with dimensions
rbm = qaml.nn.RBM(VISIBLE_SIZE, HIDDEN_SIZE, 'SPIN')

# Initialize biases
_ = torch.nn.init.constant_(rbm.b,0.1)
_ = torch.nn.init.zeros_(rbm.c)
_ = torch.nn.init.uniform_(rbm.W,-0.1,0.1)

# Set up optimizer
optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate,
                            weight_decay=weight_decay,momentum=momentum)

# Set up training mechanisms
gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm,BATCH_SIZE)
CD = qaml.autograd.ContrastiveDivergence

# NUM_CHAINS = 100
# persist_sampler = qaml.sampler.GibbsNetworkSampler(rbm,NUM_CHAINS)

################################## Model Training ##############################
# Set the model to training mode
rbm.train()
p_log = []
r_log = []
err_log = []
score_log = []
kl_div_log = []
b_log = [rbm.b.detach().clone().numpy()]
c_log = [rbm.c.detach().clone().numpy()]
W_log = [rbm.W.detach().clone().numpy().flatten()]
for t in range(35):
    kl_div = torch.Tensor([0.])
    epoch_error = torch.Tensor([0.])
    for img_batch,labels_batch in train_loader:
        input_data = img_batch.flatten(1)

        # Positive Phase
        v0, p_h0 = gibbs_sampler(input_data.detach(), k=0)
        # Negative Phase
        vk, p_hk = gibbs_sampler(v0.detach(),k=1)
        # p_vk, p_hk = persist_sampler(k=5)

        # Reconstruction error from Contrastive Divergence
        err = CD.apply(gibbs_sampler,(v0,p_h0),(p_vk,p_hk), *rbm.parameters())

        # Do not accumulate gradients
        optimizer.zero_grad()

        # Compute gradients
        err.backward()

        # Update parameters
        optimizer.step()

        #Accumulate error for this epoch
        epoch_error  += err
        vk = gibbs_sampler.sample(p_vk)
        kl_div += qaml.perf.free_energy_smooth_kl(rbm,v0,vk)

    # KL-Divergence
    kl_div_log.append(kl_div.item())
    print(f"Epoch {t} KL-Divergence = {kl_div.item()}")
    # Error Log
    b_log.append(rbm.b.detach().clone().numpy())
    c_log.append(rbm.c.detach().clone().numpy())
    W_log.append(rbm.W.detach().clone().numpy().flatten())
    err_log.append(epoch_error.item())
    print(f"Epoch {t} Reconstruction Error = {epoch_error.item()}")
    # BAS score
    precision, recall, score = train_dataset.score(p_vk.bernoulli().view(-1,*SHAPE))
    p_log.append(precision); r_log.append(recall); score_log.append(score)
    print(f"Precision {precision:.2} Recall {recall:.2} Score {score:.2}")

# Set the model to evaluation mode
# rbm.eval()

################################## Sampling ####################################
num_samples = 1000
init = torch.FloatTensor(num_samples, VISIBLE_SIZE).uniform_(-1, +1)
# init =  torch.rand(num_samples,DATA_SIZE)
prob_v,_ = gibbs_sampler(init,k=1)
img_samples = prob_v.view(num_samples,*SHAPE).bernoulli()
# PLot some samples
fig,axs = plt.subplots(4,5)
for ax,img in zip(axs.flat,img_samples):
    ax.matshow(img.view(*SHAPE),vmin=0,vmax=1); ax.axis('off')
plt.tight_layout()
