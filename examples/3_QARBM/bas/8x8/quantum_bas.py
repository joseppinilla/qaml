# # Quantum-Assisted RBM training on the BAS Dataset for Reconstruction
# This is an example on quantum-assisted training of an RBM on the BAS(4,4)
# dataset.
# Developed by: Jose Pinilla

# Required packages
import qaml
import torch
torch.manual_seed(0) # For deterministic weights

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as torch_transforms


################################# Hyperparameters ##############################
M,N = SHAPE = (8,8)
DATA_SIZE = N*M
HIDDEN_SIZE = 64
EPOCHS = 200
SAMPLES = None
BATCH_SIZE = 400
TRAIN,TEST = SPLIT = 350,160
# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

#################################### Input Data ################################
bas_dataset = qaml.datasets.BAS(*SHAPE,embed_label=True,transform=torch.Tensor)
train_dataset,test_dataset = torch.utils.data.random_split(bas_dataset,[*SPLIT])
train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=False,
                                               num_samples=SAMPLES)
train_loader = torch.utils.data.DataLoader(train_dataset,sampler=train_sampler,
                                           batch_size=BATCH_SIZE)

# PLot all data
fig,axs = plt.subplots(6,5)
for ax,(img,label) in zip(axs.flat,train_dataset):
    ax.imshow(img.view(*SHAPE),interpolation='nearest'); ax.axis('off')
plt.tight_layout()
plt.savefig('BAS 8x8.svg')

# %%
################################# Model Definition #############################
# Specify model with dimensions
beta = 2.5
rbm = qaml.nn.RBM(DATA_SIZE,HIDDEN_SIZE)

# Initialize biases
weight_init = 4.0
_ = torch.nn.init.zeros_(rbm.b)
_ = torch.nn.init.zeros_(rbm.c)
_ = torch.nn.init.uniform_(rbm.W,-weight_init,weight_init)

# Set up optimizers
optimizer = torch.optim.SGD(rbm.parameters(),lr=learning_rate,
                            weight_decay=weight_decay,momentum=momentum)

# Set up training mechanisms
auto_scale = True
solver_name = "Advantage_system4.1"
qa_sampler = qaml.sampler.QASampler(rbm,solver=solver_name,beta=beta)

# Loss and autograd
CD = qaml.autograd.SampleBasedConstrastiveDivergence()

# %%
################################## Model Training ##############################
# Set the model to training mode
rbm.train()
err_log = []
scalar_log = []
accuracy_log = []
b_log = [rbm.b.detach().clone().numpy()]
c_log = [rbm.c.detach().clone().numpy()]
W_log = [rbm.W.detach().clone().numpy().flatten()]
for t in range(200):
    epoch_error = 0
    for img_batch, labels_batch in train_loader:
        input_data = img_batch.flatten(1)

        # Negative Phase
        vk, prob_hk = qa_sampler(BATCH_SIZE,auto_scale=auto_scale,num_spin_reversal_transforms=2)
        # Positive Phase
        scale = qa_sampler.scalar*qa_sampler.beta if auto_scale else 1.0
        v0, prob_h0 = input_data, rbm(input_data,scale=scale)

        # Reconstruction error from Contrastive Divergence
        err = CD.apply((v0,prob_h0), (vk,prob_hk), *rbm.parameters())

        # Do not accumulate gradients
        optimizer.zero_grad()

        # Compute gradients
        err.backward()

        # Update parameters
        optimizer.step()

        #Accumulate error for this epoch
        epoch_error  += err.item()

    # Error Log
    b_log.append(rbm.b.detach().clone().numpy())
    c_log.append(rbm.c.detach().clone().numpy())
    W_log.append(rbm.W.detach().clone().numpy().flatten())
    err_log.append(epoch_error)
    scalar_log.append(qa_sampler.scalar)
    print(f"Epoch {t} Reconstruction Error = {epoch_error}")
    ############################## CLASSIFICATION ##################################
    count = 0
    for test_data, test_label in test_dataset:
        test_data[-2:,-1] = 0.5
        test_hid = rbm(test_data.flatten(),scale=1.0)
        label_pred = rbm.generate(test_hid,scale=1.0).view(*SHAPE)[-2:,-1]
        if label_pred.argmax() == test_label.argmax():
            count+=1
    accuracy_log.append(count/TEST)
    print(f"Testing accuracy: {count}/{TEST} ({count/TEST:.2f})")





# %%
############################## RECONSTRUCTION ##################################
scale = 1.0
bce = []
gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm,beta=scale)
mask = torch_transforms.functional.erase(torch.ones(1,M,N),4,4,4,4,0).flatten()
for img, label in train_dataset:
    input_data = img.flatten()
    prob_vk,prob_hk = gibbs_sampler.reconstruct(input_data,k=1,mask=mask)
    bce.append(F.binary_cross_entropy(input_data,prob_vk).item())
_ = plt.hist(bce,bins=100)

# %%
############################ MODEL VISUALIZATION ###############################

# Scalar graph
fig, ax = plt.subplots()
plt.plot(scalar_log)
plt.ylabel("Scaling factor")
plt.xlabel("Epoch")
plt.savefig("quantum_scalar.pdf")

# Testing accuracy graph
fig, ax = plt.subplots()
plt.plot(accuracy_log)
plt.ylabel("Testing Accuracy")
plt.xlabel("Epoch")
plt.savefig("quantum_accuracy.pdf")

# L1 error graph
fig, ax = plt.subplots()
plt.plot(err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")
plt.savefig("quantum_err_log.pdf")

# Visible bias graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',DATA_SIZE).colors))
lc_v = ax.plot(b_log)
plt.legend(iter(lc_v),[f'b{i}' for i in range(DATA_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Visible Biases")
plt.xlabel("Epoch")
plt.savefig("quantum_b_log.pdf")

# Hidden bias graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',HIDDEN_SIZE).colors))
lc_h = plt.plot(c_log)
plt.legend(lc_h,[f'c{i}' for i in range(HIDDEN_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Hidden Biases")
plt.xlabel("Epoch")
plt.savefig("quantum_c_log.pdf")

# Weights graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',rbm.V*rbm.H).colors))
lc_w = plt.plot(W_log)
plt.ylabel("Weights")
plt.xlabel("Epoch")
plt.savefig("quantum_W_log.pdf")

# %%
################################## ENERGY ######################################
data_energies = []
for img,label in bas_dataset:
    data = img.flatten()
    data_energies.append(rbm.free_energy(data).item())

rand_energies = []
rand_data = torch.rand(len(bas_dataset)*100,rbm.V)
for img in rand_data:
    rand_energies.append(rbm.free_energy(img.bernoulli()).item())

gibbs_energies = []
gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm,beta=1.0)
for img,label in bas_dataset:
    data = img.flatten()
    prob_v,prob_h = gibbs_sampler(data,k=5)
    gibbs_energies.append(rbm.free_energy(prob_v.bernoulli()).item())

qa_energies = []
qa_sampleset = qa_sampler(num_reads=BATCH_SIZE,auto_scale=True,num_spin_reversal_transforms=2)
for s_v,s_h in zip(*qa_sampleset):
    qa_energies.append(rbm.free_energy(s_v.detach()).item())

plot_data = [(data_energies,  'Data',    'blue'),
             (rand_energies,  'Random',  'red'),
             (gibbs_energies, 'Gibbs-5', 'green'),
             (qa_energies,    'Quantum', 'orange')]

hist_kwargs = {'ec':'k','lw':2.0,'alpha':0.5,'histtype':'stepfilled','bins':100}
weights = lambda data: [1./len(data) for _ in data]

fig, ax = plt.subplots(figsize=(15,10))
for data,name,color in plot_data:
    ax.hist(data,weights=weights(data),label=name,color=color,**hist_kwargs)

plt.xlabel("Energy")
plt.ylabel("Count/Total")
plt.legend(loc='upper right')
plt.savefig("quantum_energies.pdf")

# %%
################################## VISUALIZE ###################################
plt.matshow(rbm.b.detach().view(*SHAPE))
plt.colorbar()
plt.savefig("quantum_b.pdf")
plt.matshow(rbm.c.detach().view(1,HIDDEN_SIZE))
plt.yticks([])
plt.colorbar()
plt.savefig("quantum_c.pdf")

fig,axs = plt.subplots(HIDDEN_SIZE//4,4)
for i,ax in enumerate(axs.flat):
    weight_matrix = rbm.W[i].detach().view(*SHAPE)
    ms = ax.matshow(weight_matrix)
    ax.axis('off')
fig.subplots_adjust(wspace=0.1, hspace=0.1)
cbar = fig.colorbar(ms, ax=axs.ravel().tolist(), shrink=0.95)
plt.savefig("quantum_weights.pdf")

# %%
########################### Check parameter range ##############################
h_range = qa_sampler.sampler.properties['h_range']
J_range = qa_sampler.sampler.properties['extended_j_range']
target_ising = qa_sampler.embed_bqm(chain_strength=0.8)
linear = target_ising.linear.values()
quad = target_ising.quadratic.values()
print(f"Linear range [{min(linear):.2} <> {max(linear):.2}] @ device={h_range}")
print(f"Quadratic range [{min(quad):.2} <> {max(quad):.2}] @ device={J_range}")
