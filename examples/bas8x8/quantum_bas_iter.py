import os
import qaml
import torch

import matplotlib.pyplot as plt
import torchvision.transforms as torch_transforms

################################# Hyperparameters ##############################
M,N = SHAPE = (8,8)
DATA_SIZE = N*M
HIDDEN_SIZE = 64
EPOCHS = 200
SAMPLES = None
BATCH_SIZE = 400
TRAIN,TEST = SPLIT = 400,110
# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 0.001
momentum = 0.5

#################################### Input Data ################################
bas_dataset = qaml.datasets.BAS(*SHAPE,embed_label=True,transform=torch.Tensor)
train_dataset,test_dataset = torch.utils.data.random_split(bas_dataset,[*SPLIT])
train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=False,
                                               num_samples=SAMPLES)
train_loader = torch.utils.data.DataLoader(train_dataset,sampler=train_sampler,
                                           batch_size=BATCH_SIZE)

beta = 2.5
weight_init = 0.1
auto_scale = False
num_reads = BATCH_SIZE

sampler_type = 'Adv'
solver_name = "Advantage_system4.1"

directory = f"BAS88_beta{beta}_{weight_init}"
directory += f"{'' if auto_scale else 'no'}scale_"
directory += f"{EPOCHS}_{'batch' if num_reads==BATCH_SIZE else num_reads}"
directory += f"{sampler_type}_wd{weight_decay}"
directory = directory.replace('.','')
print(directory)

if not os.path.exists(f"./{solver_name}/{sampler_type}/"):
    os.makedirs(f"./{solver_name}/{sampler_type}/")

for SEED in [0]:
    ######################################## RNG ###################################
    torch.manual_seed(SEED)
    ############################## Logging Directory ###############################
    if not os.path.exists(directory):
            os.makedirs(directory)
    if not os.path.exists(f'{directory}/{SEED}'):
            os.makedirs(f'{directory}/{SEED}')
    ################################# Model Definition #############################
    # Specify model with dimensions
    rbm = qaml.nn.RBM(DATA_SIZE,HIDDEN_SIZE)
    # Set up optimizers
    optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate,
                                weight_decay=weight_decay,momentum=momentum)

    # Set up training mechanisms
    if sampler_type == 'Adv':
        qa_sampler = qaml.sampler.QASampler(rbm,solver=solver_name,beta=beta)
    elif sampler_type == 'Adachi':
        qa_sampler = qaml.sampler.AdachiQASampler(rbm,solver=solver_name,beta=beta)
        qaml.prune.adaptive_unstructured(rbm,'W',qa_sampler)
        print(f"Edges pruned: {len((rbm.state_dict()['W_mask']==0).nonzero())}")
        torch.save(rbm.state_dict()['W_mask'],f"./{solver_name}/{sampler_type}/mask.pt")
    elif sampler_type == 'Adapt':
        qa_sampler = qaml.sampler.AdaptiveQASampler(rbm,solver=solver_name,beta=beta)
        qaml.prune.adaptive_unstructured(rbm,'W',qa_sampler)
        print(f"Edges pruned: {len((rbm.state_dict()['W_mask']==0).nonzero())}")
        torch.save(rbm.state_dict()['W_mask'],f"./{directory}/{SEED}/mask.pt")
    elif sampler_type == 'Prio':
        qa_sampler = qaml.sampler.AdaptiveQASampler(rbm,solver=solver_name,beta=beta)
        qaml.prune.priority_embedding_unstructured(rbm,'W',qa_sampler,priority=[rbm.V-1,rbm.V-8-1])
        print(f"Edges pruned: {len((rbm.state_dict()['W_mask']==0).nonzero())}")
        torch.save(rbm.state_dict()['W_mask'],f"./{directory}/{SEED}/mask.pt")
    elif sampler_type == 'Rep':
        qa_sampler = qaml.sampler.RepurposeQASampler(rbm,solver=solver_name,beta=beta)
        qaml.prune.adaptive_unstructured(rbm,'W',qa_sampler)
        print(f"Edges pruned: {len((rbm.state_dict()['W_mask']==0).nonzero())}")
        torch.save(rbm.state_dict()['W_mask'],f"./{directory}/{SEED}/mask.pt")

    # Loss and autograd
    CD = qaml.autograd.SampleBasedConstrastiveDivergence()

    # Initialize biases
    _ = torch.nn.init.uniform_(rbm.b,-weight_init,weight_init)
    _ = torch.nn.init.uniform_(rbm.c,-weight_init,weight_init)
    _ = torch.nn.init.uniform_(rbm.W,-weight_init,weight_init)

    ################################## Model Training ##############################
    # Set the model to training mode
    rbm.train()
    err_log = []
    scalar_log = []
    accuracy_log = []
    b_log = [rbm.b.detach().clone().numpy()]
    c_log = [rbm.c.detach().clone().numpy()]
    W_log = [rbm.W.detach().clone().numpy().flatten()]
    for t in range(50):
        epoch_error = 0
        for img_batch, labels_batch in train_loader:
            input_data = img_batch.flatten(1)

            # Negative Phase
            vk, prob_hk = qa_sampler(num_reads=num_reads,auto_scale=auto_scale,num_spin_reversal_transforms=4)
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
        # scalar_log.append(qa_sampler.scalar)
        print(f"Epoch {t} Reconstruction Error = {epoch_error}")
        ############################## CLASSIFICATION ##################################
        count = 0
        for test_data, test_label in test_dataset:
            test_data[-2:,-1] = 0.5
            recon_hk = rbm(test_data.flatten(),scale=scale)
            label_pred = rbm.generate(recon_hk,scale=scale).view(*SHAPE)[-2:,-1]
            if label_pred.argmax() == test_label.argmax():
                count+=1
        accuracy_log.append(count/TEST)
        print(f"Testing accuracy: {count}/{TEST} ({count/TEST:.2f})")

    ############################ Store Model and Logs ##############################
    torch.save(b_log,f"./{directory}/{SEED}/b.pt")
    torch.save(c_log,f"./{directory}/{SEED}/c.pt")
    torch.save(W_log,f"./{directory}/{SEED}/W.pt")
    torch.save(err_log,f"./{directory}/{SEED}/err.pt")
    torch.save(scalar_log,f"./{directory}/{SEED}/scalar.pt")
    torch.save(accuracy_log,f"./{directory}/{SEED}/accuracy.pt")


# Scaling factor graph
fig, ax = plt.subplots()
plt.plot(scalar_log)
plt.ylabel("Scaling Factor")
plt.xlabel("Epoch")

# Testing accuracy graph
fig, ax = plt.subplots()
plt.plot(accuracy_log)
plt.ylabel("Testing Accuracy")
plt.xlabel("Epoch")

# Visible bias graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',DATA_SIZE).colors))
lc_v = ax.plot(b_log)
plt.legend(iter(lc_v),[f'b{i}' for i in range(DATA_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Visible Biases")
plt.xlabel("Epoch")

# Hidden bias graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',HIDDEN_SIZE).colors))
lc_h = plt.plot(c_log)
plt.legend(lc_h,[f'c{i}' for i in range(HIDDEN_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Hidden Biases")
plt.xlabel("Epoch")

# Weights graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',rbm.V*rbm.H).colors))
lc_w = plt.plot(W_log)
plt.ylabel("Weights")
plt.xlabel("Epoch")


# %%
################################## ENERGY ######################################
data_energies = []
for img,label in bas_dataset:
    data = img.flatten()
    data_energies.append(rbm.energy(data,rbm(data)).item())

rand_energies = []
rand_data = torch.rand(len(bas_dataset)*100,rbm.V)
rand_hidd = torch.rand(len(bas_dataset)*100,rbm.H)
for img,hid in zip(rand_data,rand_hidd):
    rand_energies.append(rbm.energy(img.bernoulli(),hid.bernoulli()).item())

gibbs_energies = []
gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm,beta=scale)
for img,label in bas_dataset:
    data = img.flatten()
    prob_v,prob_h = gibbs_sampler(data,k=5)
    gibbs_energies.append(rbm.energy(prob_v.bernoulli(),prob_h.bernoulli()).item())

qa_energies = []
qa_sampleset = qa_sampler(num_reads=BATCH_SIZE,auto_scale=auto_scale,num_spin_reversal_transforms=4)
for s_v,s_h in zip(*qa_sampleset):
    qa_energies.append(rbm.energy(s_v.detach(),s_h.detach()).item())

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
