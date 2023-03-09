
# Required packages
import qaml
import dimod
import torch

torch.manual_seed(42) # For deterministic weights

import matplotlib.pyplot as plt
import torchvision.transforms as torch_transforms

################################# Hyperparameters ##############################
EPOCHS = 1
DATA_SIZE = 10
LABEL_SIZE = 1
VISIBLE_SIZE = DATA_SIZE + LABEL_SIZE
HIDDEN_SIZE = 16
BATCH_SIZE = 4
# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

torch.manual_seed(42) # For deterministic weights
#################################### Input Data ################################
phase_dataset = qaml.datasets.PhaseState(DATA_SIZE,labeled=True,
                                         transform=qaml.datasets.ToSpinTensor(),
                                         target_transform=qaml.datasets.ToSpinTensor())

TRAIN = len(phase_dataset)*2//3
SPLIT = TRAIN,len(phase_dataset)-TRAIN
train_dataset,test_dataset = torch.utils.data.random_split(phase_dataset,[*SPLIT])

train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=False)
train_loader = torch.utils.data.DataLoader(train_dataset,sampler=train_sampler,batch_size=BATCH_SIZE)
test_sampler = torch.utils.data.RandomSampler(test_dataset,replacement=False)
test_loader = torch.utils.data.DataLoader(test_dataset,sampler=test_sampler)

rbm = qaml.nn.RestrictedBoltzmannMachine(VISIBLE_SIZE,HIDDEN_SIZE,'SPIN')

_ = torch.nn.init.uniform_(rbm.b,-4.0,4.0)
_ = torch.nn.init.uniform_(rbm.c,-4.0,4.0)
_ = torch.nn.init.uniform_(rbm.W,-1.0,1.0)

gibbs_sampler =  qaml.sampler.GibbsNetworkSampler(rbm,BATCH_SIZE)

CD = qaml.autograd.ContrastiveDivergence

optimizer = torch.optim.SGD(rbm.parameters(),lr=learning_rate,
                            weight_decay=weight_decay,momentum=momentum)

rbm.train()
p_log = []
r_log = []
err_log = []
score_log = []
kl_div_log = []
epoch_err_log = []
b_log = [rbm.b.detach().clone().numpy()]
c_log = [rbm.c.detach().clone().numpy()]
W_log = [rbm.W.detach().clone().numpy().flatten()]

for t in range(50):
    stack_vk = None
    kl_div = torch.Tensor([0.])
    epoch_error = torch.Tensor([0.])
    for train_data, train_label in train_loader:

        input_data = torch.cat((train_data,train_label),axis=1)
        # Positive Phase
        v0, p_h0 = gibbs_sampler(input_data,k=0)
        # Negative Phase
        p_vk, p_hk = gibbs_sampler(input_data,k=5)

        err = CD.apply(gibbs_sampler,(v0,p_h0),(p_vk,p_hk),*rbm.parameters())
        optimizer.zero_grad()

        # Compute gradients
        err.backward()

        # Update parameters
        optimizer.step()
        epoch_error += err
        err_log.append(err.item())
        vk = gibbs_sampler.sample(p_vk)
        kl_div += qaml.perf.free_energy_smooth_kl(rbm,v0,vk)

        # For full score
        stack_vk = vk if stack_vk is None else torch.cat([stack_vk,vk],dim=0)

    # Samples score
    precision, recall, score = phase_dataset.score((stack_vk[:,:DATA_SIZE]+1)/2)
    p_log.append(precision); r_log.append(recall); score_log.append(score)
    print(f"Precision {precision:.2} Recall {recall:.2} Score {score:.2}")
    # Parameter log
    b_log.append(rbm.b.detach().clone().numpy())
    c_log.append(rbm.c.detach().clone().numpy())
    W_log.append(rbm.W.detach().clone().numpy().flatten())
    # KL-Divergence
    kl_div_log.append(kl_div.item())
    print(f"Epoch {t} KL-Divergence = {kl_div.item()}")
    # Error Log
    epoch_err_log.append(epoch_error.item())
    print(f"Epoch {t} Reconstruction Error = {epoch_error.item()}")


torch.save(err_log,f'err_log_{num_reads}.pt')
torch.save(p_log,f'p_log_{num_reads}.pt')
torch.save(r_log,f'r_log_{num_reads}.pt')
torch.save(score_log,f'score_log_{num_reads}.pt')
torch.save(accuracy_log,f'accuracy_log_{num_reads}.pt')
torch.save(epoch_err_log,f'epoch_err_log_{num_reads}.pt')

# KL Divergence
fig, ax = plt.subplots()
ax.plot(kl_div_log)
plt.ylabel("KL Divergence")
plt.xlabel("Epoch")

# Iteration Error
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


# L1 error graph
fig, ax = plt.subplots()
ax.plot(epoch_err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")
