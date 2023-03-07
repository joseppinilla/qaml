
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
HIDDEN_SIZE = 8
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
train_loader = torch.utils.data.DataLoader(train_dataset,sampler=train_sampler,batch_size=7)
test_sampler = torch.utils.data.RandomSampler(test_dataset,replacement=False)
test_loader = torch.utils.data.DataLoader(test_dataset,sampler=test_sampler)

bm = qaml.nn.BoltzmannMachine(VISIBLE_SIZE,HIDDEN_SIZE,'SPIN')

_ = torch.nn.init.uniform_(bm.b,-4.0,4.0)
_ = torch.nn.init.uniform_(bm.c,-4.0,4.0)
_ = torch.nn.init.uniform_(bm.vv,-1.0,1.0)
_ = torch.nn.init.uniform_(bm.hh,-1.0,1.0)
_ = torch.nn.init.uniform_(bm.W,-1.0,1.0)


sa_sampler = qaml.sampler.SimulatedAnnealingNetworkSampler(bm)

ex_sampler = qaml.sampler.ExactNetworkSampler(bm)

ML = qaml.autograd.MaximumLikelihood

optimizer = torch.optim.SGD(bm.parameters(),
                            lr=learning_rate,
                            weight_decay=weight_decay,
                            momentum=momentum)

20%10

bm.train()
p_log = []
r_log = []
err_log = []
score_log = []
epoch_err_log = []
accuracy_log = []
b_log = [bm.b.detach().clone().numpy()]
c_log = [bm.c.detach().clone().numpy()]
vv_log = [bm.vv.detach().clone().numpy().flatten()]
hh_log = [bm.hh.detach().clone().numpy().flatten()]
W_log = [bm.W.detach().clone().numpy().flatten()]

for t in range(10):
    epoch_error = torch.Tensor([0.])
    for train_data, train_label in train_loader:

        input_data = torch.cat((train_data,train_label),axis=1)
        # # Positive Phase
        v0, prob_h0 = ex_sampler(input_data,num_reads=20)
        v0, prob_h0 = sa_sampler(input_data,num_reads=20)

        # # Negative Phase
        vk, prob_hk = qa_sampler(num_reads=num_reads)
        # err = ML.apply(sa_sampler,(v0,prob_h0),(vk,prob_hk),*bm.parameters())
        err = ML.apply(qa_sampler,(v0,prob_h0),(vk,prob_hk),*bm.parameters())s
        optimizer.zero_grad()

        # Compute gradients
        err.backward()

        # Update parameters
        optimizer.step()
        epoch_error += err
        err_log.append(err.item())

    b_log.append(bm.b.detach().clone().numpy())
    c_log.append(bm.c.detach().clone().numpy())
    vv_log.append(bm.vv.detach().clone().numpy())
    hh_log.append(bm.hh.detach().clone().numpy())
    W_log.append(bm.W.detach().clone().numpy().flatten())
    epoch_err_log.append(epoch_error.item())
    print(f"Epoch {t} Reconstruction Error = {epoch_error.item()}")
    precision, recall, score = phase_dataset.score((vk[:,:DATA_SIZE]+1)/2)
    p_log.append(precision); r_log.append(recall); score_log.append(score)
    print(f"Precision {precision:.2} Recall {recall:.2} Score {score:.2}")
    ############################## CLASSIFICATION ##################################
    count = 0
    for i,(test_data, test_label) in enumerate(test_loader):
        input_data = torch.cat((test_data.flatten(),test_label))
        if bm.vartype is dimod.SPIN:
            input_data = 2*input_data - 1.0
        data_mask = torch.ones_like(test_data.flatten())
        label_mask = torch.zeros_like(test_label.flatten())
        mask = torch.cat((data_mask,label_mask))
        v_recon,h_recon = qa_sampler.reconstruct(input_data,mask=mask,num_reads=5)

        _,label_pred = v_recon.split((DATA_SIZE,LABEL_SIZE),dim=1)

        if label_pred.mode(0)[0].item() == test_label.item():
            count+=1
    accuracy_log.append(count/len(test_dataset))
    print(f"Testing accuracy: {count}/{len(test_dataset)} ({count/len(test_dataset):.2f})")



torch.save(err_log,f'err_log_{num_reads}.pt')
torch.save(p_log,f'p_log_{num_reads}.pt')
torch.save(r_log,f'r_log_{num_reads}.pt')
torch.save(score_log,f'score_log_{num_reads}.pt')
torch.save(accuracy_log,f'accuracy_log_{num_reads}.pt')
torch.save(epoch_err_log,f'epoch_err_log_{num_reads}.pt')

# Testing accuracy graph
fig, ax = plt.subplots()
ax.plot(accuracy_log)
plt.ylabel("Testing Accuracy")
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
