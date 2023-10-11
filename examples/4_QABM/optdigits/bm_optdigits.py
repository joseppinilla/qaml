############################## OptDigits RBM Example ############################
# Quantum BM training on the OptDigits Dataset for reconstruction and classification
# This is an example on quantum annealing training of an BM on the OptDigits
# dataset.
# Developed by: Jose Pinilla

# Required packages
import os
import qaml
import torch

import matplotlib.pyplot as plt

################################# Hyperparameters ##############################
EPOCHS = 5
M,N = SHAPE = (8,8)
SUBCLASSES = [0,1,2,3,5,6,7,8]
POS_READS, NEG_READS, TEST_READS = 100, 2500, 20

# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

################################# Model Definition #############################
VISIBLE_SIZE = M*N
HIDDEN_SIZE = 16

# Specify model with dimensions
bm = qaml.nn.BoltzmannMachine(VISIBLE_SIZE,HIDDEN_SIZE,'SPIN')

# For deterministic weights
SEED = 8
torch.manual_seed(SEED)
# Initialize biases
_ = torch.nn.init.uniform_(bm.b,-4.0,4.0)
_ = torch.nn.init.uniform_(bm.c,-4.0,4.0)
_ = torch.nn.init.uniform_(bm.vv,-1.0,1.0)
_ = torch.nn.init.uniform_(bm.hh,-1.0,1.0)
_ = torch.nn.init.uniform_(bm.W,-1.0,1.0)

# Set up optimizer
optimizer = torch.optim.SGD(bm.parameters(),lr=learning_rate,
                            weight_decay=weight_decay,
                            momentum=momentum)

# Set up training mechanisms
SOLVER_NAME = "Advantage_system4.1"
pos_sampler = qaml.sampler.BatchQASampler(bm,solver=SOLVER_NAME,mask=True)
neg_sampler = qaml.sampler.QASampler(bm,solver=SOLVER_NAME)
TRAIN_BATCH = len(pos_sampler.batch_embeddings)

# Loss and autograd
ML = qaml.autograd.MaximumLikelihood

#################################### Input Data ################################
opt_train = qaml.datasets.OptDigits(root='./data/',train=True,download=True,
                                        transform=qaml.datasets.ToSpinTensor())
qaml.datasets._subset_classes(opt_train,SUBCLASSES)
qaml.datasets._embed_labels(opt_train,encoding='one_hot',scale=255)
train_sampler = torch.utils.data.RandomSampler(opt_train,replacement=False)
train_loader = torch.utils.data.DataLoader(opt_train,TRAIN_BATCH,sampler=train_sampler)

#################################### Test Data ################################


opt_test = qaml.datasets.OptDigits(root='./data/',train=False,download=True,
                                       transform=qaml.datasets.ToSpinTensor())
qaml.datasets._subset_classes(opt_test,SUBCLASSES)
set_label,get_label = qaml.datasets._embed_labels(opt_test,encoding='one_hot',
                                                  scale=255,setter_getter=True)
test_sampler = torch.utils.data.RandomSampler(opt_test,False)

mask = set_label(torch.ones(1,1,bm.V),0)
rec_sampler = qaml.sampler.BatchQASampler(bm,solver=SOLVER_NAME,mask=mask)
TEST_BATCH = len(rec_sampler.batch_embeddings)
test_loader = torch.utils.data.DataLoader(opt_test,TEST_BATCH,sampler=test_sampler)

# Visualize
fig,axs = plt.subplots(4,5)
for img_batch,label_batch in test_loader:
    for ax,img,label in zip(axs.flat,img_batch,label_batch):
        ax.matshow(img.squeeze())
        ax.set_title(int(label))
        ax.axis('off')
plt.tight_layout()

################################## Training Log ################################
err_log = []
accuracy_log = []
batch_err_log = []
pos_scalar_log = []
neg_scalar_log = []
b_log = [bm.b.detach().clone().numpy()]
c_log = [bm.c.detach().clone().numpy()]
W_log = [bm.W.detach().clone().numpy().flatten()]
vv_log = [bm.vv.detach().clone().numpy().flatten()]
hh_log = [bm.hh.detach().clone().numpy().flatten()]


pos_sampler.alpha
neg_sampler.alpha
################################## Model Training ##############################
%%time
for t in range(1):
    print(f"Epoch {t}")
    epoch_error = torch.Tensor([0.])
    epoch_kl_div = torch.Tensor([0.])
    for img_batch, labels_batch in train_loader:

        # Positive Phase
        v0, prob_h0 = pos_sampler(img_batch.flatten(1),num_reads=POS_READS)
        # Negative Phase
        vk, prob_hk = neg_sampler(num_reads=NEG_READS)

        # Reconstruction error from Contrastive Divergence
        err = ML.apply(neg_sampler,(v0,prob_h0), (vk,prob_hk), *bm.parameters())

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
    pos_scalar_log.append(pos_sampler.alpha)
    neg_scalar_log.append(neg_sampler.alpha)
    b_log.append(bm.b.detach().clone().numpy())
    c_log.append(bm.c.detach().clone().numpy())
    vv_log.append(bm.vv.detach().clone().numpy())
    hh_log.append(bm.hh.detach().clone().numpy())
    W_log.append(bm.W.detach().clone().numpy().flatten())

    # Error Log
    err_log.append(epoch_error.item())
    print(f"Reconstruction Error = {epoch_error.item()}")

    ############################## CLASSIFICATION ##################################
    count = 0
    for img_batch, label_batch in test_loader:
        input_data = img_batch.flatten(1)
        mask = set_label(torch.ones(1,1,bm.V),0).flatten()
        v_recon,h_recon = rec_sampler(input_data,mask=mask,num_reads=TEST_READS)
        label_pred = get_label(v_recon.view(-1,1,*SHAPE))
        label_true = get_label(img_batch)
        for recon,input in zip(label_pred,label_true):
            if (recon==input).all():
                count+=1
    accuracy_log.append(count/len(opt_test))
    print(f"Testing accuracy: {count}/{len(opt_test)} ({count/len(opt_test):.2f})")


############################ MODEL VISUALIZATION ###############################
directory = f"{HIDDEN_SIZE}_batch{TRAIN_BATCH}"

if not os.path.exists(directory):
        os.makedirs(directory)
if not os.path.exists(f'{directory}/{SEED}'):
        os.makedirs(f'{directory}/{SEED}')

torch.save(b_log,f"./{directory}/{SEED}/b.pt")
torch.save(c_log,f"./{directory}/{SEED}/c.pt")
torch.save(vv_log,f"./{directory}/{SEED}/vv.pt")
torch.save(hh_log,f"./{directory}/{SEED}/hh.pt")
torch.save(W_log,f"./{directory}/{SEED}/W.pt")
torch.save(err_log,f"./{directory}/{SEED}/err.pt")
torch.save(batch_err_log,f"./{directory}/{SEED}/batch_err.pt")
torch.save(pos_scalar_log,f"./{directory}/{SEED}/scalar.pt")
torch.save(neg_scalar_log,f"./{directory}/{SEED}/scalar.pt")
torch.save(accuracy_log,f"./{directory}/{SEED}/accuracy.pt")

fig, ax = plt.subplots()
plt.plot(pos_scalar_log)
plt.ylabel("Positive Alpha")
plt.xlabel("Epoch")

fig, ax = plt.subplots()
plt.plot(neg_scalar_log)
plt.ylabel("Negative Alpha")
plt.xlabel("Epoch")

# Testing accuracy graph
fig, ax = plt.subplots()
plt.plot(accuracy_log)
plt.ylabel("Testing Accuracy")
plt.xlabel("Epoch")
plt.savefig("accuracy.pdf")

# Error graph
fig, ax = plt.subplots()
plt.plot(batch_err_log)
plt.ylabel("Batch Reconstruction Error")
plt.xlabel("Epoch")
plt.savefig("quantum_err.pdf")

# Error graph
fig, ax = plt.subplots()
plt.plot(err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")
plt.savefig("quantum_err.pdf")
