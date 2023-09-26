# # Quantum-Assisted BM training on the BAS Dataset for Reconstruction
# This is an example on quantum-assisted training of an BM on the BAS(8,8)
# dataset.
# Developed by: Jose Pinilla

# Required packages
import os
import qaml
import torch

import matplotlib.pyplot as plt
import torchvision.transforms as torch_transforms

################################# Hyperparameters ##############################
EPOCHS = 5
M,N = SHAPE = (8,8)
DATA_SIZE = M*N
TRAIN, TEST = SPLIT = (360,150) #(8,8)
TEST_SAMPLES = 20
# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

################################# Model Definition #############################
VISIBLE_SIZE = DATA_SIZE
HIDDEN_SIZE = 16

bm = qaml.nn.BoltzmannMachine(VISIBLE_SIZE,HIDDEN_SIZE,'SPIN')

# For deterministic weights
SEED = 42
torch.manual_seed(SEED)

# Set up optimizers
optimizer = torch.optim.SGD(bm.parameters(), lr=learning_rate,
                            weight_decay=weight_decay,momentum=momentum)

# Heuristic embedding
solver_name = "Advantage_system6.2"
device = qaml.sampler.QASampler.get_device(solver=solver_name)
embedding = qaml.minor.miner_heuristic(bm,device,seed=SEED)

# Save
torch.save(embedding,f'./heur_embedding.pt')
torch.save(bm.to_networkx_graph(),'./source.pt')
torch.save(device.to_networkx_graph(),'./target.pt')

# Set up training mechanisms
neg_sampler = qaml.sampler.QASampler(bm,solver=solver_name,embedding=embedding)
pos_sampler = qaml.sampler.BatchQASampler(bm,solver=solver_name,mask=True)

# Loss and autograd
ML = qaml.autograd.MaximumLikelihood

#################################### Input Data ################################
bas_dataset = qaml.datasets.BAS(*SHAPE,transform=qaml.datasets.ToSpinTensor())
set_label,get_label = qaml.datasets._embed_labels(bas_dataset,setter_getter=True)
train_dataset,test_dataset = torch.utils.data.random_split(bas_dataset,[*SPLIT])

BATCH_SIZE = pos_sampler.batch_size
train_sampler = torch.utils.data.RandomSampler(train_dataset,False)
train_loader = torch.utils.data.DataLoader(train_dataset,BATCH_SIZE,sampler=train_sampler)

# Reconstruction sampler from label
mask = set_label(torch.ones(1,*SHAPE),0).flatten()
recon_sampler = qaml.sampler.BatchQASampler(bm,solver=solver_name,mask=mask)

RECON_SIZE = recon_sampler.batch_size
TEST_SAMPLES = RECON_SIZE * (len(test_dataset)//RECON_SIZE)
test_sampler = torch.utils.data.RandomSampler(test_dataset,False,TEST_SAMPLES)
test_loader = torch.utils.data.DataLoader(test_dataset,RECON_SIZE,sampler=test_sampler)

# Visualize
fig,axs = plt.subplots(4,5)
for img_batch,labels_batch in test_loader:
    for ax,img,label in zip(axs.flat,img_batch,labels_batch):
        ax.matshow(img.squeeze())
        ax.set_title(int(label))
        ax.axis('off')
plt.tight_layout()

################################## Model Training ##############################
# Set the model to training mode
bm.train()
p_log = []
r_log = []
score_log = []
err_log = []
accuracy_log = []
batch_err_log = []
b_log = [bm.b.detach().clone().numpy()]
c_log = [bm.c.detach().clone().numpy()]
vv_log = [bm.vv.detach().clone().numpy().flatten()]
hh_log = [bm.hh.detach().clone().numpy().flatten()]
W_log = [bm.W.detach().clone().numpy().flatten()]

for t in range(3):
    epoch_error = 0

    for img_batch, labels_batch in train_loader:
        # Positive Phase

        v0, h0 = pos_sampler(img_batch.flatten(1),num_reads=100)

        # Negative Phase
        vk, hk = neg_sampler(num_reads=1000,num_spin_reversal_transforms=4)

        # Reconstruction error from Contrastive Divergence
        err = ML.apply(neg_sampler,(v0,h0),(vk,hk),*bm.parameters())

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

    # Error report
    err_log.append(epoch_error)
    print(f"Epoch {t} Reconstruction Error = {epoch_error}")
    # Score
    precision, recall, score = bas_dataset.score(((vk+1)/2).view(-1,*SHAPE))
    p_log.append(precision); r_log.append(recall); score_log.append(score)
    print(f"Precision {precision:.2} Recall {recall:.2} Score {score:.2}")

    ############################# CLASSIFICATION ##################################
    count = 0
    for test_data, test_labels in test_loader:
        input_data = test_data.flatten(1)
        mask = set_label(torch.ones(1,*SHAPE),0).flatten()
        v_batch,h_batch = recon_sampler(input_data,mask=mask,num_reads=5)
        for v_recon,v_test in zip(v_batch,test_data):
            label_pred = get_label(v_recon.view(1,*SHAPE))
            label_test = get_label(v_test.view(1,*SHAPE))

            if (torch.argmax(label_pred) == torch.argmax(label_test)):
                count+=1

    accuracy_log.append(count/TEST_SAMPLES)
    print(f"Testing accuracy: {count}/{TEST_SAMPLES} ({count/TEST_SAMPLES:.2f})")

len(accuracy_log)

################################## LOGGING #####################################
directory = f"{HIDDEN_SIZE}_batch{BATCH_SIZE}"
directory = directory.replace('.','')
if not os.path.exists(directory): os.makedirs(directory)
if not os.path.exists(f'{directory}/{SEED}'): os.makedirs(f'{directory}/{SEED}')


################################### SAVE #######################################
torch.save(b_log,f"./{directory}/{SEED}/b.pt")
torch.save(c_log,f"./{directory}/{SEED}/c.pt")
torch.save(W_log,f"./{directory}/{SEED}/W.pt")
torch.save(hh_log,f"./{directory}/{SEED}/hh.pt")
torch.save(vv_log,f"./{directory}/{SEED}/vv.pt")
torch.save(err_log,f"./{directory}/{SEED}/err.pt")
torch.save(batch_err_log,f"./{directory}/{SEED}/batch_err.pt")
torch.save(accuracy_log,f"./{directory}/{SEED}/accuracy.pt")
torch.save(p_log,f"./{directory}/{SEED}/p_log.pt")
torch.save(r_log,f"./{directory}/{SEED}/r_log.pt")
torch.save(score_log,f"./{directory}/{SEED}/score_log.pt")
