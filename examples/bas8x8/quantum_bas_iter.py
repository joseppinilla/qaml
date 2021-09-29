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
weight_decay = 0.01
momentum = 0.5

#################################### Input Data ################################
bas_dataset = qaml.datasets.BAS(*SHAPE,embed_label=True,transform=torch.Tensor)
train_dataset,test_dataset = torch.utils.data.random_split(bas_dataset,[*SPLIT])
train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=False,
                                               num_samples=SAMPLES)
train_loader = torch.utils.data.DataLoader(train_dataset,sampler=train_sampler,
                                           batch_size=BATCH_SIZE)

for SEED in [0]:
    ######################################## RNG ###################################
    torch.manual_seed(SEED)
    ################################# Model Definition #############################
    # Specify model with dimensions
    rbm = qaml.nn.RBM(DATA_SIZE,HIDDEN_SIZE)

    # Initialize biases
    torch.nn.init.constant_(rbm.b,-0.5)
    torch.nn.init.zeros_(rbm.c)
    torch.nn.init.uniform_(rbm.W,-0.5,0.5)

    # Set up optimizers
    optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate,
                                weight_decay=weight_decay,momentum=momentum)

    # Set up training mechanisms
    # Trainable inverse temperature with separate optimizer
    beta = torch.nn.Parameter(torch.tensor(4.0), requires_grad=True)
    beta_optimizer = torch.optim.SGD([beta],lr=0.001)
    solver_name = "Advantage_system1.1"
    # qa_sampler = qaml.sampler.QASampler(rbm,solver=solver_name,beta=beta)
    # qa_sampler = qaml.sampler.AdachiQASampler(rbm,solver=solver_name,beta=beta)
    # qa_sampler = qaml.sampler.AdaptiveQASampler(rbm,solver=solver_name,beta=beta)
    qa_sampler = qaml.sampler.RepurposeQASampler(rbm,solver=solver_name,beta=beta)
    qaml.prune.adaptive_unstructured(rbm,'W',qa_sampler)
    # qaml.prune.priority_embedding_unstructured(rbm,'W',qa_sampler,priority=[rbm.V-1,rbm.V-8-1])
    print(f"Edges pruned: {len((rbm.state_dict()['W_mask']==0).nonzero())}")

    # Loss and autograd
    CD = qaml.autograd.SampleBasedConstrastiveDivergence()
    betaGrad = qaml.autograd.AdaptiveBeta()

    ################################## Model Training ##############################
    # Set the model to training mode
    rbm.train()
    err_log = []
    beta_log = []
    scalar_log = []
    err_beta_log = []
    accuracy_log = []
    b_log = [rbm.b.detach().clone().numpy()]
    c_log = [rbm.c.detach().clone().numpy()]
    W_log = [rbm.W.detach().clone().numpy().flatten()]
    for t in range(EPOCHS):
        epoch_error = 0
        epoch_error_beta = 0

        for img_batch, labels_batch in train_loader:
            input_data = img_batch.flatten(1)

            # Negative Phase
            vk, prob_hk = qa_sampler(BATCH_SIZE,auto_scale=True)
            # Positive Phase
            v0, prob_h0 = input_data, rbm(input_data,scale=qa_sampler.scalar*qa_sampler.beta)

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

            err_beta = betaGrad.apply(rbm.energy(v0,prob_h0),rbm.energy(vk,prob_hk),beta)
            beta_optimizer.zero_grad()
            err_beta.backward()
            # beta_optimizer.step()
            # beta.data = torch.clamp(beta,min=1.0)
            epoch_error_beta  += err_beta.item()

        # Error Log
        b_log.append(rbm.b.detach().clone().numpy())
        c_log.append(rbm.c.detach().clone().numpy())
        W_log.append(rbm.W.detach().clone().numpy().flatten())
        err_log.append(epoch_error)
        beta_log.append(beta.item())
        scalar_log.append(qa_sampler.scalar)
        err_beta_log.append(epoch_error_beta)
        print(f"Epoch {t} Reconstruction Error = {epoch_error}")
        print(f"Beta = {qa_sampler.beta}")
        print(f"Beta Error = {epoch_error_beta}")
        print(f"Scaling factor = {qa_sampler.scalar}")
        ############################## CLASSIFICATION ##################################
        count = 0
        mask = torch.ones(1,M,N)
        for test_data, test_label in test_dataset:
            test_data[-2:,-1] = 0.5
            prob_hk = rbm(test_data.flatten(),scale=qa_sampler.scalar*qa_sampler.beta)
            label_pred = rbm.generate(prob_hk,scale=qa_sampler.scalar*qa_sampler.beta).view(*SHAPE)[-2:,-1]
            if label_pred.argmax() == test_label.argmax():
                count+=1
        accuracy_log.append(count/TEST)
        print(f"Testing accuracy: {count}/{TEST} ({count/TEST:.2f})")

    ############################## Logging Directory ###############################
    directory = 'BAS88_beta40_scale_200_Adv_Rep_wd001'
    if not os.path.exists(directory):
            os.makedirs(directory)
    seed = torch.initial_seed()
    if not os.path.exists(f'{directory}/{seed}'):
            os.makedirs(f'{directory}/{seed}')

    ############################ Store Model and Logs ##############################
    torch.save(b_log,f"./{directory}/{seed}/b.pt")
    torch.save(c_log,f"./{directory}/{seed}/c.pt")
    torch.save(W_log,f"./{directory}/{seed}/W.pt")
    torch.save(err_log,f"./{directory}/{seed}/err.pt")
    torch.save(beta_log,f"./{directory}/{seed}/beta.pt")
    torch.save(scalar_log,f"./{directory}/{seed}/scalar.pt")
    torch.save(accuracy_log,f"./{directory}/{seed}/accuracy.pt")
    torch.save(err_beta_log,f"./{directory}/{seed}/err_beta_log.pt")
    torch.save(dict(qa_sampler.embedding),f"./{directory}/{seed}/embedding.pt")
    torch.save(rbm.state_dict()['W_mask'],f"./{directory}/{seed}/mask.pt")
    # torch.save(dict(qa_sampler.embedding_orig),f"./{directory}/{seed}/embedding_orig.pt")
rbm.V
rbm.H
# Scaling factor graph
fig, ax = plt.subplots()
plt.plot(scalar_log)
plt.ylabel("Scaling Factor")
plt.xlabel("Epoch")

# Error Beta graph
fig, ax = plt.subplots()
plt.plot(err_beta_log)
plt.ylabel("Beta Error")
plt.xlabel("Epoch")

# Beta graph
fig, ax = plt.subplots()
plt.plot(beta_log)
plt.ylabel("Beta")
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
