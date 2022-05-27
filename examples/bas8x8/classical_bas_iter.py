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
weight_decay = 1e-4
momentum = 0.5

#################################### Input Data ################################
bas_dataset = qaml.datasets.BAS(*SHAPE,embed_label=True,transform=torch.Tensor)
train_dataset,test_dataset = torch.utils.data.random_split(bas_dataset,[*SPLIT])
train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=False,
                                               num_samples=SAMPLES)
train_loader = torch.utils.data.DataLoader(train_dataset,sampler=train_sampler,
                                           batch_size=BATCH_SIZE)

beta=1.0
weight_inits = [0.1,0.5,1.0,4.0]
for weight_init in weight_inits:
    directory = f'BAS88_bcW{weight_init}_classical5_lr05beta10_200_wd00001'
    directory = directory.replace('.','')
    print(directory)
    for SEED in [0,2,7,8,42]:
        ######################################## RNG ###################################
        torch.manual_seed(SEED)
        ############################## Logging Directory ###############################
        if not os.path.exists(directory):
                os.makedirs(directory)
        if not os.path.exists(f'{directory}/{SEED}'):
                os.makedirs(f'{directory}/{SEED}')
        ################################# Model Definition #############################
        # Specify model with dimensions
        rbm = qaml.nn.RBM(DATA_SIZE, HIDDEN_SIZE)

        # Initialize biases
        # _ = torch.nn.init.constant_(rbm.b,0.5)
        # _ = torch.nn.init.zeros_(rbm.c)
        _ = torch.nn.init.uniform_(rbm.b,-weight_init,weight_init)
        _ = torch.nn.init.uniform_(rbm.c,-weight_init,weight_init)
        _ = torch.nn.init.uniform_(rbm.W,-weight_init,weight_init)

        # Set up optimizer
        optimizer = torch.optim.SGD(rbm.parameters(),lr=learning_rate,
                                    weight_decay=weight_decay,momentum=momentum)

        # Set up training mechanisms
        gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm,beta=beta)
        CD = qaml.autograd.SampleBasedConstrastiveDivergence()

        # mask = torch.load("./Advantage_system4.1/Adachi/mask.pt")
        # mask = torch.load("./Advantage_system4.1/Adapt/mask.pt")
        # mask = torch.load("./Advantage_system4.1/Prio/mask.pt")
        # mask = torch.load("./Advantage_system4.1/Rep/mask.pt")
        # torch.nn.utils.prune.custom_from_mask(rbm,'W',mask)
        # print(f"Edges pruned: {len((rbm.state_dict()['W_mask']==0).nonzero())}")

        ################################## Model Training ##############################
        # Set the model to training mode
        rbm.train()
        err_log = []
        accuracy_log = []
        b_log = [rbm.b.detach().clone().numpy()]
        c_log = [rbm.c.detach().clone().numpy()]
        W_log = [rbm.W.detach().clone().numpy().flatten()]
        for t in range(EPOCHS):
            epoch_error = 0
            for img_batch, labels_batch in train_loader:
                input_data = img_batch.flatten(1)

                # Positive Phase
                v0, prob_h0 = input_data, rbm(input_data,scale=beta)
                # Negative Phase
                vk, prob_hk = gibbs_sampler(v0.detach(), k=5)

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
            print(f"Epoch {t} Reconstruction Error = {epoch_error}")
            ############################## CLASSIFICATION ##################################
            count = 0
            for test_data, test_label in test_dataset:
                test_data[-2:,-1] = 0.5
                recon_hk = rbm(test_data.flatten(),scale=beta)
                label_pred = rbm.generate(recon_hk,scale=beta).view(*SHAPE)[-2:,-1]
                if label_pred.argmax() == test_label.argmax():
                    count+=1
            accuracy_log.append(count/TEST)
            print(f"Testing accuracy: {count}/{TEST} ({count/TEST:.2f})")

        # ############################ Store Model and Logs ##############################
        torch.save(b_log,f"./{directory}/{SEED}/b.pt")
        torch.save(c_log,f"./{directory}/{SEED}/c.pt")
        torch.save(W_log,f"./{directory}/{SEED}/W.pt")
        torch.save(err_log,f"./{directory}/{SEED}/err.pt")
        torch.save(accuracy_log,f"./{directory}/{SEED}/accuracy.pt")


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
