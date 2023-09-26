# %% markdown
# # This example does not use QAML or Boltzmann Machines

# %%
import torch

import matplotlib.pyplot as plt

import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

# N is batch size; D_in is input dimension;
# H is hidden dimension
N, D_in, H, D_out = 64, 784, 128, 10

TEST_SIZE = 400
TRAIN_SIZE = 2400

################################# Hyperparameters ##############################
EPOCHS = 20
BATCH_SIZE = 64

# Stochastic Gradient Descent
learning_rate = 1e-3
weight_decay = 1e-4
momentum = 0.5

#################################### Input Data ################################
train_dataset = torch_datasets.MNIST(root='./data/', train=True,
                                     transform=torch_transforms.ToTensor(),
                                     download=True)
train_sampler = torch.utils.data.RandomSampler(train_dataset,num_samples=TRAIN_SIZE)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           sampler=train_sampler)

test_dataset = torch_datasets.MNIST(root='./data/', train=False,
                                    transform=torch_transforms.ToTensor(),
                                    download=True)
test_sampler = torch.utils.data.RandomSampler(test_dataset,num_samples=TEST_SIZE)
test_loader = torch.utils.data.DataLoader(test_dataset,sampler=test_sampler)

################################## Model Training ##############################
model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                            torch.nn.ReLU(),
                            torch.nn.Linear(H, D_out),)
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

err_log = []
accuracy_log = []
for t in range(EPOCHS):
    for v_batch, labels_batch in train_loader:
        y_ref = torch.nn.functional.one_hot(labels_batch,10).float()
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(v_batch.view(len(v_batch),D_in))

        # Compute and print loss.
        loss = loss_fn(y_pred, y_ref)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the Tensors it will update
        # (which are the learnable weights of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()

        # The step function on an Optimizer makes an update to its parameters
        optimizer.step()
    err_log.append(loss.item())
    print(f"Epoch {t} Loss = {loss.item()}")

    ############################## CLASSIFICATION ##############################
    count = 0
    for test_data, test_label in test_loader:
        label_pred = model(test_data.view(1,D_in)).argmax()
        if label_pred == test_label:
            count+=1
    accuracy_log.append(count/TEST_SIZE)
    print(f"Testing accuracy: {count}/{TEST_SIZE} ({count/TEST_SIZE:.2f})")

fig, ax = plt.subplots()
plt.plot(err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")

fig, ax = plt.subplots()
plt.plot(accuracy_log)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
