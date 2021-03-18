import qaml
import torch

import matplotlib.pyplot as plt

import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

# N is batch size; D_in is input dimension;
# H is hidden dimension
N, D_in, H, D_out = 64, 784, 128, 10

################################# Hyperparameters ##############################
EPOCHS = 10
BATCH_SIZE = 64
# Stochastic Gradient Descent
learning_rate = 1e-3
weight_decay = 1e-4
momentum = 0.5
# Contrastive Divergence (CD-k)
cd_k = 1

#################################### Input Data ################################
train_dataset = torch_datasets.MNIST(root='./data/', train=True,
                                     transform=torch_transforms.ToTensor(),
                                     download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           shuffle=True)

test_dataset = torch_datasets.MNIST(root='./data/', train=False,
                                    transform=torch_transforms.ToTensor(),
                                    download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)


################################# Model Definition #############################
# Specify model with dimensions
rbm = qaml.nn.RBM(D_in, H)

# Set up optimizer
optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate,
                                              weight_decay=weight_decay,
                                              momentum=momentum)
# Set up training mechanisms
CD = qaml.autograd.ConstrastiveDivergence()

################################## Model Training ##############################
# Set the model to training mode
rbm.train()
for t in range(EPOCHS):
    for v_batch, labels_batch in train_loader:
        # Negative LogLikehood from Contrastive divergence
        err = CD.apply(v_batch.view(len(v_batch),D_in),*rbm.parameters())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the likelihood with respect to
        # model parameters
        err.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
    print(f"Epoch {t} Reconstruction Error = {err.item()}")
# Set the model to evaluation mode
rbm.eval()
plt.matshow(rbm.bv.detach().view(28, 28))

# Save model
# torch.save(rbm)


model = torch.nn.Sequential(rbm,
                            torch.nn.Linear(H,D_out),
                            torch.nn.ReLU())
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for t in range(20):
    for v_batch, labels_batch in train_loader:
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(v_batch.view(len(v_batch),D_in))

        # Compute and print loss.
        loss = loss_fn(y_pred,  torch.nn.functional.one_hot(labels_batch,10)*1.0)
        print(t, loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the Tensors it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

test = test_loader.dataset.data[60].view(1,784)*1.0
plt.matshow(test.view(28, 28))

loss

model(test).argmax()

from torchviz import make_dot

make_dot(loss)
