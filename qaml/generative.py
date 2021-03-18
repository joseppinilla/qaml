import qaml
import torch

import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

# N is batch size; D_in is input dimension;
# H is hidden dimension
N, D_in, H = 64, 728, 128

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
train_dataset = torch_datasets.MNIST(root='data/mnist', train=True,
                                     transform=torch_transforms.ToTensor(),
                                     download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           shuffle=True)

test_dataset = torch_datasets.MNIST(root='data/mnist', train=False,
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
        nll = CD.apply(v_batch,*rbm.paramters())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the likelihood with respect to
        # model parameters
        nll.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
# Set the model to evaluation mode
rbm.eval()


# Save model
torch.save(rbm)
