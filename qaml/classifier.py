import qaml
import torch

import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 728, 128, 10

################################# Hyperparameters ##############################
EPOCHS = 10
BATCH_SIZE = 64
# Stochastic Gradient Descent
learning_rate = 1e-3
weight_decay = 1e-4
momentum = 0.5
# Contrastive Divergence (CD-k)
cd_k = 1

#################################### Input Data #################################
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
rbm = qaml.nn.RBM(D_in, H)
# Set the model to training mode
rbm.train()
# Set up optimizer
optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate,
                                              weight_decay=weight_decay,
                                              momentum=momentum)
grad_fn = qaml.autograd.LogLikelihood()
################################# Model Training #############################
for t in range(EPOCHS):
    for v_batch, labels_batch in train_loader:
        # Forward pass: compute hidden units. Modifies input in-place.
        pH_v = rbm(v_batch)

        # Compute loglikelihood
        ll = grad_fn()

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the likelihood with respect to
        # model parameters
        ll.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

# Set the model to evaluation mode
pre_model.eval()

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    pre_model, # Replaces: torch.nn.Linear(D_in, H)
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algorithms. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
