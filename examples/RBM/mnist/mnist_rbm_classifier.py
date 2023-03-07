import qaml
import torch

import matplotlib.pyplot as plt

import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

################################# Hyperparameters ##############################
EPOCHS = 10
BATCH_SIZE = 64
M,N = SHAPE = (28,28)
# Stochastic Gradient Descent
learning_rate = 1e-2
weight_decay = 1e-4
momentum = 0.5

############################ Dataset and Transformations #######################
mnist_train = torch_datasets.MNIST(root='./data/', train=True, download=True,
                                     # transform=qaml.datasets.ToBinaryTensor())
                                     transform=qaml.datasets.ToSpinTensor())
qaml.datasets._embed_labels(mnist_train,axis=1,encoding='one_hot',scale=255)
train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=BATCH_SIZE)

mnist_test = torch_datasets.MNIST(root='./data/', train=False, download=True,
                                  # transform=qaml.datasets.ToBinaryTensor())
                                  transform=qaml.datasets.ToSpinTensor())
set_label,get_label = qaml.datasets._embed_labels(mnist_test,encoding='one_hot',
                                                 scale=255,setter_getter=True)
test_loader = torch.utils.data.DataLoader(mnist_test)

# Visualize
fig,axs = plt.subplots(4,5)
for ax,(img,label) in zip(axs.flat,test_loader):
    ax.matshow(img.squeeze())
    ax.set_title(int(label))
    ax.axis('off')
plt.tight_layout()

################################# Model Definition #############################
VISIBLE_SIZE = M*N
HIDDEN_SIZE = 128

# Specify model with dimensions
rbm = qaml.nn.RBM(VISIBLE_SIZE, HIDDEN_SIZE, 'SPIN')

# Set up optimizer
optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate,
                                              weight_decay=weight_decay,
                                              momentum=momentum)
# Set up training mechanisms
sampler = qaml.sampler.GibbsNetworkSampler(rbm,BATCH_SIZE)
CD = qaml.autograd.ContrastiveDivergence

################################## Model Training ##############################
# Set the model to training mode
rbm.train()
err_log = []
for t in range(EPOCHS):
    epoch_error = torch.Tensor([0.])
    for img_batch, labels_batch in train_loader:

        input_data = img_batch.flatten(1)

        # Positive Phase
        v0, p_h0 = sampler(input_data, k=0)
        # Negative Phase
        p_vk, p_hk = sampler(v0, k=5)

        # Reconstruction error from Contrastive Divergence
        err = CD.apply(sampler, (v0,p_h0), (p_vk,p_hk), *rbm.parameters())

        # Do not accumulated gradients
        optimizer.zero_grad()
        # Compute gradients. Save compute graph at last epoch
        err.backward(retain_graph = (t==EPOCHS-1))

        # Update parameters
        optimizer.step()
        epoch_error  += err
    err_log.append(epoch_error.item())
    print(f"Epoch {t} Reconstruction Error = {epoch_error.item()}")


# Reconstruction error graph
fig, ax = plt.subplots()
plt.plot(err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")

# %%
################################# VISUALIZE ####################################
# Computation Graph
from torchviz import make_dot
make_dot(err)

# %% raw
# Option to save for future use
torch.save(rbm,"mnist_unsupervised.pt")

# %% raw
# Option to load existing model
rbm = torch.load("mnist_unsupervised.pt")

# %%
################################# ANIMATION ####################################
from matplotlib.animation import FuncAnimation
img = torch.full((1,M,N),-1.0)
label = (torch.nn.functional.one_hot(torch.LongTensor([3]),10)*2)-1

FRAMES = 100
img_data = []
for _ in range(FRAMES):
    clamped = set_label(img*0.2,label)
    prob_hk = rbm.forward(clamped.flatten())
    img.data = 2*rbm.generate(2*prob_hk-1).view(1,M,N)-1
    img_data.append(img.detach().clone().numpy())

fig = plt.figure()
plot = plt.matshow(img_data[0].reshape(*SHAPE),fignum=0)

def init():
    plot.set_data(img_data[0].reshape(*SHAPE))
    return [plot]

def update(j):
    plot.set_data(img_data[j].reshape(*SHAPE))
    return [plot]

anim = FuncAnimation(fig,update,init_func=init,frames=FRAMES,interval=200,blit=True)
plt.show(block=False)
anim.save("./animation.gif","pillow")

# %%
############################## CLASSIFICATION ##################################
count = 0
for test_data, test_label in test_loader:
    input_data = set_label(test_data.view(1,*SHAPE),0)
    prob_hk = rbm.forward(input_data.flatten(1))
    label_pred = get_label(rbm.generate(prob_hk).view(1,*SHAPE))
    if label_pred.argmax() == test_label.item():
        count+=1

print(f"Testing accuracy: {count}/{len(mnist_test)}")
