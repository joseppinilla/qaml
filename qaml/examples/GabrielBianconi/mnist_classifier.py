import torch
import torchvision.datasets
import torchvision.models
import torchvision.transforms

import numpy as np
import matplotlib.pyplot as plt

from rbm import RBM

########## CONFIGURATION ##########
BATCH_SIZE = 64
INPUT_SIZE = 784  # 28 x 28 images
LABEL_SIZE = 10
VISIBLE_UNITS = INPUT_SIZE + LABEL_SIZE
HIDDEN_UNITS = 128
CD_K = 5
EPOCHS = 10

DATA_FOLDER = 'data/mnist'

CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0

if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)


########## LOADING DATASET ##########
print('Loading dataset...')

train_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

########## TRAINING RBM ##########
print('Training RBM...')
rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, use_cuda=CUDA)


for epoch in range(EPOCHS):
    epoch_error = 0.0

    for data_batch, labels_batch in train_loader:
        flat_data = data_batch.view(len(data_batch),INPUT_SIZE)
        one_hot_label = torch.nn.functional.one_hot(labels_batch,10)
        batch = torch.cat((flat_data,one_hot_label),1)
        if CUDA:
            batch = batch.cuda()

        batch_error = rbm.contrastive_divergence(batch)

        epoch_error += batch_error

    print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))

torch.save(rbm,"rbm_classifier.pt")


rbm = torch.load("rbm_classifier.pt")

torch.seed()
sample = rbm.sample(k=100)
plt.matshow(sample[0][0:INPUT_SIZE].view(28, 28))
sample[0][INPUT_SIZE:INPUT_SIZE+LABEL_SIZE].argmax()

############## SAMPLE WITH CLAMPED LABEL
%matplotlib qt
from matplotlib.animation import FuncAnimation

SAMPLES = 1000
data = torch.zeros(SAMPLES,28,28)
visible_activations = torch.randn(VISIBLE_UNITS)
clamp = torch.nn.functional.one_hot(torch.Tensor([3]).to(torch.int64),10)
for i in range(SAMPLES):
    visible_activations[INPUT_SIZE:] = clamp
    hidden_probabilities = rbm.sample_hidden(visible_activations)
    visible_probabilities = rbm.sample_visible(hidden_probabilities)
    visible_activations = (visible_probabilities >= rbm._random_probabilities(rbm.num_visible)).float()
    data[i] = visible_probabilities[:INPUT_SIZE].view(28,28)



fig = plt.figure()
plot = plt.matshow(data[0],fignum=0)

def init():
    plot.set_data(data[0])
    return plot,

def update(j):
    plot.set_data(data[j])
    return [plot]


anim = FuncAnimation(fig, update, init_func = init, frames=SAMPLES, interval = 10, blit=True)
plt.show()


clamp = torch.nn.functional.one_hot(torch.Tensor([3]).to(torch.int64),10)
visible_activations =  torch.zeros(VISIBLE_UNITS)

for step in range(100):
    visible_activations[INPUT_SIZE:] = clamp
    hidden_probabilities = rbm.sample_hidden(visible_activations)
    visible_probabilities = rbm.sample_visible(hidden_probabilities)
    visible_activations = (visible_probabilities >= rbm._random_probabilities(rbm.num_visible)).float()

plt.matshow(visible_probabilities[0:INPUT_SIZE].view(28, 28))

############## CLASSIFICATION WITH CLAMPED INPUT

clamp = test_loader.dataset.data[100].view(INPUT_SIZE)
print(test_loader.dataset.targets[100])
plt.matshow(clamp.view(28, 28))
visible_activations = torch.zeros(VISIBLE_UNITS)
for step in range(100):
    visible_activations[:INPUT_SIZE] = clamp
    hidden_probabilities = rbm.sample_hidden(visible_activations)
    visible_probabilities = rbm.sample_visible(hidden_probabilities)
    visible_activations = (visible_probabilities >= rbm._random_probabilities(rbm.num_visible)).float()

probs = visible_probabilities[INPUT_SIZE:]/sum(visible_probabilities[INPUT_SIZE:])
print({k:f"{p.item():.0%}" for k,p in enumerate(probs)})
