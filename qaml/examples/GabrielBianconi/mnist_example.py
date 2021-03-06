import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.datasets
import torchvision.models
import torchvision.transforms
import matplotlib.pyplot as plt

from rbm import RBM


########## CONFIGURATION ##########
BATCH_SIZE = 64
VISIBLE_UNITS = 784  # 28 x 28 images
HIDDEN_UNITS = 128
CD_K = 2
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

    for batch, _ in train_loader:
        batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data

        if CUDA:
            batch = batch.cuda()

        batch_error = rbm.contrastive_divergence(batch)

        epoch_error += batch_error

    print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))

plt.matshow(rbm.visible_bias.view(28, 28))


########## EXTRACT FEATURES ##########
print('Extracting features...')

train_features = np.zeros((len(train_dataset), HIDDEN_UNITS))
train_labels = np.zeros(len(train_dataset))
test_features = np.zeros((len(test_dataset), HIDDEN_UNITS))
test_labels = np.zeros(len(test_dataset))

for i, (batch, labels) in enumerate(train_loader):
    batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data

    if CUDA:
        batch = batch.cuda()

    train_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = rbm.sample_hidden(batch).cpu().numpy()
    train_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()

for i, (batch, labels) in enumerate(test_loader):
    batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data

    if CUDA:
        batch = batch.cuda()

    test_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = rbm.sample_hidden(batch).cpu().numpy()
    test_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()


########## CLASSIFICATION ##########
print('Classifying...')

clf = LogisticRegression(solver='liblinear', multi_class='auto')
clf.fit(train_features, train_labels)
predictions = clf.predict(test_features)

print('Result: %d/%d' % (sum(predictions == test_labels), test_labels.shape[0]))

########## SAMPLE ##########
torch.seed()
sample = rbm.sample(k=100)
plt.matshow(sample.view(28, 28))

########## NOISE RECONSTRUCTION ##########
input_data, label = train_loader.dataset[2]
corrupt_data = (input_data + torch.randn_like(input_data)*0.2).view(1,VISIBLE_UNITS)
visible_activations = corrupt_data

for step in range(100):
    hidden_probabilities = rbm.sample_hidden(visible_activations)
    visible_probabilities = rbm.sample_visible(hidden_probabilities)
    visible_activations = (visible_probabilities >= rbm._random_probabilities(rbm.num_visible)).float()

plt.matshow(corrupt_data.view(28, 28))
plt.matshow(visible_probabilities.view(28, 28))

########## RECONSTRUCTION ##########
input_data, label = train_loader.dataset[4]
mask = torch.ones_like(input_data)
for i in range(0,15):
    for j in range(0,15):
        mask[0][j][i] = 0
plt.matshow(input_data.view(28, 28))
corrupt_data = (input_data*mask).view(1,VISIBLE_UNITS)
plt.matshow(corrupt_data.view(28, 28))

visible_activations = corrupt_data
for step in range(5):
    hidden_probabilities = rbm.sample_hidden(visible_activations)
    visible_probabilities = rbm.sample_visible(hidden_probabilities)
    visible_activations = (visible_probabilities >= rbm._random_probabilities(rbm.num_visible)).float()
plt.matshow(visible_probabilities.view(28, 28))
