import torch
import torch.utils.data

import numpy as np
import matplotlib.pyplot as plt

from RBM_helper import RBM

batch_size = 64
epochs = 200
gpu = False

dummy_training = False

#DUMMY TRAINING SET
# ------------------------------------------------------------------------------
#define a simple training set and check if rbm.draw() returns this after training.
if dummy_training:
    data = np.array([[1,0,1,0,1,0,1,0,1,0]]*1000 + [[0,1,0,1,0,1,0,1,0,1]]*1000)
    np.random.shuffle(data)
    data = torch.FloatTensor(data)
else:
    data = np.repeat(np.load('./bars_and_stripes.npy'),100,axis=0)
    np.random.shuffle(data)
    data = torch.FloatTensor(data)
# ------------------------------------------------------------------------------

vis = len(data[0]) #input dimension

rbm = RBM(n_vis = vis, n_hin = vis*2, k=1, gpu = gpu)
if gpu:
    rbm = rbm.cuda()
    all_spins = all_spins.cuda()

for epoch in range(epochs):
    print(f"Epoch: {epoch}\n")
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,shuffle=True)
    momentum = 1 - 0.1*(epochs-epoch)/epochs #starts at 0.9 and goes up to 1
    lr = (0.1*np.exp(-epoch/epochs*10))+0.0001
    rbm.train(train_loader, lr = lr, momentum = momentum)

print('DRAW SAMPLES')
fig, ax = plt.subplots(1, 10, figsize=(9, 1))
for i in range(10):
    bas = rbm.draw_sample(gibbs_k=10)
    ax[i].matshow(bas.reshape(3,3).detach().numpy(), vmin=-1, vmax=1)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
