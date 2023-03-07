############################## OptDigits RBM Example ############################
# Quantum BM training on the OptDigits Dataset for reconstruction and classification
# This is an example on quantum annealing training of an BM on the OptDigits
# dataset.
# Developed by: Jose Pinilla

# Required packages
import os
import qaml
import torch

import matplotlib.pyplot as plt


################################# Hyperparameters ##############################
EPOCHS = 75
M,N = SHAPE = (8,8)
SUBCLASSES = [0,1,2,3,5,6,7,8]
POS_READS, NEG_READS, TEST_READS = 100, 10000, 20

# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

################################# Model Definition #############################
VISIBLE_SIZE = M*N
HIDDEN_SIZE = 16
BATCH_SIZE = 32

# Specify model with dimensions
bm = qaml.nn.BoltzmannMachine(VISIBLE_SIZE,HIDDEN_SIZE,'SPIN')

# For deterministic weights
SEED = 8
torch.manual_seed(SEED)
# Initialize biases
_ = torch.nn.init.uniform_(bm.b,-4.0,4.0)
_ = torch.nn.init.uniform_(bm.c,-4.0,4.0)
_ = torch.nn.init.uniform_(bm.vv,-1.0,1.0)
_ = torch.nn.init.uniform_(bm.hh,-1.0,1.0)
_ = torch.nn.init.uniform_(bm.W,-1.0,1.0)

# Set up optimizer
optimizer = torch.optim.SGD(bm.parameters(),lr=learning_rate,
                            weight_decay=weight_decay,
                            momentum=momentum)

# Set up training mechanisms
sa_sampler =  qaml.sampler.SimulatedAnnealingNetworkSampler(bm)

# Loss and autograd
ML = qaml.autograd.MaximumLikelihood

#################################### Input Data ################################
opt_train = qaml.datasets.OptDigits(root='./data/',train=True,download=True,
                                        transform=qaml.datasets.ToSpinTensor())
qaml.datasets._subset_classes(opt_train,SUBCLASSES)
qaml.datasets._embed_labels(opt_train,encoding='one_hot',scale=255)
train_sampler = torch.utils.data.RandomSampler(opt_train,replacement=False)
train_loader = torch.utils.data.DataLoader(opt_train,BATCH_SIZE,sampler=train_sampler)


opt_test = qaml.datasets.OptDigits(root='./data/',train=False,download=True,
                                       transform=qaml.datasets.ToSpinTensor())
qaml.datasets._subset_classes(opt_test,SUBCLASSES)
set_label,get_label = qaml.datasets._embed_labels(opt_test,encoding='one_hot',
                                                  scale=255,setter_getter=True)
test_sampler = torch.utils.data.RandomSampler(opt_test,False)
test_loader = torch.utils.data.DataLoader(opt_test,BATCH_SIZE,sampler=test_sampler)

# Visualize
# fig,axs = plt.subplots(4,5)
# for ax,(img,label) in zip(axs.flat,test_loader):
#     ax.matshow(img.squeeze())
#     ax.set_title(int(label))
#     ax.axis('off')
# plt.tight_layout()

################################## Training Log ################################
err_log = []
kl_div_log = []
scalar_log = []
accuracy_log = []
batch_err_log = []
b_log = [bm.b.detach().clone().numpy()]
c_log = [bm.c.detach().clone().numpy()]
W_log = [bm.W.detach().clone().numpy().flatten()]
vv_log = [bm.vv.detach().clone().numpy().flatten()]
hh_log = [bm.hh.detach().clone().numpy().flatten()]

################################## Model Training ##############################
%%time
for t in range(1):
    print(f"Epoch {t}")
    epoch_error = torch.Tensor([0.])
    epoch_kl_div = torch.Tensor([0.])
    for img_batch, labels_batch in train_loader:

        # Positive Phase
        v0, prob_h0 = sa_sampler(img_batch.flatten(1),num_reads=POS_READS)
        # Negative Phase
        vk, prob_hk = sa_sampler(num_reads=NEG_READS,num_spin_reversal_transforms=4)

        # Reconstruction error from Contrastive Divergence
        err = ML.apply(qa_sampler,(v0,prob_h0), (vk,prob_hk), *bm.parameters())

        # Do not accumulate gradients
        optimizer.zero_grad()

        # Compute gradients
        err.backward()

        # Update parameters
        optimizer.step()

        #Accumulate error for this epoch
        epoch_error  += err
        batch_err_log.append(err.item())
        # epoch_kl_div += qaml.perf.free_energy_smooth_kl(bm,v0,vk)

    # Parameter log
    scalar_log.append(qa_sampler.scalar)
    b_log.append(bm.b.detach().clone().numpy())
    c_log.append(bm.c.detach().clone().numpy())
    vv_log.append(bm.vv.detach().clone().numpy())
    hh_log.append(bm.hh.detach().clone().numpy())
    W_log.append(bm.W.detach().clone().numpy().flatten())

    # Error Log
    kl_div_log.append(epoch_kl_div.item())
    print(f"KL Divergence = {epoch_kl_div.item()}")
    err_log.append(epoch_error.item())
    print(f"Reconstruction Error = {epoch_error.item()}")
# With batch_mode
# Epoch 0 Reconstruction Error = 942.0760498046875
# Wall time: 6min 16s

############################## CLASSIFICATION ##################################
%%time

count = 0
for test_data, test_label in test_loader:
    input_data = test_data.flatten(1)
    mask = set_label(torch.ones_like(test_data),0).flatten()
    #TODO: reconstruct can't use batch_embeddings, would need different ones
    v_recon,h_recon = qa_sampler.reconstruct(input_data,mask=mask,num_reads=TEST_READS)
    label_pred = get_label(v_recon.view(-1,*SHAPE))
    if (label_pred.mode(0)[0] == get_label(test_data[0])).all():
        count+=1
accuracy_log.append(count/TEST_READS)
print(f"Testing accuracy: {count}/{TEST_READS} ({count/TEST_READS:.2f})")

# Without batch_mode
# Testing accuracy: 1373/20 (68.65)
# Wall time: 9min 10s


%matplotlib qt
fig,axs = plt.subplots(10,5)
for ax,img in zip(axs.flat,vk):
    ax.matshow(img.view(*SHAPE))
    ax.axis('off')
plt.tight_layout()


# fig = plot_joint_samplesets([qa_sampler.sampleset],shape=(VISIBLE_SIZE,HIDDEN_SIZE))
# fig.canvas.manager.set_window_title(f'Input {test_label.argmax()+1}')

fig = plot_joint_samplesets([qa_sampler.sampleset],shape=(VISIBLE_SIZE,HIDDEN_SIZE))
fig.canvas.manager.set_window_title(f'Negative Phase')

############################ MODEL VISUALIZATION ###############################
directory = f"{HIDDEN_SIZE}_batch{BATCH_SIZE}_beta{beta}"
directory = directory.replace('.','')

if not os.path.exists(directory):
        os.makedirs(directory)
if not os.path.exists(f'{directory}/{SEED}'):
        os.makedirs(f'{directory}/{SEED}')

torch.save(b_log,f"./{directory}/{SEED}/b.pt")
torch.save(c_log,f"./{directory}/{SEED}/c.pt")
torch.save(W_log,f"./{directory}/{SEED}/W.pt")
torch.save(err_log,f"./{directory}/{SEED}/err.pt")
torch.save(batch_err_log,f"./{directory}/{SEED}/batch_err.pt")
torch.save(scalar_log,f"./{directory}/{SEED}/scalar.pt")
torch.save(accuracy_log,f"./{directory}/{SEED}/accuracy.pt")

# Testing accuracy graph
fig, ax = plt.subplots()
plt.plot(accuracy_log)
plt.ylabel("Testing Accuracy")
plt.xlabel("Epoch")
plt.savefig("accuracy.pdf")

# Scalar graph
fig, ax = plt.subplots()
plt.plot(scalar_log)
plt.ylabel("Alpha")
plt.xlabel("Epoch")
plt.savefig("alpha.pdf")

# Error graph
fig, ax = plt.subplots()
plt.plot(batch_err_log)
plt.ylabel("Batch Reconstruction Error")
plt.xlabel("Epoch")
plt.savefig("quantum_err.pdf")

# Error graph
fig, ax = plt.subplots()
plt.plot(err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")
plt.savefig("quantum_err.pdf")

# Visible bias graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',DATA_SIZE).colors))
lc_v = ax.plot(b_log)
plt.legend(iter(lc_v),[f'b{i}' for i in range(DATA_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Visible Biases")
plt.xlabel("Epoch")
plt.savefig("quantum_visible_bias_log.pdf")

# Hidden bias graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',HIDDEN_SIZE).colors))
lc_h = plt.plot(c_log)
plt.legend(lc_h,[f'c{i}' for i in range(HIDDEN_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Hidden Biases")
plt.xlabel("Epoch")
plt.savefig("quantum_hidden_bias_log.pdf")

# Weights graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',HIDDEN_SIZE*DATA_SIZE).colors))
lc_w = plt.plot(W_log)
plt.ylabel("Weights")
plt.xlabel("Epoch")
plt.savefig("quantum_weights_log.pdf")



################################## ENERGY ######################################

data_energies = []
for img,label in subdataset:
    data = torch.cat((torch.tensor(img).flatten(0),one_hot(torch.LongTensor([int(label)]))),0)
    data_energies.append(rbm.free_energy(data).item())

rand_energies = []
rand_data = torch.rand(len(tr_idx)*10,rbm.V)
for img in rand_data:
    rand_energies.append(rbm.free_energy(img.bernoulli()).item())

gibbs_energies = []
gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm,beta=beta)
for img,label in subdataset:
    data = torch.cat((torch.tensor(img).flatten(0),one_hot(torch.LongTensor([int(label)]))),0)
    prob_v,prob_h = gibbs_sampler(data,k=5)
    gibbs_energies.append(rbm.free_energy(prob_v.bernoulli()).item())

qa_energies = []
qa_sampleset = qa_sampler(num_reads=1000,auto_scale=True)
for s_v,s_h in zip(*qa_sampleset):
    qa_energies.append(rbm.free_energy(s_v.detach()).item())

plot_data = [(data_energies,  'Data',    'blue'),
             (rand_energies,  'Random',  'red'),
             (gibbs_energies, 'Gibbs-5', 'green'),
             (qa_energies,    'Quantum', 'orange')]

hist_kwargs = {'ec':'k','lw':2.0,'alpha':0.5,'histtype':'stepfilled','bins':100}
weights = lambda data: [1./len(data) for _ in data]

fig, ax = plt.subplots(figsize=(15,10))
for data,name,color in plot_data:
    ax.hist(data,weights=weights(data),label=name,color=color,**hist_kwargs)

plt.xlabel("Energy")
plt.ylabel("Count/Total")
plt.legend(loc='upper right')
plt.savefig("quantum_energies.pdf")



################################## VISUALIZE ###################################
plt.matshow(rbm.b.detach()[:DATA_SIZE].view(*SHAPE), cmap='viridis')
plt.colorbar()

fig,axs = plt.subplots(HIDDEN_SIZE//4,4)
for i,ax in enumerate(axs.flat):
    weight_matrix = rbm.W[i].detach()[:DATA_SIZE].view(*SHAPE)
    ms = ax.matshow(weight_matrix, cmap='viridis', vmin=-1, vmax=1)
    ax.axis('off')
cbar = fig.colorbar(ms, ax=axs.ravel().tolist(), shrink=0.95)
plt.savefig("quantum_weights.pdf")














import embera
import matplotlib
import scipy.stats
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D


def plot_joint_samplesets(samplesets, shape=None, gray=False, labels=None, savefig=True):
    nplots = len(samplesets)
    fig = plt.figure()

    def gray2bin(n):
        w = len(n)
        n = int(n, 2)
        mask = n
        while mask != 0:
            mask >>= 1
            n ^= mask
        return format(n,f'0{w}b')

    if shape is None:
        size = len(samplesets[0].variables)
        width = size//2
        height = size-size//2
    else:
        size = sum(shape)
        width,height = shape

    grid = plt.GridSpec(5, 5*nplots, hspace=0.0, wspace=0.0)

    maxX = 2**(width)-1
    maxY = 2**(height)-1

    minE = float('Inf')
    maxE = -float('Inf')
    x = {}; y = {}; E = {}; c = {}
    for i,sampleset in enumerate(samplesets):

        x[i] = []; y[i] = []; E[i] = []; c[i] = []
        if not sampleset: continue

        # Reverse iteration allows plotting lower (important) samples on top.
        for datum in sampleset.data(sorted_by='energy',reverse=True):
            value = ''.join(str(int(1+datum.sample[k])//2) for k in sorted(datum.sample))
            x_point = gray2bin(value[0:width]) if gray else value[0:width]
            y_point = gray2bin(value[width:]) if gray else value[width:]
            x[i].append(int(x_point,2)/maxX)
            y[i].append(int(y_point,2)/maxY)
            c[i].append(datum.num_occurrences)
            E[i].append(datum.energy)
            if datum.energy < minE: minE = datum.energy
            if datum.energy > maxE: maxE = datum.energy

    ims = []
    xlim=ylim=(0.0,1.0)
    rangeE = maxE - minE
    for i,sampleset in enumerate(samplesets):
        # Set up the axes with gridspec
        main_ax = fig.add_subplot(grid[1:5,i*5:4+(i*5)],xlim=xlim,ylim=ylim)

        h_params = {'frameon':False,'autoscale_on':False,'xticks':[],'yticks':[]}
        y_hist = fig.add_subplot(grid[1:5,4+(i*5)],sharey=main_ax,**h_params)
        x_hist = fig.add_subplot(grid[0,i*5:4+(i*5)],sharex=main_ax,**h_params)

        if not sampleset: main_ax.set_xlabel('N/A'); continue

        # Scatter points on the main axes
        ratE = [5+250*(((energy-minE)/rangeE)**2) for energy in E[i]]
        sct = main_ax.scatter(x[i],y[i],s=ratE,c=E[i],cmap="jet",alpha=0.5)

        minXY = [(x[i][ie],y[i][ie]) for ie,e in enumerate(E[i]) if e==minE]
        if minXY: main_ax.scatter(*zip(*minXY),s=100,linewidths=1,c='k',marker='x')

        # Histograms on the attached axes
        x_hist.hist(x[i], 100, histtype='stepfilled',
                    orientation='vertical', color='gray')

        y_hist.hist(y[i], 100, histtype='stepfilled',
                    orientation='horizontal', color='gray')

        ims.append(sct)
        if labels is None:
            labelX = 'VISIBLE'
            labelY = 'HIDDEN'
        else:
            labelX,labelY = labels
        main_ax.set_xlabel(labelX)
        main_ax.set_ylabel(labelY)

    # Color Bar
    vmin,vmax = zip(*[im.get_clim() for im in ims])

    for i,im in enumerate(ims):
        im.set_clim(vmin=min(vmin),vmax=max(vmax))

    plt.subplots_adjust(top=1,bottom=0.25,left=.05,right=.95,hspace=0,wspace=0)

    cax = fig.add_axes([0.25,0.15,0.5,0.02]) # [left,bottom,width,height]
    plt.colorbar(sct,orientation='horizontal',cax=cax)
    _ = cax.set_xlabel('Energy')

    if savefig:
        path = savefig if isinstance(savefig,str) else "./samplesets_joint.pdf"
        plt.savefig(path)

    return fig

# %%
