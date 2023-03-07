%matplotlib qt
import qaml
import torch
import itertools
import minorminer
import matplotlib
import numpy as np
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
torch.set_printoptions(precision=2)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

def plot_joint_samplesets(samplesets, shape=None, gray=False, labels=None, savefig=False):
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

        if not sampleset: main_ax.set_xlabel('N/A'); continue

        # Scatter points on the main axes
        ratE = [5+250*(((energy-minE)/rangeE)**2) for energy in E[i]]
        sct = main_ax.scatter(x[i],y[i],c=E[i],cmap="jet",alpha=0.5,marker='s')

        minXY = [(x[i][ie],y[i][ie]) for ie,e in enumerate(E[i]) if e==minE]
        if minXY: main_ax.scatter(*zip(*minXY),s=100,linewidths=1,c='k',marker='x')

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



rbm = qaml.nn.RestrictedBoltzmannMachine(5,5)
solver = qaml.sampler.ExactNetworkSampler(rbm)

solver.beta.data = torch.tensor(3.0)
solver.beta
_ = solver.get_energies()
_ = plt.figure(1)
plot_joint_samplesets([solver.sampleset],gray=False)
plt.savefig('landscape.svg')
