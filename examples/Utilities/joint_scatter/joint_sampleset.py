
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

def plot_joint_samplesets(sampleset,shape):
    # Use shape to find size of the graph
    horizontal,vertical = shape
    maxX, maxY = 2**(horizontal)-1, 2**(vertical)-1

    # Treat the numbers as Gray code to plot similar numbers close to each other
    # e.g. '0110' -> 4, '0111' -> 5, '0101' -> 6, ...
    def bin_to_gray_index(n):
        mask = n = int(n,2)
        while mask != 0:
            mask >>= 1; n ^= mask
        return n

    # Takes a dimod.SampleSet and returns a string of the sample values
    # e.g. sample={0: 0, 1: 0, 2: 1, 3: 0, 4: 1, 5: 1} -> "00101"
    def sample_to_string(datum):
        key_sorted = sorted(datum.sample)
        return ''.join(str(int(1+datum.sample[k])//2) for k in key_sorted)

    # Initilize results
    minE, maxE = None, None
    x = []; y = []; E = []; c = []
    # Reverse iteration plots lower energy samples on top if there is overlap
    for datum in sampleset.data(sorted_by='energy',reverse=True):
        value = sample_to_string(datum)
        x_point = bin_to_gray_index(value[0:horizontal])
        y_point = bin_to_gray_index(value[horizontal:])
        x.append(x_point/maxX); y.append(y_point/maxY)
        c.append(datum.num_occurrences); E.append(datum.energy)
        if (minE is None) or (datum.energy < minE): minE = datum.energy
        if (maxE is None) or (datum.energy > maxE): maxE = datum.energy


    fig = plt.figure(figsize=(8,8))
    ax = plt.gca()
    # Scatter points
    sct = ax.scatter(x,y,c=E,cmap="jet",alpha=0.5,marker='s')
    ax.set_xlabel('VISIBLE')
    ax.set_ylabel('HIDDEN')
    # Point at lowest value (or values) with an 'X'
    minXY = [(x[ie],y[ie]) for ie,e in enumerate(E) if e==minE]
    ax.scatter(*zip(*minXY),s=100,linewidths=1,c='k',marker='x')
    # Plot colorbar
    cax = fig.add_axes([0.0,0.0,1.0,0.02]) # [left,bottom,width,height]
    plt.colorbar(sct,orientation='horizontal',cax=cax,label='Energy')

################################################################################
################################################################################
SHAPE = (8,8)
rbm = qaml.nn.RestrictedBoltzmannMachine(*SHAPE)
solver = qaml.sampler.ExactNetworkSampler(rbm)
_ = solver.get_energies()

plot_joint_samplesets(solver.sampleset,SHAPE)
plt.savefig('landscape.svg')
