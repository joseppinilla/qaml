import qaml
import torch
import itertools
import minorminer
import matplotlib
import numpy as np
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt

torch.manual_seed(42)

np.set_printoptions(precision=2)
torch.set_printoptions(precision=2)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

def plot_joint_samplesets(sampleset, shape, dataset=[], figsize=None):
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

    if figsize is None:
        fig_scale = 10/max(shape)
        figsize = (horizontal*fig_scale,vertical*fig_scale)
    fig = plt.figure(figsize=figsize) # (width,height)
    ax = plt.gca()
    # Scatter points
    sct = ax.scatter(x,y,c=E,cmap="jet",alpha=0.5,marker='s')
    ax.set_xlabel('VISIBLE')
    ax.set_ylabel('HIDDEN')
    # Point at lowest value (or values) with an 'X'
    minXY = [(x[ie],y[ie]) for ie,e in enumerate(E) if e==minE]
    ax.scatter(*zip(*minXY),s=100,linewidths=1,c='silver',marker='x')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    # Plot dataset lines
    for datum in dataset:
        value = ''.join(str(int(v.item())) for v in datum)
        x_point = bin_to_gray_index(value)
        ax.axvline(x=x_point/maxX)

    # Plot colorbar
    plt.subplots_adjust(top=.9,bottom=0.1,left=0.1,right=.9,hspace=0,wspace=0)
    cax = fig.add_axes([0.93,0.2,0.01,0.6]) # [left,bottom,width,height]
    cbar = plt.colorbar(sct,orientation='vertical',cax=cax,extend='neither',
                        ticks=[minE,minE + (maxE-minE)/2,maxE])

    return

################################################################################
################################################################################
SHAPE = (36,16)
bm = qaml.nn.BoltzmannMachine(*SHAPE,lin_range=[-4,4],quad_range=[-1,1])

# bm.b.data = torch.Tensor(np.array(torch.load("../../4_QABM/bas/bm36_16-100/full/8/b_log.pt")))[0]
# bm.c.data = torch.Tensor(np.array(torch.load("../../4_QABM/bas/bm36_16-100/full/8/c_log.pt")))[0]
# bm.vv.data = torch.Tensor(np.array(torch.load("../../4_QABM/bas/bm36_16-100/full/8/vv_log.pt")))[0]
# bm.hh.data = torch.Tensor(np.array(torch.load("../../4_QABM/bas/bm36_16-100/full/8/hh_log.pt")))[0]
# bm.W.data = torch.Tensor(np.array(torch.load("../../4_QABM/bas/bm36_16-100/full/8/W_log.pt")))[0].view(16,36)
# dataset = train_dataset.data.view(-1,36)

################################################################################
################################################################################
sa_sampler = qaml.sampler.SimulatedAnnealingNetworkSampler(bm)
sa_kwargs = {'num_sweeps':100,'proposal_acceptance_criteria':"Gibbs"}
vk,hk = sa_sampler(num_reads=10000,seed=42,**sa_kwargs)
plot_joint_samplesets(sa_sampler.sampleset,SHAPE)
# plt.savefig('sa_landscape_data.svg')


solver = qaml.sampler.ExactNetworkSampler(bm)
_ = solver.get_energies()
plot_joint_samplesets(solver.sampleset,SHAPE)
plt.savefig('exact_landscape.svg')

qa_sampler = qaml.sampler.QASampler(bm,solver="Advantage_system4.1")
qa_kwargs = {'num_spin_reversal_transforms':4,'auto_scale':True}
vq,hq = qa_sampler(num_reads=2500,**qa_kwargs)
plot_joint_samplesets(qa_sampler.sampleset,SHAPE)
# plt.savefig('qa_landscape_data.svg')

qa2_sampler = qaml.sampler.QASampler(bm,solver="Advantage2_prototype2.2")
qa2_kwargs = {'num_spin_reversal_transforms':4,'auto_scale':True}
v2,h2 = qa2_sampler(num_reads=2500,**qa_kwargs)
plot_joint_samplesets(qa2_sampler.sampleset,SHAPE)
# plt.savefig('qa2_landscape_data.svg')
