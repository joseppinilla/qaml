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


def plot_boltzmann(solver,beta):
    E,P,Z = solver.get_energies(), solver.get_probabilities()
    dict_EP = dict(zip(E,P))
    sorted_E = sorted(E)
    sorted_P = [dict_EP[e].item() for e in sorted_E]
    plt.plot(sorted_E,sorted_P,label=beta,lw=5.0)

rbm = qaml.nn.RestrictedBoltzmannMachine(5,5)
solver = qaml.sampler.ExactNetworkSampler(rbm)

BETAS = [1.0,2.5,4.0]
for beta in BETAS:
    solver.beta.data = torch.tensor(beta)
    plot_boltzmann(solver,beta)
plt.legend([fr'$\beta$ = {beta}' for beta in BETAS])

ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.set_xlabel(r'$E_i$',size=20)
ax.set_ylabel(r'$p(\mathbf{x})$',size=20,rotation='horizontal',labelpad=20)
# plt.ylabel("Y Axis", labelpad=15)
plt.tight_layout()
plt.savefig('boltzmann.svg')
