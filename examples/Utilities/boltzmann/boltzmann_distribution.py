import qaml
import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
torch.set_printoptions(precision=2)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

rbm = qaml.nn.RestrictedBoltzmannMachine(8,8)
solver = qaml.sampler.ExactNetworkSampler(rbm)
E = solver.get_energies()

probabilities = []
BETAS = [1.0,2.5,4.0]
for beta in BETAS:
    solver.beta.data = torch.tensor(beta)
    probabilities.append(solver.get_probabilities())
all_sorted = zip(*sorted(zip(E,*probabilities)))
sorted_E = next(all_sorted)
sorted_probs = list(all_sorted)

def plot_distributions(E,P):
    """ Args:
            E (iterable): Iterable of ``sorted'' energies.
            P (iterable of iterables): Iterables of ``sorted'' probabilities.
        Returns:
            ax (matplotlib.axes.Axes): Axis of multiple lines.
    """
    _ = plt.figure()
    for prob in P:
        plt.plot(E,prob,lw=5.0)
    # Make it pretty
    ax = plt.gca()
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xlabel(r'$E_i$',size=20)
    ax.set_ylabel(r'$p(\mathbf{x})$',size=20,rotation='horizontal',labelpad=20)
    return ax

ax1 = plot_distributions(sorted_E,sorted_probs)
ax1.legend([fr'$\beta$ = {beta}' for beta in BETAS])
ax1.set_xlim(sorted_E[0],sorted_E[10])
plt.tight_layout()
plt.savefig('boltzmann.svg')


ax2 = plot_distributions(sorted_E,sorted_probs)
ax2.legend([fr'$\beta$ = {beta}' for beta in BETAS])
ax2.set_yscale('log')
plt.tight_layout()
plt.savefig('log_boltzmann.svg')
