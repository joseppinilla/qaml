import qaml
import torch
import dimod
import numpy as np
import matplotlib.pyplot as plt

model = qaml.nn.RestrictedBoltzmannMachine(4,4)
model.b.data = model.b.data*4
model.c.data = model.c.data*4

model.b.data = model.b.data*0
model.c.data = model.c.data*0
beta_range = np.linspace(0,0.8,33)



# beta,dist = qaml.perf.finite_sampling_error(model)
# plt.plot(beta,dist)

gibbs_sampler = qaml.sampler.GibbsNetworkSampler(model)
v5,h5 = gibbs_sampler(torch.rand(10000,model.V),k=5)
v10,h10 = gibbs_sampler(torch.rand(10000,model.V),k=10)

qaml.perf.free_energy_smooth_kl(model,v5,v10,bins=None,smooth=0)

qaml.perf.free_energy_smooth_kl(model,v5,v10,smooth=0)

_,_,edges = free_energy_histogram(model,v5,v10)

len(edges)

def free_energy_histogram(model, dataset, samples, bins=32, smooth=1e-6):

    E_samples = model.free_energy(samples)
    E_dataset = model.free_energy(dataset)

    E_min = min(E_samples.min(),E_dataset.min()).item()
    E_max = max(E_samples.max(),E_dataset.max()).item()

    d_hist, bin_edges = torch.histogram(E_dataset, bins=bins, range=(E_min,E_max))
    s_hist, _ = torch.histogram(E_samples, bins=bins, range=(E_min,E_max))

    return d_hist/len(dataset), s_hist/len(samples), bin_edges


@torch.no_grad()
def kl_divergence(model, dataset, samples, bins=32, smooth=1e-6):
    """

    Based on implementation by Cameron Perot:
    https://jugit.fz-juelich.de/qip/qbm-quant-finance/-/tree/main
    """
    P = np.zeros(bins)
    Q = np.zeros(bins)

    E_samples = model.free_energy(samples)
    s_energies, s_counts = np.unique(E_samples, return_counts=True)

    E_dataset = model.free_energy(dataset)
    d_energies, d_counts = np.unique(E_dataset, return_counts=True)

    E_min = min(s_energies.min(),d_energies.min())
    E_max = min(s_energies.max(),d_energies.max())

    buffer = max(np.abs(E_min),np.abs(E_min)) * 1e-15
    bin_edges = np.linspace(E_min - buffer, E_max + buffer, bins + 1)

    for i, (a, b) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if i < bins - 1:
            P[i] = (d_counts[np.logical_and(d_energies >= a, d_energies < b)].sum() / len(dataset))
            Q[i] = (s_counts[np.logical_and(s_energies >= a, s_energies < b)].sum() / len(samples))
        else:
            P[i] = d_counts[d_energies >= a].sum() / len(dataset)
            Q[i] = s_counts[s_energies >= a].sum() / len(samples)

    print()

    # smoothing of sample data i.e. divisor
    smooth_mask = np.logical_and(P > 0, Q == 0)
    not_smooth_mask = np.logical_not(smooth_mask)
    Q[smooth_mask] = P[smooth_mask] * smooth
    Q[not_smooth_mask] -= Q[smooth_mask].sum() / not_smooth_mask.sum()

    # take intersection of supports, i.e. ignore if both zero
    support_intersection = np.logical_and(P > 0, Q > 0)
    P = P[support_intersection]
    Q = Q[support_intersection]

    return (P*np.log(P/Q)).sum()
