import qaml
import torch
import numpy as np

@torch.no_grad()
def distance_from_gibbs(model, samples, beta_range=None):
    """ Test a range of inverse temperature values to find the closests match
        to the given distribution. If possible to sample exactly.
        Arg:
            model (qaml.nn.BoltzmannMachine): A BM model
            samples (Tensor,Tensor): Visible and hidden samples.
            beta_range (iterable or None): Range of beta values to evaluate
                against.
        Return:
            beta (float):
            distance (float):
    """
    if beta_range is None:
        beta_range = iter(np.linspace(1,6,11))

    solver = qaml.sampler.ExactNetworkSampler(model)
    sampleset = torch.cat(samples,1).tolist()
    E_samples = solver.to_qubo().energies(sampleset)
    unique, counts = np.unique(E_samples, return_counts=True)
    hist_samples = dict(zip(unique, counts/len(E_samples)))

    dist_eff = float('inf')
    energies = solver.get_energies()
    for beta_i in beta_range:
        Z = np.exp(-beta_i*energies).sum()
        probs = torch.Tensor(np.exp(-beta_i*energies)/Z)

        EP_exact = {}
        for e,p in zip(energies,probs):
            EP_exact[e] = EP_exact.get(e,0)+p

        EP_diff = {}
        for k in EP_exact:
            EP_diff[k] = abs(EP_samples.get(k,0)-EP_exact.get(k,0))

        dist_i = sum(EP_diff.values())/2

        if dist_i < dist_eff:
            dist_eff = dist_i
            beta_eff = beta_i

    return beta_eff, dist_eff.item()


@torch.no_grad()
def finite_sampling_error(model, beta_range=None, num_reads=int(10e6),
                          reduction='sum'):
    """ Test a range of inverse temperature values and find the
        Arg:
            model (qaml.nn.BoltzmannMachine): A BM model
            samples (Tensor,Tensor): Visible and hidden samples.
            beta_range (iterable or None): Range of beta values to evaluate
                against.
        Return:
            beta (float):
            distance (float):
    """
    if beta_range is None:
        beta_range = np.linspace(0,1.0,21)

    solver = qaml.sampler.ExactNetworkSampler(model)
    energies = solver.get_energies()
    solutions = solver.sampleset

    distances = []
    for beta_i in beta_range:
        Z = np.exp(-beta_i*energies).sum()
        probs = torch.Tensor(np.exp(-beta_i*energies)/Z)

        EP_exact = {}
        for e,p in zip(energies,probs):
            EP_exact[e] = EP_exact.get(e,0)+p

        idx = torch.multinomial(probs,num_reads,replacement=True)
        E_samples = energies[idx]
        unique, counts = np.unique(E_samples,return_counts=True)
        EP_samples = dict(zip(unique, counts/len(E_samples)))

        EP_diff = {}
        for k in EP_exact:
            EP_diff[k] = abs(EP_samples.get(k,0)-EP_exact.get(k,0))

        if reduction == 'sum':
            dist_i = sum(EP_diff.values())/2
        elif reduction == 'mac':
            dist_i = max(EP_diff.values())/2
        distances.append(dist_i)
    return beta_range, distances
