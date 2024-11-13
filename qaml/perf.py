import qaml
import torch
import numpy as np

import matplotlib.pyplot as plt

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
    # Join visible and hidden
    sampleset = torch.cat(samples,1).tolist()

    if beta_range is None:
        beta_range = iter(np.linspace(1,6,11))

    #  Calculate exact energies
    solver = qaml.sampler.ExactNetworkSampler(model)
    energies = solver.get_energies()

    # Calculate sample energies
    E_samples = solver.to_bqm().energies(sampleset)
    unique, counts = np.unique(E_samples, return_counts=True)
    EP_samples = dict(zip(unique, counts/len(E_samples)))


    # Iterate over beta distributions
    dist_eff = float('inf')
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

        yield beta_i, dist_i.item()


@torch.no_grad()
def finite_sampling_error(model, beta_range=None, num_reads=int(10e6),
                          reduction='sum'):
    """ Test a range of inverse temperature values and find the closest match
        betwween exact probabilities and num_reads-samples.
        Arg:
            model (qaml.nn.BoltzmannMachine): A BM model
            beta_range (iterable or None): Range of beta values to evaluate
                against.
            num_reads (int): Number of samples
        Return:
            beta (list of floats): same as the provided range or default range
            distance (list of floats): distances from the corresponding beta
    """
    if beta_range is None:
        beta_range = np.linspace(0,1.0,21)

    solver = qaml.sampler.ExactNetworkSampler(model)
    energies = solver.get_energies()

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
        elif reduction == 'max':
            dist_i = max(EP_diff.values())/2
        distances.append(dist_i)
    return beta_range, distances

@torch.no_grad()
def free_energy_smooth_kl(model, dataset, samples, bins=32, smooth=1e-6):
    """ Computes the smoothed KL divergence between two sample sets.
        e.g. Designed for the dataset and a sampler output.
        Arg:
            model (qaml.nn.BoltzmannMachine): A BM model
            dataset (Tensor(D,V)): Dataset input
            samples (Tensor(S,V)): Sampler output
            bins (int): Number of bins in histograms
            smooth (float): Smoothing epsilon value. smooth=0 for no smoothing
        Returns:
            kl_div (Tensor(1)): D_KL(p_dataset || p_samples)
    """
    # Free energies
    E_samples = model.free_energy(samples)
    E_dataset = model.free_energy(dataset)
    # Find valid range
    E_min = min(E_samples.min(),E_dataset.min()).item()
    E_max = max(E_samples.max(),E_dataset.max()).item()

    if bins is None:
        # Automatic bins and range from dataset
        d_hist, bin_edges = torch.histogram(E_dataset,range=(E_min,E_max))
        bins = len(bin_edges)-1
    else: # Histograms with shared bin edges
        d_hist, bin_edges = torch.histogram(E_dataset,bins,range=(E_min,E_max))

    # Matching range and bin for samples
    s_hist, _ = torch.histogram(E_samples,bins,range=(E_min,E_max))

    return smooth_kl(d_hist/len(dataset), s_hist/len(samples), smooth)

@torch.no_grad()
def smooth_kl(P, Q, smooth=1e-6):
    """ Computes the smoothed KL divergence between two probabilites.
        Based on implementation by Cameron Perot:
        https://jugit.fz-juelich.de/qip/qbm-quant-finance/
        Arg:
            P (Tensor): Probability array
            Q (Tensor): Probability array
            smooth (float): Smoothing epsilon value. smooth=0 for no smoothing
        Returns:
            kl_div (Tensor(1)): D_KL(P || Q)
    """
    # Smoothing of sample data i.e. divisor
    smooth_mask = (P > 0) & (Q == 0)
    not_smooth_mask = ~smooth_mask
    Q[smooth_mask] = P[smooth_mask] * smooth
    Q[not_smooth_mask] -= Q[smooth_mask].sum() / not_smooth_mask.sum()

    # Take intersection of supports, i.e. ignore if both zero
    support_intersection = (P > 0) & (Q > 0)
    P = P[support_intersection]
    Q = Q[support_intersection]

    return (P*torch.log(P/Q)).sum()


def chain_grad_variance(embedding,b_log,c_log):
    """ Given an embedding, and the value of the linear weights during training
        computes and plots the gradient at each epoch for each chain.
        Meant to check the relation between chain length and gradient distribution.
    """
    chain_length = {k:len(chain) for k,chain in embedding.items()}


    bias_data = {k:[] for k in embedding}
    for b_set,c_set in zip(b_log,c_log):
        for i,b in enumerate(b_set):
            bias_data[i].append(b)
        for j,c in enumerate(c_set,start=64):
            bias_data[j].append(c)

    index= []
    data = []
    length = []
    for i, (key, val) in enumerate(bias_data.items()):
        length.append(chain_length[i])
        index.append(key)
        data.append(val)

    sorted_length, sorted_data = zip(*sorted(zip(length, data)))

    gradient_data = []
    for data in sorted_data:
        gradient_bias = []
        for a,b in zip(data[:-1],data[1:]):
            gradient_bias.append(b-a)
        gradient_data.append(gradient_bias)

    _ = plt.figure()
    ax = plt.gca()
    _ = ax.boxplot(gradient_data)
    _ = ax.set_xticklabels(sorted_length)
