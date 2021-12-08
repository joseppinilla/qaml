import qaml
import torch

import numpy as np

from collections import Counter

def distance_from_gibbs(model, samples, beta_range=None, num_samples=1e4, k=5):
    """ Test a range of inverse temperature values to find the closests match
        to the given distribution. Proximity to a Gibbs distribution doesn't
        directly
        Arg:
            samples (tensor):
            sampler (qaml.sampler.NetworkSampler):
        Return:
            beta (float):
            distance (float):
    """
    if beta_range is None:
        beta_range = np.linspace(1,6,11)

    E_samples = model.energy(*samples)
    unique, counts = np.unique(E_samples.numpy(), return_counts=True)
    hist_samples = dict(zip(unique, counts/len(E_samples)))

    gibbs_sampler = qaml.sampler.GibbsNetworkSampler(model)

    beta_eff = 1.0
    distance = float('inf')
    for beta_i in beta_range:
        gibbs_sampler.beta = beta_i
        vk,hk = gibbs_sampler(torch.rand(num_samples,model.V),k=k)
        E_gibbs = model.energy(vk.bernoulli(),hk.bernoulli())
        unique, counts = np.unique(E_gibbs.numpy(), return_counts=True)
        hist_gibbs = dict(zip(unique, counts/num_samples))
        E_set = set(hist_samples) | set(hist_gibbs)
        E_diff = {k:abs(hist_samples.get(k,0)-hist_gibbs.get(k,0)) for k in E_set}
        dist_i = sum(E_diff.values())/2
        if dist_i < distance:
            distance = dist_i
            beta_eff = beta_i

    return beta_eff, distance
