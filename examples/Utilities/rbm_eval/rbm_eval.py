import qaml
import dimod
import dwave
import torch
import random
import minorminer
import torch.nn.functional as F



import numpy as np
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import torchvision.transforms as torch_transforms

SHAPE = (10,10)
num_samples = 10000

beta_range = np.linspace(1,10,41)

# 10,10 - 15s
# 12,12 - 6m


SEEDS = [2,42,85]
PRUNES = np.linspace(0,0.4,9)

for prune in PRUNES:
    for seed in SEEDS:
        torch.manual_seed(seed)
        rbm_pruned = qaml.nn.RestrictedBoltzmannMachine(*SHAPE,vartype=dimod.SPIN)
        _ = torch.nn.init.uniform_(rbm_pruned.W,-1,1)
        _ = torch.nn.init.uniform_(rbm_pruned.b,-4,4)
        _ = torch.nn.init.uniform_(rbm_pruned.c,-4,4)
        torch.nn.utils.prune.random_unstructured(rbm_pruned,'W',prune)
        qa_sampler_rbm_pruned = qaml.sampler.QuantumAnnealingNetworkSampler(rbm_pruned,solver='Advantage_system4.1')
        qa_sampler_rbm_pruned.scalar
        qvrp,qhrp = qa_sampler_rbm_pruned(num_samples)
        beta_eff_rp = list(qaml.perf.distance_from_gibbs(rbm_pruned,(qvrp,qhrp),beta_range=beta_range))
        miner = minorminer.miner(dimod.to_networkx_graph(qa_sampler_rbm_pruned.bqm),qa_sampler_rbm_pruned.to_networkx_graph())
        QK = miner.quality_key(qa_sampler_rbm_pruned.embedding)
        print(QK)
        print(beta_eff_rp)
        filename = f'embedding_{seed}_{prune}.pt'
        torch.save(QK,filename)
        filename = f'measurement_{seed}_{prune}.pt'
        torch.save(beta_eff_rp,filename)
