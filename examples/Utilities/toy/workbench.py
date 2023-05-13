%matplotlib qt
import qaml
import dimod
import dwave
import torch
import random
import matplotlib
import minorminer

import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as torch_transforms

font = {'size'   : 22}

matplotlib.rc('font', **font)

SEEDS = [2,42,85]
PRUNES = np.linspace(0,0.35,8)

for prune in PRUNES:
    prune = round(prune,2)
    for i,seed in enumerate(SEEDS):
        filename = f'./RBM/embedding_{seed}_{prune}.pt'
        print(torch.load(filename))



# SEEDS = [2,42,85]
# PRUNES = [0.0]

for prune in PRUNES:
    prune = round(prune,2)
    df = pd.DataFrame()
    for i,seed in enumerate(SEEDS):
        filename = f'./RBM/measurement_{seed}_{prune}.pt'
        beta_dist = torch.load(filename)
        beta,dist = zip(*beta_dist)
        if i==0:
            df.insert(0,'beta',beta,True)
        if prune==0.0:
            label = f'Original'
        else:
            label = f'Pruned {prune:.2f}'
        df.insert(i+1,seed,dist,True)
        # df.T.rename_axis('beta').reset_index()
        # df = df.T.rename_axis('Beta').reset_index()
    df = df.melt(id_vars=["beta"],
                 value_vars=list(df.columns[1:]),
                 var_name="seed",
                 value_name="Distance")

    fig = sns.lineplot(x="beta",y="Distance",
                 data=df,ci="sd",lw=5,label=label,
                 err_kws={"alpha":0.15})

    plt.legend(framealpha=0.5)

fig.set_xlabel(r"$\beta$")
sns.set(font_scale = 2)




SHAPE = (10,10)
num_samples = 10000

beta_range = np.linspace(1,10,41)

# 10,10 - 15s
# 12,12 - 6m

torch.manual_seed(42)
rbm = qaml.nn.RestrictedBoltzmannMachine(*SHAPE,vartype=dimod.SPIN)
_ = torch.nn.init.uniform_(rbm.W,-1,1)
_ = torch.nn.init.uniform_(rbm.b,-4,4)
_ = torch.nn.init.uniform_(rbm.c,-4,4)
qa_sampler_rbm = qaml.sampler.QuantumAnnealingNetworkSampler(rbm,solver='Advantage_system4.1',auto_scale=True)
qvr,qhr = qa_sampler_rbm(num_samples)
beta_eff_r = list(qaml.perf.distance_from_gibbs(rbm,(qvr,qhr),beta_range=beta_range))
print(beta_eff_r)
miner = minorminer.miner(dimod.to_networkx_graph(qa_sampler_rbm.bqm),qa_sampler_rbm.to_networkx_graph())
miner.quality_key(qa_sampler_rbm.embedding)
# sampler_rbm = qaml.sampler.SimulatedAnnealingNetworkSampler(rbm)
# vr,hr = sampler_rbm(10)
# beta_eff_r = qaml.perf.distance_from_gibbs(rbm,(vr,hr))

# bqm = qa_sampler.to_qubo()
# embedding = qa_sampler.embedding
# source = dimod.to_networkx_graph(bqm)
# plt.figure(1)
# dnx.draw_pegasus_embedding(qa_sampler.networkx_graph,embedding,source)

torch.manual_seed(42)
rbm_pruned = qaml.nn.RestrictedBoltzmannMachine(*SHAPE,vartype=dimod.SPIN)
_ = torch.nn.init.uniform_(rbm_pruned.W,-1,1)
_ = torch.nn.init.uniform_(rbm_pruned.b,-4,4)
_ = torch.nn.init.uniform_(rbm_pruned.c,-4,4)
torch.nn.utils.prune.random_unstructured(rbm_pruned,'W',0.4)
qa_sampler_rbm_pruned = qaml.sampler.QuantumAnnealingNetworkSampler(rbm_pruned,solver='Advantage_system4.1')
qa_sampler_rbm_pruned.scalar
qvrp,qhrp = qa_sampler_rbm_pruned(num_samples)
beta_eff_rp = list(qaml.perf.distance_from_gibbs(rbm_pruned,(qvrp,qhrp),beta_range=beta_range))
print(beta_eff_rp)
miner = minorminer.miner(dimod.to_networkx_graph(qa_sampler_rbm_pruned.bqm),qa_sampler_rbm_pruned.to_networkx_graph())
miner.quality_key(qa_sampler_rbm_pruned.embedding)
# sampler_rbm_pruned = qaml.sampler.SimulatedAnnealingNetworkSampler(rbm_pruned)
# vrp,hrp = sampler_rbm_pruned(10)
# beta_eff_rp = qaml.perf.distance_from_gibbs(rbm_pruned,(vrp,hrp))

# bqm = qa_sampler.to_qubo()
# embedding = qa_sampler.embedding
# source = dimod.to_networkx_graph(bqm)
# plt.figure(2)
# dnx.draw_pegasus_embedding(qa_sampler.networkx_graph,embedding,source)

torch.manual_seed(48)
bm = qaml.nn.BoltzmannMachine(*SHAPE,vartype=dimod.SPIN)
_ = torch.nn.init.uniform_(bm.W,-1,1)
_ = torch.nn.init.uniform_(bm.b,-4,4)
_ = torch.nn.init.uniform_(bm.c,-4,4)
_ = torch.nn.init.uniform_(bm.vv,-1,1)
_ = torch.nn.init.uniform_(bm.hh,-1,1)
qa_sampler_bm = qaml.sampler.QuantumAnnealingNetworkSampler(bm,solver='Advantage_system4.1')
qa_sampler_bm.scalar
qvg,qhg = qa_sampler_bm(num_samples)
beta_eff_g = list(qaml.perf.distance_from_gibbs(bm,(qvg,qhg),beta_range=beta_range))
print(beta_eff_g)
miner = minorminer.miner(dimod.to_networkx_graph(qa_sampler_bm.bqm),qa_sampler_bm.to_networkx_graph())
miner.quality_key(qa_sampler_bm.embedding)
# sampler_bm = qaml.sampler.SimulatedAnnealingNetworkSampler(bm)
# vg,hg = sampler_bm(10)
# beta_eff_g = qaml.perf.distance_from_gibbs(bm,(vg,hg))

# bqm = qa_sampler.to_qubo()
# embedding = qa_sampler.embedding
# source = dimod.to_networkx_graph(bqm)
# plt.figure(1)
# dnx.draw_pegasus_embedding(qa_sampler.networkx_graph,embedding,source)

torch.manual_seed(48)
bm_pruned = qaml.nn.BoltzmannMachine(*SHAPE,vartype=dimod.SPIN)
_ = torch.nn.init.uniform_(bm_pruned.W,-1,1)
_ = torch.nn.init.uniform_(bm_pruned.b,-4,4)
_ = torch.nn.init.uniform_(bm_pruned.c,-4,4)
_ = torch.nn.init.uniform_(bm_pruned.vv,-1,1)
_ = torch.nn.init.uniform_(bm_pruned.hh,-1,1)
torch.nn.utils.prune.random_unstructured(bm_pruned,'W',0.4)
torch.nn.utils.prune.random_unstructured(bm_pruned,'vv',0.4)
torch.nn.utils.prune.random_unstructured(bm_pruned,'hh',0.4)
qa_sampler_bm_pruned = qaml.sampler.QuantumAnnealingNetworkSampler(bm_pruned,solver='Advantage_system4.1')
qa_sampler_bm_pruned.scalar
qvgp,qhgp = qa_sampler_bm_pruned(num_samples)
beta_eff_gp = list(qaml.perf.distance_from_gibbs(bm_pruned,(qvgp,qhgp),beta_range=beta_range))
print(beta_eff_gp)
miner = minorminer.miner(dimod.to_networkx_graph(qa_sampler_bm_pruned.bqm),qa_sampler_bm_pruned.to_networkx_graph())
miner.quality_key(qa_sampler_bm_pruned.embedding)

# sampler_bm_pruned = qaml.sampler.SimulatedAnnealingNetworkSampler(bm_pruned)
# vgp,hgp = sampler_bm_pruned(10)
# beta_eff_gp = qaml.perf.distance_from_gibbs(bm_pruned,(vgp,hgp))

# bqm = qa_sampler.to_qubo()
# embedding = qa_sampler.embedding
# source = dimod.to_networkx_graph(bqm)
# plt.figure(2)
# dnx.draw_pegasus_embedding(qa_sampler.networkx_graph,embedding,source)
