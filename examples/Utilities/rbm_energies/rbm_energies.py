import qaml
import torch

import matplotlib.pyplot as plt


opt_train = qaml.datasets.OptDigits(root='./data/',train=True,download=True,
                                        transform=qaml.datasets.ToSpinTensor())

rbm = qaml.nn.RestrictedBoltzmannMachine(64,64,'SPIN')


################################## ENERGY ######################################

data_energies = []
for img,label in opt_train:
    data_energies.append(rbm.free_energy(img.flatten(1)).item())

rand_energies = []
rand_data = torch.rand(len(opt_train)*10,rbm.V)
for img in rand_data:
    rand_energies.append(rbm.free_energy(img.bernoulli()).item())

gibbs_energies = []
gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm,len(opt_train))
for img,label in opt_train:
    prob_v,prob_h = gibbs_sampler(img.flatten(1),k=5)
    gibbs_energies.append(rbm.free_energy(prob_v.bernoulli()).item())

qa_energies = []
SOLVER_NAME = "Advantage_system4.1"
qa_sampler = qaml.sampler.QuantumAnnealingNetworkSampler(rbm,solver_name=SOLVER_NAME)
qa_sampleset = qa_sampler(num_reads=1000)
for s_v,s_h in zip(*qa_sampleset):
    qa_energies.append(rbm.free_energy(s_v.detach()).item())

plot_data = [(data_energies,  'Data',    'blue'),
             (rand_energies,  'Random',  'red'),
             (gibbs_energies, 'Gibbs-5', 'green'),
             (qa_energies,    'Quantum', 'orange')]

hist_kwargs = {'ec':'k','lw':2.0,'alpha':0.5,'histtype':'stepfilled','bins':100}
weights = lambda data: [1./len(data) for _ in data]

fig, ax = plt.subplots(figsize=(15,10))
for data,name,color in plot_data:
    ax.hist(data,weights=weights(data),label=name,color=color,**hist_kwargs)

plt.xlabel("Energy")
plt.ylabel("Count/Total")
plt.legend(loc='upper right')
plt.title("Pre-training energies")
plt.savefig('energies.pdf')
