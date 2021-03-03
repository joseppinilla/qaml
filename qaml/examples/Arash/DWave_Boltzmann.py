#import the necessary shit
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

import networkx as nx
from dimod.reference.samplers import SimulatedAnnealingSampler
import dwave_networkx as dnx
from minorminer import find_embedding
# from dwave.system.samplers import DWaveSampler
# from dwave.system.composites import FixedEmbeddingComposite
%matplotlib inline

#Training rate
𝜁=0.05

class SRQBM():
    def __init__(self,num_visible,num_hidden):
        self.num_visible=num_visible
        self.num_hidden=num_hidden
        self.Q={}
        self.network={'lateral_hidden_weights': 0*np.random.rand(num_hidden, num_hidden-1),
                      'lateral_visible_weights': 0*np.random.rand(num_visible, num_visible-1),
                      'visible_to_hidden_weights': np.random.rand(num_visible, num_hidden)-0.5,
                      'vbias': np.random.rand(num_visible)-0.5,'hbias': np.random.rand(num_hidden)-0.5}
    def set_up_QUBO(self,v=None,h=None):
        if v is None:
            v=self.network['vbias']
        if h is None:
            h=self.network['hbias']
        for i in range(0,self.num_visible):
            self.Q[(i,i)]=v[i]
            for j in range(0,self.num_visible):
                if i >j:
                    self.Q[(i,j)]=self.network['lateral_visible_weights'][i][j]
                elif j> i:
                    self.Q[(i,j)]=self.network['lateral_visible_weights'][i][j-1]
            for j in range(self.num_visible,self.num_visible+self.num_hidden):
                self.Q[(i,j)]=self.network['visible_to_hidden_weights'][i][j-self.num_visible]
        for j in range(self.num_visible,self.num_visible+self.num_hidden):
            self.Q[(j,j)]=h[j-self.num_visible]
    def do_embedding(self, A):
        i=0
        self.embedding=None
        while self.embedding is None and i<10:
            self.embedding = find_embedding(self.Q, A)
            i+=1
        self.max_chain_length=max(len(chain) for chain in self.embedding.values())

    def sample_h_fixed(self, x, num_runs=100):
        chainstrength=2
        self.set_up_QUBO(h=(-4*x+2))
        #h=FixedEmbeddingComposite(DWaveSampler(solver={'lower_noise': True, 'qpu': True}), self.embedding).sample_qubo(self.Q, chain_strength=chainstrength, auto_scale=False, num_reads=10000)
        h=SimulatedAnnealingSampler().sample_qubo(self.Q, num_reads=num_runs)
        ph=deepcopy(h.record[0][0][:])*h.record[0][2]
        sample_length=len(h)
        for i in range(1,sample_length):
            ph+= h.record[i][0][:]*h.record[0][2]
        ph=ph/sample_length
        return ph, h

    def sample_v_fixed(self, x, num_runs=10):
        chainstrength=2
        self.set_up_QUBO(v=(-4*x+2))
        #v=FixedEmbeddingComposite(DWaveSampler(solver={'lower_noise': True, 'qpu': True}), self.embedding).sample_qubo(self.Q, chain_strength=chainstrength, auto_scale=False, num_reads=10000)
        v=SimulatedAnnealingSampler().sample_qubo(self.Q, num_reads=num_runs)
        pv=deepcopy(v.record[0][0][:])*v.record[0][2] #<Za>v
        ZaZb_v=np.array([v.record[0][0][:]]) *  np.array([v.record[0][0][:]]).T*v.record[0][2] #<ZaZb>v
        sample_length=len(v)
        for i in range(1,sample_length):
            pv+= v.record[i][0][:]*v.record[i][2]  #(self.num_visible)
            ZaZb_v+=np.array([v.record[i][0][:]]) *  np.array([v.record[i][0][:]]).T*v.record[i][2]
        pv=pv/sample_length
        ZaZb_v=ZaZb_v/sample_length
        ZaZb_v=ZaZb_v[~np.eye(ZaZb_v.shape[0],dtype=bool)].reshape(ZaZb_v.shape[0],-1)
        return pv, v, ZaZb_v

    def sample(self,num_runs=10):
        chainstrength=2
        self.set_up_QUBO()
        #v=FixedEmbeddingComposite(DWaveSampler(solver={'lower_noise': True, 'qpu': True}), self.embedding).sample_qubo(self.Q, chain_strength=chainstrength, auto_scale=False, num_reads=num_runs)
        v=SimulatedAnnealingSampler().sample_qubo(self.Q, num_reads=num_runs)
        ps=deepcopy(v.record[0][0][:]) #<Za>
        ZaZb=np.array([v.record[0][0][:]]) *  np.array([v.record[0][0][:]]).T*v.record[0][2] #<ZaZb>v #<ZaZb>
        sample_length=len(v)
        for i in range(1,sample_length):
            ps+= v.record[i][0][:]*v.record[i][2]  #(self.num_visible)
            ZaZb+=np.array([v.record[i][0][:]]) *  np.array([v.record[i][0][:]]).T*v.record[i][2]
        ps=ps/sample_length
        ZaZb=ZaZb/sample_length
        ZaZb=ZaZb[~np.eye(ZaZb.shape[0],dtype=bool)].reshape(ZaZb.shape[0],-1)
        return ps, v, ZaZb

    def classical_train(self, Za_v_bar, Za, ZaZb_v_bar, ZaZb):
        self.network['visible_to_hidden_weights'] = self.network['visible_to_hidden_weights'] + 𝜁 *(ZaZb_v_bar[self.num_visible:(self.num_visible+self.num_hidden),:self.num_visible].T-ZaZb[self.num_visible:(self.num_visible+self.num_hidden),:self.num_visible].T)
        #print(Za_v_bar[:self.num_visible]-Za[:self.num_visible])
        self.network['vbias'] = self.network['vbias'] + 𝜁 *(Za_v_bar[:self.num_visible]-Za[:self.num_visible])
        self.network['hbias'] = self.network['hbias'] + 𝜁 *(Za_v_bar[self.num_visible:]-Za[self.num_visible:])

    def draw_graph(self):
        connectivity_structure = dnx.chimera_graph(16,16)
        fig=plt.figure(figsize=(25, 25))
        dnx.draw_chimera_embedding(connectivity_structure, self.embedding)


def train_rbm(rbm, nb_epoch, train_dataset):
    for epoch in range(0, nb_epoch):
        train_loss = 0
        Za_v_bar=0
        ZaZb_v_bar=0
        s = 0
        train_len=len(train_dataset)
        for i in range(0, train_len):
            v0 = train_dataset[i]
            Za_v,_,ZaZb_v=rbm.sample_v_fixed(v0,num_runs=300)
            Za_v_bar += (1/train_len)*Za_v
            ZaZb_v_bar += (1/train_len)*ZaZb_v
            a=Za_v_bar[:rbm.num_visible]
            b=Za_v[:rbm.num_visible]
            s += 1.
        Za,_,ZaZb = rbm.sample(num_runs=300)
        rbm.classical_train(Za_v_bar, Za, ZaZb_v_bar, ZaZb)
        train_loss = np.mean((Za_v_bar - Za))
        print('epoch: '+str(epoch)+' loss: '+str(train_loss))
        rbm.set_up_QUBO()
    return train_loss

def data_generator(num_data_points, noise_probability,dim):
    one = np.zeros((dim,dim))
    two = np.zeros((dim,dim))
    one[1, :] = 1
    one[:, 1] = 1
    two[np.arange(dim),np.arange(dim)]=1
    two[np.arange(dim), -np.arange(dim)-1]=1
    data={}
    for i in range(0,num_data_points):
        if np.random.binomial(1, 0.5)==1:
            data[i]=one.reshape(dim**2)+(np.random.rand(dim**2)<noise_probability).astype(int)
            data[i]=np.mod(data[i],2)
        else:
            data[i]=two.reshape(dim**2)+(np.random.rand(dim**2)<noise_probability).astype(int)
            data[i]=np.mod(data[i],2)
    return data

dim=3; input_dim = dim * dim; hidden_dim = 2; nb_epoch = 10
rbm=SRQBM(input_dim,hidden_dim)
rbm.set_up_QUBO()
G = nx.Graph(rbm.Q.keys())
subsets = {v:v<input_dim for v in G}
nx.set_node_attributes(G,subsets,"subset")
nx.draw(G,nx.multipartite_layout(G))

#generate some data
np.random.seed(42)
data=data_generator(4, 0.0, dim)
fig, ax = plt.subplots(ncols=4)
ax[0].imshow(data[0].reshape(dim,dim))
ax[1].imshow(data[1].reshape(dim,dim))
ax[2].imshow(data[2].reshape(dim,dim))
ax[3].imshow(data[3].reshape(dim,dim))
# ax[4].imshow(data[4].reshape(dim,dim))
# ax[5].imshow(data[5].reshape(dim,dim))
# ax[6].imshow(data[6].reshape(dim,dim))
# ax[7].imshow(data[7].reshape(dim,dim))

train_rbm(rbm, nb_epoch, data)
"""
epoch: 0 loss: -0.024943181818181823
epoch: 1 loss: -0.14437500000000003
epoch: 2 loss: -0.27863636363636357
epoch: 3 loss: -0.20420454545454544
epoch: 4 loss: -0.3021590909090909

-0.3021590909090909
"""

epoch: 0 loss: 7.57575757575775e-05
epoch: 1 loss: 0.0015909090909090949
epoch: 2 loss: 0.0043939393939393945
epoch: 3 loss: -0.003939393939393942
epoch: 4 loss: 0.004696969696969694
epoch: 5 loss: 0.0024242424242424195
epoch: 6 loss: 0.00022727272727273253
epoch: 7 loss: 0.0012878787878787899
epoch: 8 loss: -0.009242424242424245
epoch: 9 loss: 0.06075757575757576

0.06075757575757576

rbm.network

{'hbias': array([ 0.42966332, -0.11780132]),
 'lateral_hidden_weights': array([[0.90426376],
        [0.7010269 ]]),
 'lateral_visible_weights': array([[0.35606778, 0.19139631, 0.86572489, 0.49474599, 0.08331228,
         0.86527945, 0.62645683, 0.75081491],
        [0.72350673, 0.13811848, 0.75951941, 0.72089064, 0.41932047,
         0.71248457, 0.7145043 , 0.5986508 ],
        [0.45146785, 0.01724366, 0.85912676, 0.09513311, 0.19898541,
         0.22687998, 0.64005706, 0.96278362],
        [0.84215956, 0.33887954, 0.18209816, 0.08490477, 0.62180297,
         0.64225146, 0.7065625 , 0.24604646],
        [0.44132804, 0.58864584, 0.59825964, 0.09512128, 0.72512485,
         0.64573849, 0.70651728, 0.5314822 ],
        [0.33340497, 0.55307233, 0.12048014, 0.72628656, 0.35775214,
         0.89983478, 0.02178698, 0.8224408 ],
        [0.85928316, 0.92429754, 0.04246374, 0.90646773, 0.13434093,
         0.95559901, 0.34124218, 0.58868465],
        [0.45375006, 0.10185529, 0.9745919 , 0.30442837, 0.33637468,
         0.60064328, 0.77644923, 0.83470712],
        [0.52585177, 0.11875218, 0.66840791, 0.36272451, 0.78696463,
         0.06185352, 0.77351959, 0.40023822]]),
 'vbias': array([-0.09945093,  0.13272422,  0.37303506,  0.07340731,  0.2585682 ,
        -0.46699447,  0.35013263, -0.00119738,  0.28911578]),
 'visible_to_hidden_weights': array([[-0.50495211,  0.24562472],
        [-0.15338154,  0.1585741 ],
        [ 0.32349785, -0.06340481],
        [-0.44458475, -0.25654652],
        [ 0.26633961,  0.23983482],
        [-0.3235362 , -0.19884048],
        [-0.02423753, -0.3203886 ],
        [ 0.03597051,  0.2032059 ],
        [ 0.13933185, -0.01206328]])}

fig, ax=plt.subplots(ncols=3)
ax[0].imshow(rbm.network['vbias'].reshape(dim,dim))
ax[1].imshow(rbm.network['visible_to_hidden_weights'][:,0].reshape(dim,dim))
ax[2].imshow(rbm.network['visible_to_hidden_weights'][:,1].reshape(dim,dim))

_,H=rbm.sample_h_fixed(np.array([1,1]),num_runs=100)

fig, ax = plt.subplots(ncols=4)
for i in range(4):
    ax[i].imshow(H.record[i][0][:9].reshape(3,3))


""" Boltzmann machine useful papers:
Roux, N. L. & Bengio, Y. Representational power of restricted Boltzmann machines and deep belief networks. Neural Comput. 20, 1631–1649 (2008).
Restricted Boltzmann machines in quantum physics
Quantum Boltzmann Machine, mohammad H amiin """
