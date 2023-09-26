# Required packages
import os
import qaml
import torch
import minorminer
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt

filename = "Optdigits_RBM(64,64).svg"
solver_name = "Advantage_system4.1"

def load_graph(): torch.load(f"{abspath}/{seed}/graph.pt")
def load_embedding(): torch.load(f"{abspath}/{seed}/embedding.pt")

sampler = qaml.sampler.QASampler.get_sampler(solver=solver_name)
T = sampler.to_networkx_graph()

def plot_compare(plot_data,filename):
    for (directory,label) in plot_data:

        abspath = os.path.abspath(f'./{directory}/{solver_name}')

        S = torch.load(f"{abspath}/8/graph.pt")

        miner = minorminer.miner(S,T)

        embeddings = []
        for seed in os.listdir(abspath):
            try: embedding = torch.load(f"{abspath}/{seed}/embedding.pt")
            except: continue
            embeddings.append((seed,embedding))


        quality = [miner.quality_key(emb) for seed,emb in embeddings]
        # min_i = quality.index(min(quality))
        # seed,embedding = embeddings[min_i]
        # state,O,L = quality[min_i]
        print(label)
        for i,(seed, embedding) in enumerate(embeddings):
            state,O,L = quality[i]

            QK = {L[i]:L[i+1] for i in range(0,len(L),2)}
            _ = plt.figure(figsize=(16,16))
            dnx.draw_pegasus_embedding(T,embedding,node_size=10)
            # plt.savefig(f"{directory}_{filename}")
            plt.title(QK)
            print(f'SEED {seed}: {QK}')


# plot_data = [(<folder_name>,<label>)]
plot_data = [('vanilla','Complete'),
             ('heuristic','Heuristic'),
             ('adaptive','Adaptive'),
             ('priority','Priority'),
             ('repurpose','Repurpose'),
             ('adachi','Adachi')]

# plot_data = [('heuristic','Heur')]
plot_compare(plot_data,filename)

heur0 = {34: 1, 32: 1, 31: 4, 30: 2, 29: 3, 28: 4, 27: 3, 26: 2, 25: 3, 24: 5, 23: 9, 22: 6, 21: 11, 20: 6, 19: 8, 18: 14, 17: 17, 16: 10, 15: 8, 14: 6, 13: 3, 12: 2}
heur1 = {25: 1, 23: 3, 22: 1, 21: 1, 20: 2, 19: 6, 18: 6, 17: 18, 16: 14, 15: 20, 14: 29, 13: 22, 12: 5}
heur2 = {22: 1, 21: 1, 20: 7, 19: 5, 18: 13, 17: 11, 16: 17, 15: 27, 14: 18, 13: 23, 12: 5}


plt.bar(heur0.keys(),heur0.values(),alpha=0.5,label='heur0')
plt.bar(heur1.keys(),heur1.values(),alpha=0.5,label='heur1')
plt.bar(heur2.keys(),heur2.values(),alpha=0.5,label='heur2')
plt.xlabel('Chain length')
plt.ylabel('Count')
plt.legend()
plt.savefig('compare_embeddings_heuristic.svg')

complete = {13: 113, 12: 15}
adachi = {6: 126, 4: 1, 3: 1, 2: 1, 1: 1}
adaptive = {6: 126, 4: 1, 3: 1}
priority = {6: 126, 4: 1, 3: 1}
repurpose = {6: 126, 4: 1, 3: 1, 1: 1}


plots = [(complete,'Complete'),
         (adachi,'Adachi'),
         (adaptive,'Adaptive'),
         (priority,'Priority'),
         (repurpose,'Repurpose')]
