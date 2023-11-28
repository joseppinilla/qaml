# Required packages
import os
import qaml
import torch
import minorminer
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt

def plot_quality(directory,label,solver_graph):
    _ = plt.figure()

    abspath = os.path.abspath(f'../{directory}')
    if not os.path.isdir(abspath): print(f"Not found: {abspath}"); return

    for seed in os.listdir(abspath):
        emb_filepath = f"{abspath}/{seed}/embedding.pt"

        if seed.isnumeric():
            if os.path.exists(emb_filepath): emb = torch.load(emb_filepath)
            else: continue

        graph_filepath = f"{abspath}/{seed}/graph.pt"
        if os.path.exists(graph_filepath):
            if os.path.exists(graph_filepath): S = torch.load(f"{abspath}/{seed}/graph.pt")
            else: continue

        miner = minorminer.miner(S,solver_graph)
        state,O,QK = miner.quality_key(emb)
        label_i = label + f'_{seed}'

        plt.bar(QK[0::2], QK[1::2],alpha=0.5,label=label_i)
        plt.xlabel('Chain length')
        plt.ylabel('Count')
        plt.legend(framealpha=0.5)

def plot_embeddings(directory,label,solver_graph):

    abspath = os.path.abspath(f'../{directory}')
    if not os.path.isdir(abspath): print(f"Not found: {abspath}"); return

    for seed in os.listdir(abspath):
        emb_filepath = f"{abspath}/{seed}/embedding.pt"

        if seed.isnumeric():
            if os.path.exists(emb_filepath): emb = torch.load(emb_filepath)
            else: continue

        graph_filepath = f"{abspath}/{seed}/graph.pt"
        if os.path.exists(graph_filepath):
            if os.path.exists(graph_filepath): S = torch.load(f"{abspath}/{seed}/graph.pt")
            else: S = None

        _ = plt.figure(figsize=(16,16))
        dnx.draw_pegasus_embedding(solver_graph,emb,S,node_size=10)

def plot_batch_embeddings(directory,label,solver_graph):

    abspath = os.path.abspath(f'../{directory}')
    if not os.path.isdir(abspath): print(f"Not found: {abspath}"); return

    for seed in os.listdir(abspath):
        emb_filepath = f"{abspath}/{seed}/embedding.pt"

        if seed.isnumeric():
            if os.path.exists(emb_filepath): emb = torch.load(emb_filepath)
            else: continue

        graph_filepath = f"{abspath}/{seed}/graph.pt"
        if os.path.exists(graph_filepath):
            if os.path.exists(graph_filepath): S = torch.load(f"{abspath}/{seed}/graph.pt")
            else: S = None

        _ = plt.figure(figsize=(16,16))
        dnx.draw_pegasus_embedding(solver_graph,emb,S,node_size=10)

######################### Heuristic and Systematic #############################
EXPERIMENT = "Embedding-Heuristic_64x64"
SOLVER_NAME = "Advantage_system4.1"
SUBDIR = "3_QARBM/optdigits/64x64"
PLOT_DATA = [(f'{SUBDIR}/vanilla','Sys'),
             (f'{SUBDIR}/heuristic','Heur')]
SOLVER_GRAPH = torch.load(f'./Architectures/{SOLVER_NAME}.pt')

if not os.path.exists(f"./{EXPERIMENT}/"): os.makedirs(f"./{EXPERIMENT}/")

for directory,label in PLOT_DATA:
    plot_quality(directory,label,SOLVER_GRAPH)
    plt.savefig(f'./{EXPERIMENT}/quality_{label}.svg')

for directory,label in PLOT_DATA:
    plot_embeddings(directory,label,SOLVER_GRAPH)
