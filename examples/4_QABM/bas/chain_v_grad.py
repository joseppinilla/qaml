import qaml
import torch
import minorminer
import matplotlib.pyplot as plt


################################# FULL #########################################
SEED = 42
BATCH_SIZE = 83
HIDDEN_SIZE = 16
TYPE = 'full'

directory = f"{TYPE}/{HIDDEN_SIZE}_batch{BATCH_SIZE}"
embedding = torch.load(f'./{TYPE}/embedding.pt')
b_log = torch.load(f'{directory}/{SEED}/b.pt')
c_log = torch.load(f'{directory}/{SEED}/c.pt')

qaml.perf.chain_grad_variance(embedding,b_log,c_log)

################################ HEUR ##########################################
SEED = 42
BATCH_SIZE = 84
HIDDEN_SIZE = 16
TYPE = 'heur'

directory = f"./{TYPE}/{HIDDEN_SIZE}_batch{BATCH_SIZE}"
embedding = torch.load(f'./{TYPE}/embedding.pt')
b_log = torch.load(f'{directory}/{SEED}/b.pt')
c_log = torch.load(f'{directory}/{SEED}/c.pt')

qaml.perf.chain_grad_variance(embedding,b_log,c_log)
