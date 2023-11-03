import qaml
import torch
import minorminer

SEED = 42
torch.manual_seed(SEED)

solver_name = "Advantage_system6.2"
device = qaml.sampler.QASampler.get_device(solver_name=solver_name)
torch.save(device.to_networkx_graph(),'./target.pt')

VISIBLE_SIZE = 13*13
HIDDEN_SIZE = 7
bm = qaml.nn.BoltzmannMachine(VISIBLE_SIZE,HIDDEN_SIZE,'SPIN')
torch.save(bm.to_networkx_graph(),'./source.pt')

############################# Heuristic Embedding ##############################
embedding = qaml.minor.miner_heuristic(bm,device,seed=SEED)
torch.save(embedding,'./embedding_heur_176.pt')

############################ Systematic Embedding ##############################
sys_sampler = qaml.sampler.QASampler(bm)
torch.save(dict(sys_sampler.embedding),'./embedding_sys_176.pt')

############################### Batch Embeddings ###############################
pos_sampler = qaml.sampler.BatchQASampler(bm,mask=torch.ones(bm.V))
torch.save(pos_sampler.batch_embeddings,'./embedding_sys_batch.pt')
