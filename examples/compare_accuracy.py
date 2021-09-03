import os
import torch


directory = './bas/BAS88_beta4_noscale_312_seed0_Adv'

b_log = torch.load(f"./{directory}/quantum_b.pt")
c_log = torch.load(f"./{directory}/quantum_c.pt")
W_log = torch.load(f"./{directory}/quantum_W.pt")
err_log = torch.load(f"./{directory}/quantum_err.pt")
accuracy_log = torch.load(f"./{directory}/quantum_accuracy.pt")
embedding = torch.load(f"./{directory}/embedding.pt")
embedding_orig = torch.load(f"./{directory}/embedding_orig.pt")

rbm.b.data = torch.tensor(b_log[-1])
rbm.c.data = torch.tensor(c_log[-1])
rbm.W.data = torch.tensor(W_log[-1]).view(rbm.H,rbm.V)
qa_sampler = qaml.sampler.AdachiQASampler(rbm,solver=solver_name,
                                    beta=beta,embedding=embedding_orig)


type()
