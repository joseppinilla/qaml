import torch


device = DWaveSampler()

model = torch.nn.Sequential(RBM(),
                            torch.nn.ReLU(),
                            RBM(),
                            Linear()).to(device)
