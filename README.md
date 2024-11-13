# Quantum Assisted Machine Learning
QAML is a library for generative learning models built as an extension of PyTorch.
This library provides a set of custom Pytorch classes based on Pytorch `Module`, `Function`, and `Dataset` to allow easy
integration with Pytorch models.

This compatibility and modularity is what makes possible the use of Quantum Processing Units to accelerate
the sampling process required in generative models.

# QAML Configurations:

- Classic Restricted Boltzmann Machine (RBM)

- Quantum-Assisted Restricted Boltzmann Machine (QARBM)

- Classic Boltzmann Machine (BM)

- Quantum-Assisted Boltzmann Machine (QABM)

# Code Example

``` python
import qaml
import torch

# Model
bm = qaml.nn.BoltzmannMachine(36,16)

# Optimizer
opt = torch.optim.SGD(bm.parameters(), ...) # Hyperparameters

# Positive-phase Sampler
SAMPLER_NAME = "Advantage_system4.1" # Example of D-Wave Sampler
pos = qaml.sampler.BatchQASampler(bm,SAMPLER_NAME,vmask=True)
POS_BATCH = len(pos_sampler.batch_embeddings)

# Negative-phase Sampler
neg = qaml.sampler.BatchQASampler(bm,SAMPLER_NAME)
NEG_BATCH = len(neg_sampler.batch_embeddings)

# Automatic Differentiation
ML = qaml.autograd.MaximumLikelihood

# Dataset
data = qaml.datasets.BAS(6,6)
loader = torch.utils.data.DataLoader(data,POS_BATCH)

# Training loop
err_log = []
for epoch in range(N):
    epoch_error = torch.Tensor([0.])
    for img_batch in loader:

        # Positive Phase
        v0, h0 = pos(img_batch.detach(),num_reads=100)
        
        # Negative Phase
        vq, hq = neg(num_reads=1000)

        # Reconstruction error from Contrastive Divergence
        err = ML.apply((v0,h0),(vq,hq), *bm.parameters())

        # Do not accumulate gradients
        optimizer.zero_grad()

        # Compute gradients
        err.backward()

        # Update parameters
        optimizer.step()

        #Accumulate error for this epoch
        epoch_error  += err
    # Error Log
    err_log.append(epoch_error.item())
    print(f"Epoch {t} Recon Error = {epoch_error.item()}")
```

# References:

I want to acknowledge the projects I used as inspiration for QAML.

[Gabriel Bianconi's pytorch-rbm](https://github.com/GabrielBianconi/pytorch-rbm): Perfect example of a baseline RBM implementation along with MNIST example.

[Jan Melchior's PyDeep](https://github.com/MelJan/PyDeep.git):  Most thorough Boltzmann Machine learning library out there with impressive modularity.

[PIQUIL's QuCumber](https://github.com/PIQuIL/QuCumber): Practical application with a PyTorch RBM backend.
