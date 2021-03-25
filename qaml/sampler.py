import torch
import dimod
import warnings
import torch.nn.functional as F

class NetworkSampler(torch.nn.Module):
    def __init__(self, model):
        super(NetworkSampler, self).__init__()
        self.model = model
        # Sampler stores states
        visible_unknown = torch.Tensor([float('NaN')]*model.V)
        self.prob_v = torch.nn.Parameter(visible_unknown, requires_grad=False)

        hidden_unknown = torch.Tensor([float('NaN')]*model.H)
        self.prob_h = torch.nn.Parameter(hidden_unknown, requires_grad=False)

    @property
    def sample_visible(self):
        try:
            return self.prob_vk.bernoulli()
        except RuntimeError as e:
            warnings.warn(f"Invalid probability vector: {self.prob_vk}")
            return torch.zeros_like(self.prob_vk)

    @property
    def sample_hidden(self):
        try:
            return self.prob_hk.bernoulli()
        except RuntimeError as e:
            warnings.warn(f"Invalid probability vector: {self.prob_vk}")
            return torch.zeros_like(self.prob_hk)

class PersistentGibbsNetworkSampler(NetworkSampler):
    """ Sampler for Persistent Constrastive Divergence training with k steps.

        Args:
            model (torch.nn.Module): PyTorch `Module` with `forward` method.

            num_chains (int): PCD keeps N chains at all times. This number must
                match the batch size.


    """
    def __init__(self, model, num_chains):
        super(PersistentGibbsNetworkSampler, self).__init__(model)
        self.prob_v.data = torch.rand(num_chains,model.V)

    def forward(self, num_samples, k=1):
        vk = self.prob_v
        prob_hk = self.model(vk)

        for _ in range(k):
            prob_vk = self.model.generate(prob_hk)
            vk.data = prob_vk.bernoulli()
            prob_hk = self.model.forward(vk)

        self.prob_v.data = prob_vk
        self.prob_h.data = prob_hk
        return vk[:num_samples], prob_hk[:num_samples]

class GibbsNetworkSampler(NetworkSampler):

    def __init__(self, model):
        super(GibbsNetworkSampler, self).__init__(model)

    def forward(self, v0, k=1):
        vk = v0.clone()
        prob_hk = self.model(v0)

        for _ in range(k):
            prob_vk = self.model.generate(prob_hk)
            vk.data = prob_vk.bernoulli()
            prob_hk = self.model.forward(vk)

        self.prob_v.data = prob_vk
        self.prob_h.data = prob_hk
        return prob_vk, prob_hk
