import torch
import warnings
import torch.nn.functional as F


class Sampler(torch.nn.Module):
    def __init__(self, model):
        super(Sampler, self).__init__()
        self.model = model
        # Sampler stores states
        visible_unknown = torch.Tensor([float('NaN')]*model.V)
        self.prob_vk = torch.nn.Parameter(visible_unknown, requires_grad=False)

        hidden_unknown = torch.Tensor([float('NaN')]*model.H)
        self.prob_hk = torch.nn.Parameter(hidden_unknown, requires_grad=False)

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


class GibbsSampler(Sampler):

    def __init__(self, model):
        super(GibbsSampler, self).__init__(model)

    def forward(self, vk, k=1):
        model = self.model
        prob_vk = vk
        prob_hk = self.model(vk)

        for _ in range(k):
            prob_vk = self.model.generate(prob_hk)
            vk.data = prob_vk.bernoulli()
            prob_hk = self.model.forward(vk)

        self.prob_vk.data = prob_vk
        self.prob_hk.data = prob_hk
        return prob_hk
