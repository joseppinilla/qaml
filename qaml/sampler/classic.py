import neal
import torch

from dwave.preprocessing import ScaleComposite
from qaml.sampler.base import NetworkSampler, BinaryQuadraticModelNetworkSampler

__all__ = ['GibbsNetworkSampler',
           'SimulatedAnnealingNetworkSampler', 'SASampler']

class GibbsNetworkSampler(NetworkSampler):
    """ Sampler for k-step Contrastive Divergence training of RBMs

        Args:
            model (torch.nn.Module): PyTorch `Module` with `forward` method.
            beta (float): inverse temperature for the sampler.
            return_prob (bool, default=True): if True, returns probabilities.
        Example for Contrastive Divergence (CD-k):
            >>> pos_sampler = neg_sampler = GibbsNetworkSampler(rbm,BATCH_SIZE)
            ...
            >>> # Positive Phase
            >>> v0, prob_h0 = gibbs_sampler(input_data,k=0)
            >>> # Negative Phase
            >>> prob_vk, prob_hk = gibbs_sampler(input_data,k)

        Example for Persistent Contrastive Divergence (PCD):
            >>> pos_sampler = qaml.sampler.GibbsNetworkSampler(rbm,BATCH_SIZE)
            >>> neg_sampler = qaml.sampler.GibbsNetworkSampler(rbm,NUM_CHAINS)
            ...
            >>> # Positive Phase
            >>> v0, prob_h0 = pos_sampler(input_data,k=0)
            >>> # Negative Phase
            >>> prob_vk, prob_hk = neg_sampler(k)

    """

    def __init__(self, model, num_chains, beta=1.0, return_prob=True):
        if 'Restricted' not in repr(model):
            raise RuntimeError("Not Supported")
        super(GibbsNetworkSampler, self).__init__(model,beta)
        self.return_prob = return_prob
        self.prob_v.data = torch.rand(num_chains,model.V)
        self.prob_h.data = torch.rand(num_chains,model.H)

    @torch.no_grad()
    def reconstruct(self, input_data, mask=None, k=1):
        beta = self.beta
        model = self.model

        if mask is None: mask = torch.ones_like(self.prob_v)
        # Masked bits are filled with 0.5 if BINARY(0,1) or 0.0 if SPIN(-1,+1)
        mask_value = sum(model.vartype.value)/2.0

        clamp = torch.mul(input_data.detach().clone(),mask)
        prob_v = clamp.clone().masked_fill_((mask==0),mask_value)
        prob_h = model.forward(prob_v,scale=beta)
        for _ in range(k):
            recon = model.generate(self.sample(prob_h),scale=beta)
            prob_v.data = clamp + (mask==0)*recon
            prob_h.data = model.forward(self.sample(prob_v),scale=beta)

        return prob_v.clone(), prob_h.clone()

    @torch.no_grad()
    def forward(self, input_data=None, k=1):
        beta = self.beta
        model = self.model

        if input_data is not None:
            self.prob_v.data = input_data.clone()
            self.prob_h.data = model.forward(input_data.clone(),scale=beta)

        for _ in range(k):
            self.prob_v.data = model.generate(self.sample_h(),scale=beta)
            self.prob_h.data = model.forward(self.sample_v(),scale=beta)

        if self.return_prob:
            return self.prob_v.clone(), self.prob_h.clone()
        return self.sample_v(), self.sample_h()

""" The next samplers formulate the model as a Binary Quadratic Model (BQM) """
class SimulatedAnnealingNetworkSampler(BinaryQuadraticModelNetworkSampler):
    sample_kwargs = {"num_reads":100, "num_sweeps":1000}

    def __init__(self, model, beta=1.0, **kwargs):
        BinaryQuadraticModelNetworkSampler.__init__(self,model,beta)
        self.child = ScaleComposite(neal.SimulatedAnnealingSampler(**kwargs))

SASampler = SimulatedAnnealingNetworkSampler

#TODO: Parallel version of SimulatedAnnealingSampler. One thread per input.
# def sample_bm(self, fixed_vars=[], **kwargs):
