import dimod
import torch
import numpy as np

class MaximumLikelihood(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sampler, pos_phase, neg_phase, *weights):

        vartype =  sampler.model.vartype

        samples_v0, samples_h0 = pos_phase
        samples_vk, samples_hk = neg_phase

        def to_spin(samples):
            return 2.0*samples - 1.0

        if (sampler.return_prob) and (vartype is dimod.SPIN):
            samples_v0, samples_h0 = samples_v0, to_spin(samples_h0)
            samples_vk, samples_hk = to_spin(samples_vk), to_spin(samples_hk)

        expect_0 = torch.mean(samples_v0,dim=0)
        expect_k = torch.mean(samples_vk,dim=0)

        # Values for gradient
        ctx.save_for_backward(samples_v0,samples_h0,samples_vk,samples_hk)
        return torch.nn.functional.l1_loss(expect_k, expect_0, reduction='sum')

    @staticmethod
    def backward(ctx, grad_output):

        # Retrieve positive and negative phase values
        samples_v0, samples_h0, samples_vk, samples_hk = ctx.saved_tensors

        # Data batch and Visible size
        D,V = samples_v0.shape
        # Sampleset and Hidden size
        S,H = samples_hk.shape

        # forall the v ∈ S'

        #   for j = 1,...,m do
        #     \Delta b_j += v_j^{0} - v_j^{k}
        pos_v = samples_v0.mean(dim=0)
        neg_v = samples_vk.mean(dim=0)
        b_grad = -grad_output*(pos_v - neg_v)

        #   for i = 1,...,n do
        #     \Delta c_i += p(H_i = 1 | v^{0}) - p(H_i = 1 | v^{k})
        pos_h = samples_h0.mean(dim=0)
        neg_h = samples_hk.mean(dim=0)
        c_grad = -grad_output*(pos_h - neg_h)

        #   for i = 1,...,n, j = 1,...,m do
        #     \Delta vv_{ij} += v_i^{0}*v_j^{0} - v_i^{k}*v_j^{k}
        i,j = np.triu_indices(V,1)
        pos_vv = (samples_v0[:,i]*samples_v0[:,j]).sum(dim=0)/D
        neg_vv = (samples_vk[:,i]*samples_vk[:,j]).sum(dim=0)/S
        vv_grad = -grad_output*(pos_vv - neg_vv)

        #   for i = 1,...,n, j = 1,...,m do
        #     \Delta hh_{ij} += p(H_i=1|v^{0})*p(H_j=1|v^{0})
        #            - p(H_i=1|v^{k})*p(H_j=1|v^{k})
        i,j = np.triu_indices(H,1)
        pos_hh = (samples_h0[:,i]*samples_h0[:,j]).sum(dim=0)/D
        neg_hh = (samples_hk[:,i]*samples_hk[:,j]).sum(dim=0)/S
        hh_grad = -grad_output*(pos_hh - neg_hh)

        #   for i = 1,...,n, j = 1,...,m do
        #     \Delta w_{ij} += p(H_i=1|v^{0})*v_j^{0} - p(H_i=1|v^{k})*v_j^{k}
        pos_vh = torch.matmul(samples_h0.T,samples_v0)/D
        neg_vh = torch.matmul(samples_hk.T,samples_vk)/S
        W_grad = -grad_output*(pos_vh - neg_vh)

        return None, None, None, b_grad, c_grad, vv_grad, hh_grad, W_grad


class LimitedMaximumLikelihood(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sampler, pos_phase, neg_phase, *weights):

        vartype =  sampler.model.vartype

        samples_v0, samples_h0 = pos_phase
        samples_vk, samples_hk = neg_phase

        def to_spin(samples):
            return 2.0*samples - 1.0

        if (sampler.return_prob) and (vartype is dimod.SPIN):
            samples_v0, samples_h0 = samples_v0, to_spin(samples_h0)
            samples_vk, samples_hk = to_spin(samples_vk), to_spin(samples_hk)

        expect_0 = torch.mean(samples_v0,dim=0)
        expect_k = torch.mean(samples_vk,dim=0)

        # Values for gradient
        ctx.save_for_backward(samples_v0,samples_h0,samples_vk,samples_hk)
        return torch.nn.functional.l1_loss(expect_k, expect_0, reduction='sum')

    @staticmethod
    def backward(ctx, grad_output):

        # Retrieve positive and negative phase values
        samples_v0, samples_h0, samples_vk, samples_hk = ctx.saved_tensors

        # Data batch and Visible size
        D,V = samples_v0.shape
        # Sampleset and Hidden size
        S,H = samples_hk.shape

        # forall the v ∈ S'

        #   for j = 1,...,m do
        #     \Delta b_j += v_j^{0} - v_j^{k}
        pos_v = samples_v0.mean(dim=0)
        neg_v = samples_vk.mean(dim=0)
        b_grad = -grad_output*(pos_v - neg_v)

        #   for i = 1,...,n do
        #     \Delta c_i += p(H_i = 1 | v^{0}) - p(H_i = 1 | v^{k})
        pos_h = samples_h0.mean(dim=0)
        neg_h = samples_hk.mean(dim=0)
        c_grad = -grad_output*(pos_h - neg_h)

        #   for i = 1,...,n, j = 1,...,m do
        #     \Delta hh_{ij} += p(H_i=1|v^{0})*p(H_j=1|v^{0})
        #            - p(H_i=1|v^{k})*p(H_j=1|v^{k})
        i,j = np.triu_indices(H,1)
        pos_hh = (samples_h0[:,i]*samples_h0[:,j]).sum(dim=0)/D
        neg_hh = (samples_hk[:,i]*samples_hk[:,j]).sum(dim=0)/S
        hh_grad = -grad_output*(pos_hh - neg_hh)

        #   for i = 1,...,n, j = 1,...,m do
        #     \Delta w_{ij} += p(H_i=1|v^{0})*v_j^{0} - p(H_i=1|v^{k})*v_j^{k}
        pos_vh = torch.matmul(samples_h0.T,samples_v0)/D
        neg_vh = torch.matmul(samples_hk.T,samples_vk)/S
        W_grad = -grad_output*(pos_vh - neg_vh)

        return None, None, None, b_grad, c_grad, hh_grad, W_grad

class ContrastiveDivergence(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sampler, pos_phase, neg_phase, bias_v, bias_h, weights):

        vartype =  sampler.model.vartype

        samples_v0, samples_h0 = pos_phase
        samples_vk, samples_hk = neg_phase

        def to_spin(samples):
            return 2.0*samples - 1.0

        if (sampler.return_prob) and (vartype is dimod.SPIN):
            samples_v0, samples_h0 = samples_v0, to_spin(samples_h0)
            samples_vk, samples_hk = to_spin(samples_vk), to_spin(samples_hk)

        expect_0 = torch.mean(samples_v0,dim=0)
        expect_k = torch.mean(samples_vk,dim=0)

        # Values for gradient
        ctx.save_for_backward(samples_v0,samples_h0,samples_vk,samples_hk)
        return torch.nn.functional.l1_loss(expect_k, expect_0, reduction='sum')


    @staticmethod
    def backward(ctx, grad_output):

        # Retrieve positive and negative phase values
        samples_v0, samples_h0, samples_vk, samples_hk = ctx.saved_tensors

        # Data batch and Visible size
        D,V = samples_v0.shape
        # Sampleset and Hidden size
        S,H = samples_hk.shape

        # forall the v ∈ S'

        #   for j = 1,...,m do
        #     \Delta b_j += v_j^{0} - v_j^{k}
        pos_v = samples_v0.mean(dim=0)
        neg_v = samples_vk.mean(dim=0)
        b_grad = -grad_output*(pos_v - neg_v)

        #   for i = 1,...,n do
        #     \Delta c_i += p(H_i = 1 | v^{0}) - p(H_i = 1 | v^{k})
        pos_h = samples_h0.mean(dim=0)
        neg_h = samples_hk.mean(dim=0)
        c_grad = -grad_output*(pos_h - neg_h)

        #   for i = 1,...,n, j = 1,...,m do
        #     \Delta w_{ij} += p(H_i=1|v^{0})*v_j^{0} - p(H_i=1|v^{k})*v_j^{k}
        pos_vh = torch.matmul(samples_h0.T,samples_v0)/D
        neg_vh = torch.matmul(samples_hk.T,samples_vk)/S
        W_grad = -grad_output*(pos_vh - neg_vh)

        return None, None, None, b_grad, c_grad, W_grad


class AdaptiveBeta(torch.autograd.Function):
    """ Adaptive hyperparameter (beta) updating using the method from [1]. This
    is useful when dealing with a sampler that has an unknown effective inverse
    temperature (beta), such as quantum annealers.

    [1] Xu, G., Oates, W.S. Adaptive hyperparameter updating for training
    restricted Boltzmann machines on quantum annealers. Sci Rep 11, 2727 (2021).
    https://doi.org/10.1038/s41598-021-82197-1
    """
    @staticmethod
    def forward(ctx, energies_0, energies_k, beta):
        # Values for gradient
        energy_avg_0 = torch.mean(energies_0)
        energy_avg_k = torch.mean(energies_k)
        ctx.save_for_backward(energy_avg_0,energy_avg_k,beta)
        return torch.nn.functional.l1_loss(energy_avg_0,energy_avg_k,reduction='sum')

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve energy average from data and samples
        energy_avg_0, energy_avg_k, beta = ctx.saved_tensors

        beta_grad = -(energy_avg_0-energy_avg_k)/(beta**2)

        return  None, None, beta_grad
