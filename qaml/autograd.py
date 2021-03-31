import torch

class ConstrastiveDivergence(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pos_phase, neg_phase, bias_v, bias_h, weights):
        v0, prob_h0 = pos_phase
        prob_vk, prob_hk = neg_phase
        # Values for gradient
        ctx.save_for_backward(v0, prob_h0, prob_vk, prob_hk)
        return torch.nn.functional.mse_loss(v0, prob_vk, reduction='sum')


    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve positive and negative phase values
        v0, prob_h0, prob_vk, prob_hk = ctx.saved_tensors

        # Data batch size
        D = len(v0)

        # for j = 1,...,m do
        #     \Delta a_j += v_j^{0} - v_j^{k}
        v_grad = -grad_output*torch.mean(v0 - prob_vk, dim=0)

        # for i = 1,...,n do
        #     \Delta b_i += p(H_i = 1 | v^{0}) - p(H_i = 1 | v^{k})
        h_grad = -grad_output*torch.mean(prob_h0 - prob_hk, dim=0)

        # for i = 1,...,n, j = 1,...,m do
        #     \Delta w_{ij} += p(H_i=1|v^{0})*v_j^{0} - p(H_i=1|v^{k})*v_j^{k}
        W_grad = -grad_output*(prob_h0.t().mm(v0) - prob_hk.t().mm(prob_vk))/D

        return None, None, v_grad, h_grad, W_grad

class SampleBasedConstrastiveDivergence(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pos_phase, neg_phase, bias_v, bias_h, weights):
        samples_v0, samples_h0 = pos_phase
        samples_vk, samples_hk = neg_phase

        expect_0 = torch.mean(samples_v0, dim=0)
        expect_k = torch.mean(samples_vk, dim=0)

        # Values for gradient
        ctx.save_for_backward(samples_v0, samples_h0, samples_vk, samples_hk)
        return torch.nn.functional.l1_loss(expect_k, expect_0, reduction='sum')


    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve positive and negative phase values
        samples_v0, samples_h0, samples_vk, samples_hk = ctx.saved_tensors

        # Data batch size
        D = len(samples_v0)
        # Sampleset size
        S = len(samples_vk)

        v_grad = -grad_output*(torch.mean(samples_v0, dim=0) - torch.mean(samples_vk, dim=0))

        h_grad = -grad_output*(torch.mean(samples_h0,dim=0) - torch.mean(samples_hk, dim=0))

        W_grad = -grad_output*(samples_h0.t().mm(samples_v0)/D - samples_hk.t().mm(samples_vk)/S)

        return None, None, v_grad, h_grad, W_grad
