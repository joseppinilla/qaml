import torch
import torch.nn.functional as F

class ConstrastiveDivergence(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pos_phase, neg_phase, bias_v, bias_h, weights):
        v0, prob_h0 = pos_phase
        prob_vk, prob_hk = neg_phase
        # Value for gradient
        ctx.save_for_backward(v0, prob_h0, prob_vk, prob_hk)
        return torch.nn.functional.mse_loss(v0, prob_vk, reduction='sum')


    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve positive and negative phase values
        v0, prob_h0, prob_vk, prob_hk = ctx.saved_tensors

        # for j = 1,...,m do
        #     \Delta a_j += v_j^{0} - v_j^{k}
        v_grad = -grad_output*torch.mean(v0 - prob_vk, dim=0)

        # for i = 1,...,n do
        #     \Delta b_i += p(H_i = 1 | v^{0}) - p(H_i = 1 | v^{k})
        h_grad = -grad_output*torch.mean(prob_h0 - prob_hk, dim=0)

        # for i = 1,...,n, j = 1,...,m do
        #     \Delta w_{ij} += p(H_i=1|v^{0})*v_j^{0} - p(H_i=1|v^{k})*v_j^{k}
        W_grad = -grad_output*(prob_h0.t().mm(v0) - prob_hk.t().mm(prob_vk))/len(v0)

        return None, None, v_grad, h_grad, W_grad
