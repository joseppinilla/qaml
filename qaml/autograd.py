import torch
import torch.nn.functional as F

# class PersistentConstrastiveDivergence(torch.autograd.Function)
#    v_k = 0
    # @staticmethod
    # def forward(ctx, input, bias_v, bias_h, weights):


class ConstrastiveDivergence(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, bias_v, bias_h, weights):

        # Positive Phase
        p_h_0 = torch.sigmoid(F.linear(input, weights, bias_h))

        # Negative Phase
        p_h_k = p_h_0.clone()
        for _ in range(2):
            p_v_k = torch.sigmoid(F.linear(p_h_k, weights.t(), bias_v))
            v_k = p_v_k.bernoulli()
            p_h_k = torch.sigmoid(F.linear(v_k, weights, bias_h))

        # Value for gradient
        ctx.save_for_backward(input, p_h_0, v_k, p_h_k)
        return torch.nn.functional.mse_loss(input, p_v_k, reduction='sum')


    @staticmethod
    def backward(ctx, grad_output):
        # Retrueve positive and negative phase values
        v_0, p_h_0, v_k, p_h_k = ctx.saved_tensors

        # for j = 1,...,m do
        #     v_j^{(0)} - v_j^{(k)}
        v_grad = -(v_0 - v_k)

        # for i = 1,...,n do
        #     p(H_i = 1 | v^{(0)}) - p(H_i = 1 | v^{(k)})
        h_grad = -(p_h_0 - p_h_k)

        # for i = 1,...,n, j = 1,...,m do
        #     p(H_i = 1 | v^{(0)})*v_j^{(0)} - p(H_i = 1 | v^{(k)})*v_j^{(k)}
        W_grad = -(p_h_0.t().mm(v_0) - p_h_k.t().mm(v_k))

        return None, v_grad, h_grad, W_grad


"""
>>> from torchviz import make_dot
>>>
>>> N = 4
>>> M = 8
>>> # Create random Tensors to hold input and output
>>> x = torch.randn(10,N)
>>>
>>> # Create random Tensors for weights.
>>> v = torch.randn(1,N, requires_grad=True)
>>> h = torch.randn(1,M, requires_grad=True)
>>> W = torch.randn(M, N, requires_grad=True)
>>>
>>> CD = ConstrastiveDivergence()
>>> ll = CD.apply(x,v,h,W)
>>>
>>> ll.backward(retain_graph=True)
>>>
>>> make_dot(ll, params= {'v<sub>i</sub>':v,'h<sub>j</sub>':h,'w<sub>ij</sub>':W})
"""
