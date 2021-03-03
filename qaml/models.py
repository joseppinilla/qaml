import torch


class BoltzmannMachine(torch.nn.Module):
    r"""Boltzmann Machine.

    Args:
        n_vis (int, optional): The size of visible layer
        n_hid (int, optional): The size of hidden layer
    """

    def __init__(self, V_in, H_out):
        super(BM, self).__init__()
        # Visible linear bias
        self.bv = nn.Parameter(torch.randn(1, V_in))
        # Hidden linear bias
        self.bh = nn.Parameter(torch.randn(1, H_out))
        # Visible-Hidden quadratic bias
        self.W = nn.Parameter(torch.randn(H_out, V_in))

    def get_parameters(self):
        return [self.W, self.bv, self.bh]

    def sample(self):
        return

    def forward(self, input):

        return


BM = BoltzmannMachine

class LimitedBoltzmannMachine(BoltzmannMachine):
    def __init__(self,V_in,H_out)
        pass

LBM = LimitedBoltzmannMachine(BoltzmannMachine)

class RestrictedBoltzmannMachine(BoltzmannMachine):

    def __init__():
        pass

RBM = RestrictedBoltzmannMachine
