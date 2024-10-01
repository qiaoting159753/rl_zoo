import torch
from torch import nn
import torch.nn.functional as F


class NormalInvGamma(nn.Module):
    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units * 4)
        self.out_units = out_units

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        out = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return mu, v, alpha, beta


# Normal Inverse Gamma Negative Log-Likelihood
# from https://arxiv.org/abs/1910.02600:
# > we denote the loss, L^NLL_i as the negative logarithm of model
# > evidence ...
def nig_nll(gamma, v, alpha, beta, y):
    two_beta_lambda = 2 * beta * (1 + v)
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    nll = t1 - t2 + t3 + t4 - t5
    return nll.mean()


# Normal Inverse Gamma regularization
# from https://arxiv.org/abs/1910.02600:
# > we formulate a novel evidence regularizer, L^R_i
# > scaled on the error of the i-th prediction
def nig_reg(gamma, v, alpha, _beta, y):
    reg = (y - gamma).abs() * (2 * v + alpha)
    return reg.mean()

def evidential_regression(dist_params, y, lamb=1.0):
    return nig_nll(*dist_params, y) + lamb * nig_reg(*dist_params, y)