import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from agents.networks.world_models.bayesian.bayesian_javirantoran_util import (isotropic_gauss_loglike,
                                                                              isotropic_gauss_prior,
                                                                              gaussian)

class BayesLinear_Normalq(nn.Module):
    """Linear Layer where weights are sampled from a fully factorised Normal with learnable parameters. The likelihood
     of the weight samples under the prior and the approximate posterior are returned with each forward pass in order
     to estimate the KL term in the ELBO.
    """
    def __init__(self, n_in, n_out, prior_class):
        super(BayesLinear_Normalq, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior = prior_class
        # Learnable parameters -> Initialisation is set empirically.
        self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.2, 0.2))
        self.W_p = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-3, -1))
        self.b_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-0.2, 0.2))
        self.b_p = nn.Parameter(torch.Tensor(self.n_out).uniform_(-3, -1))
        self.lpw = 0
        self.lqw = 0

    def forward(self, X, sample=False):
        if not sample:
            # Not Updating teh probability part.
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_out)
            return output, 0, 0
        else:
            eps_W = Variable(self.W_mu.data.new(self.W_mu.size()).normal_())
            eps_b = Variable(self.b_mu.data.new(self.b_mu.size()).normal_())
            # sample parameters
            std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20)
            std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20)
            W = self.W_mu + 1 * std_w * eps_W
            b = self.b_mu + 1 * std_b * eps_b
            output = torch.mm(X, W) + b.unsqueeze(0).expand(X.shape[0], -1)  # (batch_size, n_output)
            lqw = isotropic_gauss_loglike(W, self.W_mu, std_w) + isotropic_gauss_loglike(b, self.b_mu, std_b)
            lpw = self.prior.loglike(W) + self.prior.loglike(b)
            return output, lqw, lpw


class bayes_linear_vi(nn.Module):
    """2 hidden layer Bayes By Backprop (VI) Network"""
    def __init__(self, input_dim, output_dim, n_hid, sigma):
        super(bayes_linear_vi, self).__init__()
        # self.prior_instance = isotropic_gauss_prior(mu=mu, sigma=sigma)
        self.prior_instance = isotropic_gauss_prior(0.0, sigma)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bfc1 = BayesLinear_Normalq(input_dim, n_hid, self.prior_instance)
        self.bfc2 = BayesLinear_Normalq(n_hid, n_hid, self.prior_instance)
        self.bfc4 = BayesLinear_Normalq(n_hid, output_dim, self.prior_instance)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, sample=True):
        tlqw = 0
        tlpw = 0
        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x, lqw, lpw = self.bfc1(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        # -----------------
        x = self.act(x)
        # -----------------
        x, lqw, lpw = self.bfc2(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        # -----------------
        x = self.act(x)
        # -----------------
        y, lqw, lpw = self.bfc4(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        return y, tlqw, tlpw

    def sample_predict(self, x, Nsamples):
        """Used for estimating the data's likelihood by approximately marginalising the weights with MC"""
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        tlqw_vec = np.zeros(Nsamples)
        tlpw_vec = np.zeros(Nsamples)
        for i in range(Nsamples):
            y, tlqw, tlpw = self.forward(x, sample=True)
            predictions[i] = y
            tlqw_vec[i] = tlqw
            tlpw_vec[i] = tlpw
        return predictions, tlqw_vec, tlpw_vec
