from .bayesian_javirantoran_util import *
import torch
import torch.nn.functional as F
import torch.nn as nn


def KLD_cost(mu_p, sig_p, mu_q, sig_q):
    KLD = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    # https://arxiv.org/abs/1312.6114 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    if torch.any(torch.isnan(KLD)):
        raise Exception("Bayesian Local Reparameterization KLD NaN error")
    return KLD


class BayesLinear_local_reparam(nn.Module):
    """Linear Layer where activations are sampled from a fully factorised normal which is given by aggregating
     the moments of each weight's normal distribution. The KL divergence is obtained in closed form. Only works
      with gaussian priors.
    """

    def __init__(self, n_in, n_out, sigma):
        super(BayesLinear_local_reparam, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior_sigma = sigma
        # Learnable parameters
        self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.1, 0.1))
        self.W_p = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-3, -2))
        self.b_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-0.1, 0.1))
        self.b_p = nn.Parameter(torch.Tensor(self.n_out).uniform_(-3, -2))

    def forward(self, X, sample=True):
        # # calculate std
        # calculate std
        std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20)
        std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20)
        act_W_mu = torch.mm(X, self.W_mu)  # self.W_mu + std_w * eps_W
        act_W_std = torch.sqrt(torch.mm(X.pow(2), std_w.pow(2)))
        # Tensor.new()  Constructs a new tensor of the same data type as self tensor.
        # the same random sample is used for every element in the minibatch output
        eps_W = Variable(self.W_mu.data.new(act_W_std.size()).normal_(mean=0, std=1))
        eps_b = Variable(self.b_mu.data.new(std_b.size()).normal_(mean=0, std=1))
        act_W_out = act_W_mu + act_W_std * eps_W  # (batch_size, n_output)
        act_b_out = self.b_mu + std_b * eps_b
        output = act_W_out + act_b_out.unsqueeze(0).expand(X.shape[0], -1)
        KL_loss = (KLD_cost(mu_p=0.0, sig_p=self.prior_sigma, mu_q=self.W_mu, sig_q=std_w)
                   + KLD_cost(mu_p=0.0, sig_p=self.prior_sigma, mu_q=self.b_mu, sig_q=std_b))

        # # sample gaussian noise for each weight and each bias
        # weight_epsilons = Variable(self.W_mu.data.new(self.W_mu.size()).normal_())
        # bias_epsilons = Variable(self.b_mu.data.new(self.b_mu.size()).normal_())
        # # calculate the weight and bias stds from the rho parameters
        # weight_stds = torch.log(1 + torch.exp(self.W_p))
        # bias_stds = torch.log(1 + torch.exp(self.b_p))
        # # calculate samples from the posterior from the sampled noise and mus/stds
        # weight_sample = self.W_mu + weight_epsilons * weight_stds
        # bias_sample = self.b_mu + bias_epsilons * bias_stds
        # output = torch.mm(X, weight_sample) + bias_sample
        # # computing the KL loss term
        # prior_cov, varpost_cov = self.prior_sigma ** 2, weight_stds ** 2
        # KL_loss = 0.5 * (torch.log(prior_cov / varpost_cov)).sum() - 0.5 * weight_stds.numel()
        # KL_loss = KL_loss + 0.5 * (varpost_cov / prior_cov).sum()
        # KL_loss = KL_loss + 0.5 * ((self.W_mu - 0.0) ** 2 / prior_cov).sum()
        # prior_cov, varpost_cov = self.prior_sigma ** 2, bias_stds ** 2
        # KL_loss = KL_loss + 0.5 * (torch.log(prior_cov / varpost_cov)).sum() - 0.5 * bias_stds.numel()
        # KL_loss = KL_loss + 0.5 * (varpost_cov / prior_cov).sum()
        # KL_loss = KL_loss + 0.5 * ((self.b_mu - 0.0) ** 2 / prior_cov).sum()
        return output, KL_loss, 0.0


class bayes_linear_lr(nn.Module):
    def __init__(self, input_dim, output_dim, nhid, sigma):
        super(bayes_linear_lr, self).__init__()
        n_hid = nhid
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bfc1 = BayesLinear_local_reparam(input_dim, n_hid, sigma)
        self.bfc2 = BayesLinear_local_reparam(n_hid, n_hid, sigma)
        self.bfc3 = BayesLinear_local_reparam(n_hid, output_dim, sigma)
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
        y, lqw, lpw = self.bfc3(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        return y, tlqw, tlpw

    def sample_predict(self, x, Nsamples):
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
