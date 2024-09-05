import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim
from utils import ConditionalDiagGaussian, DiagGaussian
from utils import MaskedAffineAutoregressive, normalize_obs, normalize_deltas, unnormalize_deltas, Permute, set_requires_grad
# Move model on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NVP_Flows:
    # Forward KLD: inverse back, -log_q - init_log.
    # Reverse KLD: forward, + init_log - log_det, loss = (mean - beta * mean).
    # total_params = sum(p.numel() for p in self.flows.parameters())
    # print("Normalizing Flows Model One model No. Parameters: ")
    # print(total_params)
    def __init__(self, state_dim, act_dim, batch_size):
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.shape = (state_dim + act_dim,)
        self.num_models = 5
        self.trained_times = 0
        # Construct models and Optimizers.
        self.flowss = []
        # self.init = ConditionalDiagGaussian(state_dim + act_dim)
        self.q0 = DiagGaussian(state_dim + act_dim)
        for j in range(self.num_models):
            # Define list of flows
            num_layers = 32
            flows = []
            for i in range(num_layers):
                mask = MaskedAffineAutoregressive(state_dim + act_dim,
                        hidden_features=64,
                        num_blocks=1,
                        use_residual_blocks=True,
                        random_mask=False,
                        activation=F.relu,
                        dropout_probability=0.0,
                        use_batch_norm=False)
                # Swap dimensions
                flows.append(mask)
                flows.append(Permute(state_dim + act_dim, mode='swap'))
            flows = nn.ModuleList(flows)
            self.flowss.append(flows)
        self.optimizers = [optim.Adam(model.parameters(), lr=0.003) for model in self.flowss]
        self.statistics = dict()

    def set_statistics(self, statistics):
        self.statistics = statistics

    # Forward KLD: inverse back, -log_q - init_log.
    # Reverse KLD: forward, + init_log - log_det, loss = (mean - beta * mean).
    def train_net(self, world_memory):
        # Train at each
        for i in range(self.num_models):
            flows = self.flowss[i]
            current_, actions_, _, next_, _ = world_memory.sample(batch_size=self.batch_size)
            # Target is the normalized diff
            target = (next_ - current_)
            delta_targets_normalized = normalize_deltas(target, self.statistics)
            # Input is the normalized states
            normalized_obs = normalize_obs(current_, self.statistics)
            s_a = torch.cat((normalized_obs, actions_), dim=1)
            s_n_a = torch.cat((delta_targets_normalized, actions_), dim=1)

            # Forward KL Divergence.
            # z_ = s_n_a
            # log_dets = 0
            # for flow in flows:
            #     z_, log_det = flow.inverse(z_)
            #     log_dets = log_dets + log_det
            # # Init
            # log_dets += self.q0.log_prob(s_a)
            # loss = -torch.mean(log_dets)

            # Reverse KL Divergence.
            z_ = s_a
            log_dets = 0
            for flow in flows:
                z_, log_det = flow.forward(z_)
                log_dets = log_dets + log_det
            # Reverse KLD: Log_q - Log_p = (Log_q0 - forward_log_det) - (-MSE)
            log_p = -1 * F.mse_loss(z_, s_n_a, reduction="sum")
            loss = torch.mean(-log_dets - log_p)

            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()
        self.trained_times += 1

    def forward(self, states, actions):
        rand_ind = random.randint(0, self.num_models - 1)
        flows = self.flowss[rand_ind]
        normalized_obs = normalize_obs(states, self.statistics)
        s_a = torch.cat((normalized_obs, actions), dim=1)
        z_ = s_a
        for flow in flows:
            z_, _ = flow.forward(z_)
        pred = z_[:, 0:self.state_dim]
        pred = unnormalize_deltas(pred, self.statistics)
        pred = pred + states
        return pred, pred, pred

    def reverse(self, states, actions):
        normalized_obs = normalize_obs(states, self.statistics)
        s_a = torch.cat((normalized_obs, actions), dim=1)
        # Forward first
        pred_next = []
        act_error = 0
        for i in range(self.num_models):
            flows = self.flowss[i]
            z_ = s_a
            for flow in flows:
                z_, _ = flow.forward(z_)
            act_error += torch.mean(F.mse_loss(z_[:, self.state_dim:], actions)).item()
            pred_next.append(z_)
        pred_next = torch.stack(pred_next)
        # [512]
        forward_all_uncert = torch.mean(torch.sum(torch.std(pred_next, dim=0), dim=1)).item()
        forward_act_uncert = torch.mean(torch.sum(torch.std(pred_next[:, :, self.state_dim:], dim=0), dim=1)).item()
        forward_state_uncert = torch.mean(torch.sum(torch.std(pred_next[:, :, :self.state_dim], dim=0), dim=1)).item()

        # Reverse
        reverse_curr = []
        reverse_act_error = 0
        for j in range(self.num_models):
            flows = self.flowss[j]
            # for k in range(self.num_models):
            z_ = pred_next[j]
            for flow in flows:
                z_, _ = flow.inverse(z_)
            reverse_act_error += torch.mean(F.mse_loss(z_[:, self.state_dim:], actions)).item()
            reverse_curr.append(z_)
        reverse_curr = torch.stack(reverse_curr)
        # [25, 512, 23]
        reverse_all_uncert = torch.mean(torch.sum(torch.std(reverse_curr, dim=0), dim=1)).item()
        reverse_act_uncert = torch.mean(torch.sum(torch.std(reverse_curr[:, :, self.state_dim:], dim=0), dim=1)).item()
        reverse_state_uncert = torch.mean(torch.sum(torch.std(reverse_curr[:, :, :self.state_dim], dim=0), dim=1)).item()

        return (act_error, forward_all_uncert, forward_act_uncert, forward_state_uncert,
                reverse_act_error, reverse_all_uncert, reverse_act_uncert, reverse_state_uncert)

