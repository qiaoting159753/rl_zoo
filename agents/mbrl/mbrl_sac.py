"""
A MBRL class that implemented all MBRL algorithms for SAC.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from utils import soft_update
from utils import vi
from torch.autograd import Variable
from .classifier import Generator, Discriminator


class MBRL_SAC:
    """
    A MBRL class that implemented all MBRL algorithms for SAC.
    """

    def __init__(self, actor_network, critic_network, world_model, gamma, tau,
                 state_dim, action_dim, actor_lr, critic_lr, alpha_lr, horizon,
                 use_dyna, use_critic_steve, use_critic_mve, use_actor_mve,
                 use_actor_pg, use_bound, device):

        super().__init__()
        self.batch_size = None
        self.type = "mbrl"
        # Switches
        self.use_normal = False
        self.sample_times = 512
        self.horizon = horizon
        self.use_dyna = use_dyna
        self.use_bounded_active = use_bound
        self.use_critic_steve = use_critic_steve
        self.use_critic_mve = use_critic_mve
        self.use_actor_mve = use_actor_mve
        self.use_actor_pg = use_actor_pg
        # self.dyna_use_uncertainty = use_dyna

        # Other Variables
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Actor Critic.
        self.generator = Generator(latent_variable=1,
                                   observation_size=self.state_dim,
                                   num_actions=self.action_dim)

        self.discriminator = Discriminator(observation_size=self.state_dim,
                                           num_actions=self.action_dim)

        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(),
                                               lr=0.001)
        self.optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(),
                                               lr=0.001)

        self.actor_net = actor_network.to(device)
        self.critic_net = critic_network.to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)

        self.log_alpha = torch.tensor(np.log(1.0)).float().to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(),
                                                 lr=critic_lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        # World Model
        self.world_model = world_model
        self.world_model.to(device)

    @property
    def alpha(self):
        """
        Control updating speed between rewards and other networks.
        :return:
        """
        return self.log_alpha.exp()

    # Only interact with the environment.
    def select_action_from_policy(self, state, evaluation=False,
                                  noise_scale=0):
        """
        Make decisions with the trained policy.
        # note that when evaluating this algorithm we need to select mu as action
        # _, _, action = self.actor_net.sample(state_tensor)

        :param obs:
        :param sample:
        :return:
        """
        assert len(state.shape) == 1
        self.actor_net.eval()
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(
            dim=0)
        # Evaluation
        if evaluation:
            _, _, action = self.actor_net.forward(state_tensor)
        # Exploration
        else:
            if self.use_dyna:
                # Divide action queries based on uncertainty.
                # if uncertaintu high, to env.
                # if uncertainty low, to model.
                # threshold = 0.99
                obs2 = torch.repeat_interleave(state_tensor,
                                               self.sample_times, dim=0)
                # actions, _, _ = self.actor_net.forward(obs2)
                random_actions = []
                for _ in range(obs2.shape[0]):
                    # pred_act = self.action_space.sample()
                    pred_act = np.random.uniform(-1, 1,
                                                 (self.action_dim,))
                    random_actions.append(pred_act)
                actions = torch.FloatTensor(np.array(random_actions)).to(
                    self.device)

                # _, _, mean, var = self.world_model.pred_next_states(
                #     obs2, multi_action)
                # uncertainty2 = vi(mean, var, device=self.device)
                # ind = torch.argmax(uncertainty2)
                # action = multi_action[ind].unsqueeze(0)

                # # Dyna Training.
                pred_next, _, _, _ = self.world_model.pred_next_states(
                    obs2, actions)
                pred_reward, _ = self.world_model.pred_rewards(
                    obs2, actions)
                # pred_reward = pred_reward.detach()
                pred_not_dones = torch.FloatTensor(
                    np.ones(pred_reward.shape)).to(
                    self.device)

                # Exploring.
                # prob2 = F.softmax(torch.squeeze(uncertainty2), dim=0)
                # new_dist = torch.distributions.Categorical(prob2)
                # candi = new_dist.sample(torch.Size([1])).squeeze()
                # uncert_actions = multi_action[candi]
                # action = uncert_actions.unsqueeze(0)
                # action = multi_action[0].unsqueeze(0)

                self.train_policy(
                    (obs2, actions.detach(), pred_reward.detach(),
                     pred_next.detach(), pred_not_dones, None,
                     None))

            action, _, _ = self.actor_net.forward(state_tensor)
            action = action.detach()

        assert action.ndim == 2 and action.shape[0] == 1
        action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    def update_critic(self, obs, actions, rewards, next_obs, not_dones):
        """
        Update the critic first, the critic have to learn the correct action.
        :param obs:
        :param actions:
        :param rewards:
        :param next_obs:
        :param not_dones:
        """
        with torch.no_grad():
            if self.use_critic_steve:
                # Horizon = 0
                # For next episodes used
                pred_all_next_obs = next_obs.unsqueeze(dim=0)
                pred_all_next_rewards = torch.zeros(rewards.shape).unsqueeze(
                    dim=0)
                means = []
                vars = []
                for hori in range(self.horizon):
                    horizon_rewards_list = []
                    horizon_obs_list = []
                    horizon_q_list = []
                    # For each state batch [256, 17], reward extend 5 times,
                    # next extend 5 time.
                    for stat in range(pred_all_next_obs.shape[0]):
                        pred_action, pred_log_pi, _ = self.actor_net(
                            pred_all_next_obs[stat])
                        pred_q1, pred_q2 = self.target_critic_net.sample(
                            pred_all_next_obs[stat], pred_action)
                        pred_q3, pred_q4 = self.critic_net.sample(
                            pred_all_next_obs[stat], pred_action)
                        pred_q1 -= self.alpha.detach() * pred_log_pi
                        pred_q2 -= self.alpha.detach() * pred_log_pi
                        pred_q3 -= self.alpha.detach() * pred_log_pi
                        pred_q4 -= self.alpha.detach() * pred_log_pi
                        # Predict a set of reward first
                        _, pred_rewards = self.world_model.pred_rewards(
                            obs=pred_all_next_obs[stat],
                            actions=pred_action)

                        if hori < (self.horizon - 1):
                            _, pred_obs, _, _ = self.world_model.pred_next_states(
                                pred_all_next_obs[stat], pred_action)
                            horizon_obs_list.append(pred_obs)

                        temp_disc_rewards = []
                        # For each predict reward.
                        for rwd in range(pred_rewards.shape[0]):
                            disc_pred_reward = not_dones * \
                                               (self.gamma ** (hori + 1)) * \
                                               pred_rewards[rwd]
                            if hori > 0:
                                # Horizon = 1, 2, 3, 4, 5
                                disc_sum_reward = pred_all_next_rewards[stat] + \
                                                  disc_pred_reward
                            else:
                                disc_sum_reward = not_dones * disc_pred_reward
                            temp_disc_rewards.append(disc_sum_reward)
                            assert rewards.shape == not_dones.shape == disc_sum_reward.shape
                            # Q = r + disc_rewards + pred_v
                            pred_tq1 = rewards + disc_sum_reward + not_dones * (
                                    self.gamma ** (hori + 2)) * pred_q1
                            pred_tq2 = rewards + disc_sum_reward + not_dones * (
                                    self.gamma ** (hori + 2)) * pred_q2
                            pred_tq3 = rewards + disc_sum_reward + not_dones * (
                                    self.gamma ** (hori + 2)) * pred_q3
                            pred_tq4 = rewards + disc_sum_reward + not_dones * (
                                    self.gamma ** (hori + 2)) * pred_q4

                            horizon_q_list.append(pred_tq1)
                            horizon_q_list.append(pred_tq2)
                            horizon_q_list.append(pred_tq3)
                            horizon_q_list.append(pred_tq4)

                        ## Observation Level
                        if hori < (self.horizon - 1):
                            temp_disc_rewards = torch.stack(temp_disc_rewards)
                            horizon_rewards_list.append(temp_disc_rewards)
                    ## Horizon level.
                    if hori < (self.horizon - 1):
                        pred_all_next_obs = torch.vstack(horizon_obs_list)
                        pred_all_next_rewards = torch.vstack(
                            horizon_rewards_list)
                    #     # Statistics of target q
                    h_0 = torch.stack(horizon_q_list)
                    mean_0 = torch.mean(h_0, dim=0)
                    means.append(mean_0)
                    var_0 = torch.var(h_0, dim=0)
                    var_0[torch.abs(var_0) < 0.001] = 0.001
                    var_0 = 1.0 / var_0
                    vars.append(var_0)
                all_means = torch.stack(means)
                all_vars = torch.stack(vars)
                total_vars = torch.sum(all_vars, dim=0)
                for n in range(self.horizon):
                    all_vars[n] /= total_vars
                target_q = torch.sum(all_vars * all_means, dim=0)
                # target_q = torch.mean(all_means, dim=0)
            else:
                next_actions, next_log_pi, _ = self.actor_net.sample(next_obs)
                q_1, q_2 = self.target_critic_net.sample(next_obs,
                                                         next_actions)
                t_q = torch.minimum(q_1, q_2) - self.alpha * next_log_pi
                target_q = rewards + self.gamma * not_dones * t_q

        target_q = target_q.detach()
        assert (len(target_q.shape) == 2) and (target_q.shape[1] == 1)

        current_q1, current_q2 = self.critic_net.forward(obs, actions)
        td_error1 = target_q - current_q1
        td_error2 = target_q - current_q2
        # loss_1, loss_2 = self.critic_net.loss(obs, actions, target_q)
        critic1_loss = 0.5 * (td_error1.pow(2)).mean()
        critic2_loss = 0.5 * (td_error2.pow(2)).mean()
        critic_loss = critic1_loss + critic2_loss
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs):
        """

        Update the actor with respect to critics.
        :param obs:
        """
        # MFRL
        action, first_log_pi, _ = self.actor_net.sample(obs)
        actor_q1, actor_q2 = self.critic_net.sample(obs, action)
        actor_q = torch.min(actor_q1, actor_q2)
        # Q - alpha * log = V
        actor_loss = -(actor_q - self.alpha.detach() * first_log_pi).mean()
        # optimize the actor.
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # optimize the temperature alpha.
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha * (
                first_log_pi + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def train_policy(self, transitions):
        """

        :param transitions:
        """
        # Train with normal samples.
        states, actions, rewards, next_states, not_dones, _, _ = transitions

        assert len(states.shape) >= 2
        assert len(actions.shape) == 2
        assert len(rewards.shape) == 2 and rewards.shape[1] == 1
        assert len(next_states.shape) >= 2
        assert len(not_dones.shape) == 2 and not_dones.shape[1] == 1

        # Update current Q network
        self.update_critic(states, actions, rewards, next_states, not_dones)
        # Update Actor
        self.update_actor_and_alpha(states)
        # Update target Q network
        soft_update(self.critic_net, self.target_critic_net, self.tau)

    def train_world_model(self, statistics, transitions):
        """

        :param statistics:
        :param transitions:
        """
        self.world_model.set_statistics(statistics)
        states, actions, rewards, next_states, _, next_actions, next_rewards = transitions
        # mask the nones and zeros out.
        ok_masks = []
        for i in range(len(states)):
            if torch.sum(next_actions[i]) == 0 or next_rewards[i] == np.inf:
                ok_masks.append(False)
            else:
                ok_masks.append(True)
        states = states[ok_masks]
        actions = actions[ok_masks]
        rewards = rewards[ok_masks]
        next_states = next_states[ok_masks]
        next_actions = next_actions[ok_masks]
        next_rewards = next_rewards[ok_masks]
        self.world_model.train_world(states, actions, rewards, next_states,
                                     next_actions, next_rewards)

        transi = torch.cat((states, actions, next_states), dim=1)
        self.train_classifier(transi)

    def dyna_generate_and_train(self, transitions, env):
        """
        Only off-policy Dyna will work.
        :param transitions:
        """
        states, actions, rewards, next_states, not_dones, _, _ = transitions
        self.batch_size = states.shape[0]
        pred_states = [states]
        pred_actions = [actions]
        pred_rewards = [rewards]
        pred_next_states = [next_states]

        pred_state = next_states
        for _ in range(self.horizon):
            ###    Rewards   ###
            # random_actions = []
            # for _ in range(pred_state.shape[0]):
            #     pred_act = self.env.action_space.sample()
            #     random_actions.append(pred_act)
            # pred_act = torch.FloatTensor(np.array(random_actions)).to(self.device)
            pred_act, _, _ = self.actor_net.forward(pred_state)
            ###    Predictions   ###
            pred_next_state, _, _, _ = self.world_model.pred_next_states(
                pred_state, pred_act)
            pred_reward, _ = self.world_model.pred_rewards(pred_state,
                                                           pred_act)
            # Uncertainty measures.
            fake_transi = torch.cat((pred_state.detach(),
                                     actions.detach(),
                                     pred_next_state.detach()), dim=1)
            scores = self.discriminator(fake_transi)
            scores[scores == 1.0] = 0.99
            pred_reward[pred_reward <= 0.0] = 0.01
            pred_reward = torch.log(pred_reward.detach()) + torch.log(
                scores / (1 - scores))

            ###    Append    ###
            pred_states.append(pred_state)
            pred_actions.append(pred_act)
            pred_rewards.append(pred_reward.detach())
            pred_next_states.append(pred_next_state.detach())
            ###    Move on to the next    ###
            # pred_state = pred_next_state.detach()
        pred_states = torch.vstack(pred_states)
        pred_actions = torch.vstack(pred_actions)
        pred_rewards = torch.vstack(pred_rewards)
        pred_next_states = torch.vstack(pred_next_states)

        pred_not_dones = torch.FloatTensor(np.ones(pred_rewards.shape)).to(
            self.device)
        pred_not_dones[:self.batch_size] = not_dones

        # states, actions, rewards, next_states, not_dones
        self.train_policy((pred_states, pred_actions, pred_rewards,
                           pred_next_states, pred_not_dones, None, None))

    def train_classifier(self, real_transition):
        """

        :param real_transition:
        """
        real_size = real_transition.size(0)
        valid = Variable(
            torch.FloatTensor(real_transition.size(0), 1).fill_(1.0),
            requires_grad=False)
        fake = Variable(
            torch.FloatTensor(real_transition.size(0), 1).fill_(0.0),
            requires_grad=False)

        # Train Generator
        self.optimizer_G.zero_grad()
        # Generate a batch of images
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (real_size, 1))))
        gen_imgs = self.generator(z).detach()
        loss_g = F.binary_cross_entropy(self.discriminator(gen_imgs), valid)
        loss_g.backward()
        self.optimizer_G.step()

        # Train Discriminator
        self.optimizer_D.zero_grad()
        real_loss = F.binary_cross_entropy(self.discriminator(real_transition),
                                           valid)
        fake_loss = F.binary_cross_entropy(
            self.discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()
