import copy
import numpy as np
import torch
import torch.nn.functional as F
from utils import soft_update

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MBRL_SAC:
    """
    A MBRL class that implemented all MBRL algorithms for SAC.
    """
    def __init__(self, actor_network, critic_network, world_model, gamma, tau,
                 state_dim, action_dim, actor_lr, critic_lr, alpha_lr, horizon,
                 use_dyna, use_critic_steve, use_critic_mve, use_actor_mve,
                 use_actor_pg, use_bound, device):

        super().__init__()
        self.type = "mbrl"
        # Switches
        self.horizon = horizon
        self.use_bounded_active = use_bound
        self.use_critic_steve = use_critic_steve
        self.use_critic_mve = use_critic_mve
        self.use_actor_mve = use_actor_mve
        self.use_actor_pg = use_actor_pg
        self.dyna_use_uncertainty = use_dyna

        # Other Variables
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Actor Critic.
        self.actor = actor_network.to(device)
        self.critic = critic_network.to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.log_alpha = torch.tensor(np.log(1.0)).float().to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim
        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
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
    def act(self, obs, sample=False):
        """
        Make decisions with the trained policy.
        :param obs:
        :param sample:
        :return:
        """
        assert len(obs.shape) == 1
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(dim=0)
        # Evaluation
        if not sample:
            use_action, _, _, _ = self.actor.forward(obs)
        # Exploration
        else:
            with torch.no_grad():
                if not self.use_bounded_active:
                    _, use_action, _, _ = self.actor.forward(obs)
                else:
                    _, _, _, ten_times = self.actor.forward(obs)
                    obs2 = torch.repeat_interleave(obs, 5, dim=0)
                    # Solution: s1: Epistemic / s2: Aleatoric
                    _, _, mean, var = self.world_model.pred_next_states(
                        obs2, ten_times)
                    uncertainty2 = vi(mean, var)
                    # uncertainty2=(uncertainty2-torch.min(uncertainty2))+0.001
                    prob2 = F.softmax(torch.squeeze(uncertainty2), dim=0)
                    new_dist = torch.distributions.Categorical(prob2)
                    # Sample less  of world model, sample more infavor of sac
                    candi = new_dist.sample([1]).squeeze()
                    uncert_actions = ten_times[candi]
                    use_action = uncert_actions.unsqueeze(0)
                    # Who should be closer? Random Action? Mu? Mean?
                    # dist_2_mean = torch.pow((uncert_actions - dist_mean), 2)
                    # dist_2_mean = torch.sum(dist_2_mean, dim=1).squeeze()
                    # use_action=
                    # uncert_actions[torch.argmin(dist_2_mean)].unsqueeze(0)
        action_range = [float(self.env.action_space.low.min()),
                        float(self.env.action_space.high.max())]
        use_action = use_action.clamp(*action_range)
        assert use_action.ndim == 2 and use_action.shape[0] == 1
        use_action = use_action.detach().cpu().numpy()[0]
        return use_action

    def update_critic(self, obs, actions, rewards, next_obs, not_dones):
        """
        Update the critic first, the critic have to learn the correct action.
        :param obs:
        :param actions:
        :param rewards:
        :param next_obs:
        :param not_dones:
        """
        assert len(obs.shape) >= 2
        assert len(actions.shape) == 2
        assert len(rewards.shape) == 2 and rewards.shape[1] == 1
        assert len(next_obs.shape) >= 2
        assert len(not_dones.shape) == 2 and not_dones.shape[1] == 1

        with torch.no_grad():
            if self.use_critic_steve:
                # Horizon = 0
                # For next episodes used
                pred_all_next_obs = next_obs.unsqueeze(dim=0)
                pred_all_next_rewards = rewards.unsqueeze(dim=0)
                means = []
                vars = []
                for hori in range(self.horizon):
                    pred_all_next_rewards_list = []
                    pred_all_next_next_obs = []
                    est_target_q = []
                    # For each state batch [256, 17], reward extend 5 times, next extend 5 time.
                    for stat in range(pred_all_next_obs.shape[0]):
                        _, pred_target_us, pred_log_pi, _ = self.actor.forward(
                            pred_all_next_obs[stat])
                        pred_target_q1, pred_target_q2 = self.critic_target.forward(
                            pred_all_next_obs[stat], pred_target_us)
                        pred_target_q = torch.min(pred_target_q1,
                                                  pred_target_q2) - self.alpha.detach() * pred_log_pi
                        _, pred_rewards = self.world_model.pred_rewards(
                            obs=pred_all_next_obs[stat],
                            actions=pred_target_us)

                        temp_disc_rewards = []
                        for rwd in range(pred_rewards.shape[0]):
                            disc_pred_reward = (self.discount ** (hori + 1)) * \
                                               pred_rewards[rwd]
                            if hori > 0:
                                a = pred_all_next_rewards[
                                        stat] + not_dones * disc_pred_reward
                            else:
                                a = not_dones * disc_pred_reward
                            temp_disc_rewards.append(a)
                            assert rewards.shape == not_dones.shape == a.shape == pred_target_q.shape
                            pred_q = rewards + a + not_dones * (
                                    self.discount ** (
                                    hori + 2)) * pred_target_q
                            est_target_q.append(pred_q)

                    if hori < self.horizon - 1:
                        _, pred_all_next_ob, _, _ = self.world_model.pred_next_states(
                            pred_all_next_obs[stat],
                            pred_target_us)
                        temp_disc_rewards = torch.stack(temp_disc_rewards)
                        pred_all_next_rewards_list.append(temp_disc_rewards)
                        pred_all_next_next_obs.append(pred_all_next_ob)
                        # Predict the future.
                        pred_all_next_obs = torch.vstack(
                            pred_all_next_next_obs)
                        pred_all_next_rewards = torch.vstack(
                            pred_all_next_rewards_list)

                    #     # Statistics of target q
                    h_0 = torch.stack(est_target_q)
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

            if self.use_critic_mve:
                pred_rewards = torch.zeros(rewards.shape).to(self.device)
                pred_next_obs = next_obs
                for i in range(self.horizon):
                    _, pred_act, pred_log, _ = self.actor.forward(
                        pred_next_obs)
                    pred_reward, _ = self.world_model.pred_rewards(
                        obs=pred_next_obs, actions=pred_act)
                    pred_rewards += (self.discount ** (i + 1)) * pred_reward
                    pred_next_obs, _, _, _ = self.world_model.pred_next_states(
                        pred_next_obs, pred_act)
                # In the end
                target_q1, target_q2 = self.critic_target.forward(
                    pred_next_obs, pred_act)
                target_q = torch.min(target_q1, target_q2) \
                           - self.alpha.detach() * pred_log
                target_q = rewards + not_dones * pred_rewards + \
                           not_dones * (self.discount ** (i + 2)) * target_q

            if (not self.use_critic_mve) and (not self.use_critic_steve):
                _, target_us, log_pi, _ = self.actor.forward(next_obs)
                target_q1, target_q2 = self.critic_target.forward(next_obs,
                                                                  target_us)
                target_q = torch.min(target_q1,
                                     target_q2) - self.alpha.detach() * log_pi
                target_q = rewards + not_dones * self.discount * target_q

        target_q = target_q.detach()
        assert (len(target_q.shape) == 2) and (target_q.shape[1] == 1)
        current_q1, current_q2 = self.critic.forward(obs, actions)
        td_error1 = target_q - current_q1
        td_error2 = target_q - current_q2
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
        if (not self.use_actor_mve) and (not self.use_actor_pg):
            _, action, first_log_pi, _ = self.actor.forward(obs)
            actor_q1, actor_q2 = self.critic.forward(obs, action)
            actor_q = torch.min(actor_q1, actor_q2)
            # Q - alpha * log = V
            actor_loss = -(actor_q - self.alpha.detach() * first_log_pi).mean()
        # MBRL-MVE
        else:
            ########################    Expansion    #########################
            if self.use_actor_mve:
                # MVE for actor
                sum_dist_rewards = torch.zeros((self.batch_size, 1)).to(self.device)
                for hori in range(self.horizon):
                    _, act, log_pi, _ = self.actor.forward(obs)
                    pred_next, _, _, _ = self.world_model.pred_next_states(obs,
                                                                           act)
                    pred_reward, _ = self.world_model.pred_rewards(obs, act)
                    sum_dist_rewards += (self.discount ** (hori + 1)) * pred_reward
                    sum_dist_rewards -= self.alpha * log_pi
                    obs = pred_next
                _, act, first_log_pi, _ = self.actor.forward(obs)
                q_1, q_2 = self.critic.forward(obs, act)
                temp_loss = torch.min(q_1, q_2) + sum_dist_rewards
                actor_loss = -(temp_loss - self.alpha.detach() * first_log_pi).mean()
            elif self.use_actor_pg:
                # Policy Gradient for Actor.
                pred_xs = [obs]
                u_s = []
                log_p_us = []
                first_log_pi = 0
                for h in range(self.horizon):
                    # Roll out
                    _, sample_ut, log_p_ut, _ = self.actor.forward(obs)
                    if h == 0:
                        first_log_pi = log_p_ut
                    pred_next, _, _, _ = self.world_model.pred_next_states(
                        obs, sample_ut)
                    # Add to list
                    u_s.append(sample_ut)
                    log_p_us.append(log_p_ut.squeeze())
                    obs = pred_next
                    pred_xs.append(obs)
                #########################   Last step    ##########################
                _, sample_ut, log_p_ut, _ = self.actor.forward(obs)
                u_s.append(sample_ut)
                log_p_us.append(log_p_ut.squeeze())
                ####################    Stacking all produced data    #############
                pred_xs = torch.stack(pred_xs)
                u_s = torch.stack(u_s)
                log_p_us = torch.stack(log_p_us)
                ####################    Computing the loss of the Actor    ########
                pred_v = 0
                for i in range(self.horizon):
                    q_1, q_2 = self.critic.forward(pred_xs[i, :], u_s[i, :])
                    # V = Q -alpha * log
                    v_min = torch.min(q_1, q_2).reshape(
                        self.batch_size) - self.alpha.detach() * log_p_us[i, :]
                    pred_v += v_min.sum()
                actor_loss = -1 * pred_v

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
        # Train with normal samples.
        states, actions, rewards, next_states, not_dones, _, _ = transitions
        # Update current Q network
        self.update_critic(states, actions, rewards, next_states, not_dones)
        # Update Actor
        self.update_actor_and_alpha(states)
        # Update target Q network
        soft_update(self.critic, self.critic_target, self.tau)

    def train_world_model(self, statistics, transitions):
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

    def dyna_generate_and_train(self, transitions, on_policy=False):
        states, actions, rewards, next_states, not_dones, _, _ = transitions
        pred_states = [states]
        pred_actions = [actions]
        pred_rewards = [rewards]
        pred_next_states = [next_states]
        pred_not_dones = [not_dones]
        pred_state = next_states
        for _ in range(self.horizon):
            ###    Rewards   ###
            pred_action = []
            for _ in range(states.shape[0]):
                if on_policy:
                    pred_act, _, _, _ = self.actor.forward(pred_state)
                else:
                    pred_act = self.env.action_space.sample()
                pred_action.append(pred_act)
            pred_action = torch.FloatTensor(np.array(pred_action)).to(self.device)
            ###    Predictions   ###
            pred_next_state, _, means, stds = self.world_model.pred_next_states(
                pred_state, pred_action)
            pred_reward, _ = self.world_model.pred_rewards(pred_state,
                                                           pred_action)
            if self.dyna_use_uncertainty:
                pred_not_done = vi(means, stds)
                # pred_not_done = self.uncertainty_measures(means, stds)
                pred_not_dones.append(pred_not_done)
            ###    Append    ###
            pred_states.append(pred_state)
            pred_actions.append(pred_action)
            pred_rewards.append(pred_reward.detach())
            pred_next_states.append(pred_next_state.detach())
            ###    Move on to the next    ###
            # pred_state = pred_next_state.detach()
        pred_states = torch.vstack(pred_states)
        pred_actions = torch.vstack(pred_actions)
        pred_rewards = torch.vstack(pred_rewards)
        pred_next_states = torch.vstack(pred_next_states)
        if self.dyna_use_uncertainty:
            pred_not_dones = torch.vstack(pred_not_dones)
        else:
            pred_not_dones = torch.FloatTensor(np.ones(pred_rewards.shape)).to(
                self.device)
        pred_not_dones[:self.batch_size] = not_dones
        # states, actions, rewards, next_states, not_dones
        self.train_policy((pred_states, pred_actions, pred_rewards,
                           pred_next_states, pred_not_dones, None, None))