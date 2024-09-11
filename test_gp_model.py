import numpy as np
import torch
import pyro
import pyro.contrib.gp as gp
assert pyro.__version__.startswith('1.9.1')
pyro.set_rng_seed(1)
from envs import DMCSEnvironment
from utils import set_seed

domain_names = ['cheetah', 'reacher', 'walker', 'humanoid', 'cartpole', 'hopper', 'fish', 'finger', 'acrobot',
                'ball_in_cup']
task_names = ['run', 'hard', 'walk', 'run', 'swingup', 'hop', 'swim', 'turn_hard', 'swingup', 'catch']

seed_list = [10, 25, 35]
train_collect_epis = [10, 20, 50, 100]
train_iters = [10, 50, 100, 200, 400]

# ENV 10 * SEED 5 * COLLECT 6 * ITER 7 = 50 * 42 * 10 = 33600
total_errors = np.zeros((len(domain_names), len(seed_list), len(train_collect_epis), len(train_iters), 10))
corr_results = np.zeros((len(domain_names), len(seed_list), len(train_collect_epis), len(train_iters), 10))


def test(env, gaussian_model, i_i, j_j, k_k, l_l):
    print("Test---------------------")
    for m_m in range(10):
        tstate = env.reset()
        mse_errors = 0.0
        errors = []
        covs = []
        for _ in range(100):
            t_action = env.sample_action()
            tn_state, _, _, _ = env.step(t_action)
            ttensor_state = torch.FloatTensor(tstate).unsqueeze(dim=0)
            ttensor_action = torch.FloatTensor(t_action).unsqueeze(dim=0)
            ttensor_input = torch.cat((ttensor_state, ttensor_action), dim=1)
            tmean, tcov = gaussian_model(ttensor_input, full_cov=True)
            covs.append(torch.sum(torch.squeeze(tcov)).detach().cpu().numpy())
            tmean = tmean.detach().squeeze().cpu().numpy()
            mse_loss = np.mean((tn_state - tmean) ** 2)
            errors.append(mse_loss)
            mse_errors += mse_loss
            tstate = tn_state
        errors = np.array(errors)
        covs = np.array(covs)
        corr = np.corrcoef(errors, covs)
        total_errors[i_i, j_j, k_k, l_l, m_m] = mse_errors
        corr_results[i_i, j_j, k_k, l_l, m_m] = corr[0, 1]


for i in range(len(domain_names)):
    for j in range(len(seed_list)):
        set_seed(seed_list[j])
        env = DMCSEnvironment(domain_names[i], task_names[i])
        env.set_seed(seed_list[j])
        state_dim = env.observation_space
        action_dim = env.action_num
        kernel = gp.kernels.RBF(input_dim=state_dim + action_dim)
        for k in range(len(train_collect_epis)):
            for l in range(len(train_iters)):
                states = []
                actions = []
                next_states = []
                for _ in range(train_collect_epis[k]):
                    state = env.reset()
                    for _ in range(100):
                        action = env.sample_action()
                        n_state, _, _, _ = env.step(action)
                        states.append(state)
                        next_states.append(n_state)
                        actions.append(action)
                        state = n_state
                states = np.stack(states)
                actions = np.stack(actions)
                next_states = np.stack(next_states)
                tensor_states = torch.FloatTensor(states)
                tensor_actions = torch.FloatTensor(actions)
                tensor_n_states = torch.FloatTensor(next_states)
                tensor_x = torch.cat((tensor_states, tensor_actions), dim=1)
                tensor_y = tensor_n_states.T
                gpr = gp.models.GPRegression(tensor_x, tensor_y, kernel)
                optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
                # losses = gp.util.train(gpr, num_steps=10)
                loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
                for _ in range(train_iters[l]):
                    optimizer.zero_grad()
                    loss = loss_fn(gpr.model, gpr.guide)
                    loss.backward()
                    optimizer.step()
                test(env, gpr, i, j, k, l)

np.save("statistics/gp_errors.npy", total_errors)
np.save("statistics/gp_corrs.npy", corr_results)


# def f(x):
#     return (6 * x - 2)**2 * torch.sin(12 * x - 4)
#
# def update_posterior(x_new):
#     y = f(x_new) # evaluate f at new point.
#     X = torch.cat([gpmodel.X, x_new]) # incorporate new evaluation
#     y = torch.cat([gpmodel.y, y])
#     gpmodel.set_data(X, y)
#     # optimize the GP hyperparameters using Adam with lr=0.001
#     optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
#     gp.util.train(gpmodel, optimizer)
#
# def lower_confidence_bound(x, kappa=2):
#     mu, variance = gpmodel(x, full_cov=False, noiseless=False)
#     sigma = variance.sqrt()
#     return mu - kappa * sigma
#
#
# def find_a_candidate(x_init, lower_bound=0, upper_bound=1):
#     # transform x to an unconstrained domain
#     constraint = constraints.interval(lower_bound, upper_bound)
#
#     unconstrained_x_init = transform_to(constraint).inv(x_init)
#     unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
#
#     minimizer = optim.LBFGS([unconstrained_x], line_search_fn='strong_wolfe')
#
#     def closure():
#         minimizer.zero_grad()
#         x = transform_to(constraint)(unconstrained_x)
#         y = lower_confidence_bound(x)
#         autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
#         return y
#     minimizer.step(closure)
#     # after finding a candidate in the unconstrained domain,
#     # convert it back to original domain.
#     x = transform_to(constraint)(unconstrained_x)
#     return x.detach()
#
# def next_x(lower_bound=0, upper_bound=1, num_candidates=5):
#     candidates = []
#     values = []
#
#     x_init = gpmodel.X[-1:]
#     for i in range(num_candidates):
#         x = find_a_candidate(x_init, lower_bound, upper_bound)
#         y = lower_confidence_bound(x)
#         candidates.append(x)
#         values.append(y)
#         x_init = x.new_empty(1).uniform_(lower_bound, upper_bound)
#
#     argmin = torch.min(torch.cat(values), dim=0)[1].item()
#     return candidates[argmin]
#
# def plot(gs, xmin, xlabel=None, with_title=True):
#     xlabel = "xmin" if xlabel is None else "x{}".format(xlabel)
#     Xnew = torch.linspace(-0.1, 1.1, 100)
#     ax1 = plt.subplot(gs[0])
#     ax1.plot(gpmodel.X.numpy(), gpmodel.y.numpy(), "kx")  # plot all observed data
#     with torch.no_grad():
#         loc, var = gpmodel(Xnew, full_cov=False, noiseless=False)
#         sd = var.sqrt()
#         ax1.plot(Xnew.numpy(), loc.numpy(), "r", lw=2)  # plot predictive mean
#         ax1.fill_between(Xnew.numpy(), loc.numpy() - 2*sd.numpy(), loc.numpy() + 2*sd.numpy(),
#                          color="C0", alpha=0.3)  # plot uncertainty intervals
#     ax1.set_xlim(-0.1, 1.1)
#     ax1.set_title("Find {}".format(xlabel))
#     if with_title:
#         ax1.set_ylabel("Gaussian Process Regression")
#
#     ax2 = plt.subplot(gs[1])
#     with torch.no_grad():
#         # plot the acquisition function
#         ax2.plot(Xnew.numpy(), lower_confidence_bound(Xnew).numpy())
#         # plot the new candidate point
#         ax2.plot(xmin.numpy(), lower_confidence_bound(xmin).numpy(), "^", markersize=10,
#                  label="{} = {:.5f}".format(xlabel, xmin.item()))
#     ax2.set_xlim(-0.1, 1.1)
#     if with_title:
#         ax2.set_ylabel("Acquisition Function")
#     ax2.legend(loc=1)
#
# # x = torch.linspace(0, 1, 100)
# # plt.figure(figsize=(8, 4))
# # plt.plot(x.numpy(), f(x).numpy())
# # plt.show()
#
# # initialize the model with four input points: 0.0, 0.33, 0.66, 1.0
# X = torch.tensor([0.0, 0.33, 0.66, 1.0])
# y = f(X)
# gpmodel = gp.models.GPRegression(X, y, gp.kernels.Matern52(input_dim=1),
#                                  noise=torch.tensor(0.1), jitter=1.0e-4)
#
#
# plt.figure(figsize=(12, 30))
# outer_gs = gridspec.GridSpec(5, 2)
# optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
# gp.util.train(gpmodel, optimizer)
# for i in range(8):
#     xmin = next_x()
#     gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[i])
#     plot(gs, xmin, xlabel=i+1, with_title=(i % 2 == 0))
#     update_posterior(xmin)
# plt.show()


# from scipy.spatial.transform import Rotation
# import numpy as np
# import math
#
# eval_array = [[], [], []]
#
# for i in range(1000):
#     print("----------------------")
#     a = (np.random.rand(3) - 0.5) * 2 * math.pi
#     b = (np.random.rand(3) - 0.5) * 2 * math.pi
#
#     # (x, y, z, w) -> (w, x, y, z) <a, b, c, d>
#     rot_a = Rotation.from_euler(seq='xyz', angles=a)
#     mat_a = rot_a.as_matrix()
#     quat_a = np.roll(rot_a.as_quat(canonical=True), shift=1)
#     # <e, f, g, h>
#     rot_b = Rotation.from_euler(seq='xyz', angles=b)
#     mat_b = rot_b.as_matrix()
#     quat_b = np.roll(rot_b.as_quat(canonical=True), shift=1)
#
#     # b = x * a
#     # b * at = x
#     x = np.matmul(mat_b, mat_a.transpose())
#     rot_delta = Rotation.from_matrix(x)
#     delta_euler = rot_delta.as_euler('xyz')
#     eval_array[0].append(np.linalg.norm(delta_euler))
#
#     a = quat_a[0]
#     b = quat_a[1]
#     c = quat_a[2]
#     d = quat_a[3]
#
#     e = quat_b[0]
#     f = quat_b[1]
#     g = quat_b[2]
#     h = quat_b[3]
#
#     q2m = np.matrix([[e, -1 * f, -1 * g, -1 * h],
#                      [f, e, -1 * h, g],
#                      [g, h, e, -1 * f],
#                      [h, -1 * g, f, e]])
#     q1i = np.matrix([[a], [-1 * b], [-1 * c], [-1 * d]])
#
#     quat_dis = np.matmul(q2m, q1i)
#     diff_rot = Rotation.from_quat(np.roll(np.squeeze(np.asarray(quat_dis)), shift=3))
#     diff_euler = diff_rot.as_euler('xyz')
#     eval_array[1].append(np.linalg.norm(diff_euler))
#
#     # diff_euler = abs(diff_euler)
#     # diff_euler_2 = math.pi * 2 - diff_euler
#     # min_diff_euler_2 = np.minimum(diff_euler, diff_euler_2)
#     # print(min_diff_euler_2)
#     # quaternion_dist = math.sqrt(diff_euler[0] ** 2 + diff_euler[1] ** 2 + diff_euler[2] ** 2)
#
#     p = np.array([quat_a[0], quat_a[1], quat_a[2], quat_a[3]])
#     c_q = np.array([quat_b[0], -1 * quat_b[1], -1 * quat_b[2], -1 * quat_b[3]])
#     z_0 = abs(p[0] * c_q[0] - p[1] * c_q[1] - p[2] * c_q[2] - p[3] * c_q[3])
#     quaternion_dist = (2 * math.acos(z_0))
#     print(quaternion_dist)
#     eval_array[2].append(quaternion_dist)
#     # print(quaternion_dist)
#
# eval_array = np.array(eval_array)
# np.savetxt("eval_rewards.csv", eval_array, delimiter=",")
#
#
# from envs import DMCSEnvironment
# from networks.mbrl import Ensemble_World_Reward_GAN
# import numpy as np
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
#
#
#
#
#
# env = DMCSEnvironment("cheetah", "run")
#
# world_model = Ensemble_World_Reward_GAN(state_dim=env.observation_space,
#                                         action_dim=env.action_num,
#                                         num_models=5)
#
# epochs = 500
# batch_size = 128
#
#
# states = []
# actions = []
# next_states = []
# rewards = []
# mse_loss = []
#
# for i in tqdm(range(epochs)):
#     state = env.reset()
#     for j in range(1000):
#         action = np.random.uniform(env.min_action_value, env.max_action_value,
#                                    (env.action_num,))
#         next_state, reward, done, info = env.step(action)
#         print(reward)
#         states.append(state)
#         actions.append(action)
#         next_states.append(next_state)
#         rewards.append(reward)
#
#         if len(states) == batch_size:
#             # Set statistics.
#             states_tensor = torch.FloatTensor(np.array(states[:len(states)-1]))
#             actions_tensor = torch.FloatTensor(np.array(actions[:len(states)-1]))
#             rewards_tensor = torch.FloatTensor(np.array(rewards[:len(states)-1])).unsqueeze(dim=1)
#             next_states_tensor = torch.FloatTensor(np.array(next_states[:len(states)-1]))
#             next_actions_tensor = torch.FloatTensor(np.array(actions[1:len(states)]))
#             next_rewards_tensor = torch.FloatTensor(np.array(rewards[1:len(states)])).unsqueeze(dim=1)
#
#             # Set statistics before all happens.
#             deltas_torch = next_states_tensor - states_tensor
#             statistics = {
#                 'ob_mean': states_tensor.mean(dim=0) + 0.000001,
#                 'ob_std': states_tensor.std(dim=0) + 0.000001,
#                 'delta_mean': deltas_torch.mean(dim=0) + 0.000001,
#                 'delta_std': deltas_torch.std(dim=0) + 0.000001
#             }
#             world_model.set_statistics(statistics)
#
#             # Evaluate before
#             pred, _, _, _ = world_model.pred_next_states(states_tensor,
#                                                          actions_tensor)
#             mse_loss.append(torch.mean(F.mse_loss(pred, next_states_tensor)).item())
#
#             # Then training.
#             world_model.train_world(states_tensor, actions_tensor,
#                                     rewards_tensor, next_states_tensor,
#                                     next_actions_tensor, next_rewards_tensor)
#
#             states =[]
#             actions = []
#             next_states = []
#             rewards = []
#
# # eval_array = np.array(mse_loss)
# # np.savetxt("GAN_MSE.csv", eval_array, delimiter=",")
#
#
# import math
#
# import numpy as np
# from datetime import datetime
#
# import torch
# from tqdm import trange
# from tqdm.contrib.logging import logging_redirect_tqdm
# import torch.nn.functional as F
#
# class Trainer:
#     """
#     A class that responsible for training and evaluating.
#
#     """
#
#     def __init__(self, env, agent, memory, device, name, logger):
#
#         # Should be Goal Conditioned.
#         self.logger = logger
#         self.name = name
#         self.device = device
#
#         self.batch_size = 256
#         self.max_epi_steps = 1000
#         self.num_eval = 10
#
#         self.current_step = 0
#         self.date_and_time = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
#
#         self.evaluation_array = [[], [], [], [], []]
#
#         self.env = env
#         self.memory = memory
#         self.agent = agent
#
#     def evaluate(self):
#         """
#
#         :param observe:
#         """
#         total_rewards = 0
#         reward_errors = 0
#         dynamic_errors = 0
#         counter = 1
#
#         for _ in range(self.num_eval):
#             state = self.env.reset()
#             for _ in range(self.max_epi_steps):
#                 action = self.agent.select_action_from_policy(state, evaluation=True)
#                 next_state, reward, done, _ = self.env.step(action)
#
#                 tensor_state = torch.FloatTensor(state).to(self.device).unsqueeze(dim=0)
#                 tensor_action = torch.FloatTensor(action).to(self.device).unsqueeze(dim=0)
#                 tensor_next_state = torch.FloatTensor(next_state).to(self.device).unsqueeze(dim=0)
#
#                 pred, _, _, _ = self.agent.world_model.pred_next_states(tensor_state, tensor_action)
#                 model_error = F.mse_loss(pred, tensor_next_state)
#                 model_error = model_error.item()
#
#                 pred_rwd, _, _ = self.agent.world_model.pred_rewards(tensor_state, tensor_action, tensor_next_state)
#                 pred_rwd = pred_rwd.detach().cpu().item()
#                 reward_error = math.sqrt((pred_rwd - reward) ** 2)
#
#                 reward_errors += reward_error
#                 dynamic_errors += model_error
#                 total_rewards += reward
#                 counter += 1
#
#                 state = next_state
#                 if done:
#                     break
#
#         avg_rewards = total_rewards / self.num_eval
#         reward_errors /= counter
#         dynamic_errors /= counter
#
#         self.evaluation_array[0].append(avg_rewards)
#         self.evaluation_array[1].append(dynamic_errors)
#         self.evaluation_array[2].append(reward_errors)
#         self.evaluation_array[3].append(self.current_step)
#         self.evaluation_array[4].append(0.0)
#         eval_array = np.array(self.evaluation_array)
#
#         self.logger.info(f'Evaluation: {total_rewards / self.num_eval}')
#
#         if len(self.name) > 5:
#             # Save the metrics
#             file_name = (self.name + "_" + self.date_and_time)
#             np.savetxt(file_name + "_eval_rewards.csv",
#                        eval_array, delimiter=",")
#
#     def train_agent(self):
#         """
#         Train the agent
#         :param max_epi_steps: Maximum number of steps for each episode
#         """
#         state = self.env.reset()
#         if len(self.memory) > self.batch_size:
#             statistics = self.memory.get_statistics()
#             self.agent.world_model.set_statistics(statistics)
#         for _ in range(self.max_epi_steps):
#             # Execute action and add to memory.
#             if len(self.memory) < self.batch_size + 1:
#                 action = self.env.sample_action()
#             else:
#                 action = self.agent.select_action_from_policy(state=state, evaluation=False)
#             next_state, reward, done, _ = self.env.step(action)
#             self.memory.add(state, action, reward, next_state, done)
#             # Training the world model and the agent
#             if len(self.memory) > self.batch_size:
#                 if self.agent.type == "mbrl":
#                     if len(self.memory) == (self.batch_size + 1):
#                         # First time set statics
#                         statistics = self.memory.get_statistics()
#                         self.agent.world_model.set_statistics(statistics)
#                     # Train world model many times.
#                     if self.current_step % 5 == 0:
#                         self.agent.train_world_model(self.memory, self.batch_size)
#                 self.agent.train_policy(self.memory, self.batch_size)
#             # Do evaluation for every 200
#             self.current_step += 1
#             if done:
#                 break
#             # Move to the next state
#             state = next_state
#
#     def train_loop(self):
#         """
#         The main loop. Call Tranin or evaluation.
#
#         """
#         with logging_redirect_tqdm():
#             for _ in trange(1000):
#                 self.train_agent()
#                 self.evaluate()
#
#     # def observe_critic_actor(self, state):
#     #     """
#     #     For Q evaluation policy vs Q
#     #     """
#     #     num_sample = 5
#     #     num_act_dim = 6
#     #     total = 5 * 5 * 5 * 5 * 5 * 5
#     #     as0 = np.zeros((total, num_act_dim))
#     #     as1 = np.zeros((total, num_act_dim))
#     #
#     #     action = np.zeros((num_act_dim,))
#     #     acts = [-0.8, -0.4, 0.0, 0.4, 0.8]
#     #
#     #     counter = 0
#     #     for l in range(num_sample):
#     #         action[5] = acts[l]
#     #         for k in range(num_sample):
#     #             action[4] = acts[k]
#     #             for j in range(num_sample):
#     #                 action[3] = acts[j]
#     #                 for i in range(num_sample):
#     #                     action[2] = acts[i]
#     #                     for h in range(num_sample):
#     #                         action[1] = acts[h]
#     #                         for g in range(num_sample):
#     #                             action[0] = acts[g]
#     #                             as0[counter] = action
#     #                             counter += 1
#     #
#     #     counter = 0
#     #     for l in range(num_sample):
#     #         action[5] = acts[l]
#     #         for k in range(num_sample):
#     #             action[4] = acts[k]
#     #             for j in range(num_sample):
#     #                 action[3] = acts[j]
#     #                 for i in range(num_sample):
#     #                     action[2] = acts[i]
#     #                     for h in range(num_sample):
#     #                         action[1] = acts[h]
#     #                         for g in range(num_sample):
#     #                             action[0] = acts[g]
#     #                             as0[counter] = action
#     #                             counter += 1
#     #
#     #     self.as0 = torch.FloatTensor(as0).to(self.device)
#     #     self.as1 = torch.FloatTensor(as1).to(self.device)
#     #
#     #     # Create action samples
#     #     state_tensor = torch.FloatTensor(state).to(device=self.device)
#     #     state_tensor = state_tensor.unsqueeze(dim=0)
#     #     multi_state_tensor = torch.repeat_interleave(state_tensor, 5 ** 6,
#     #                                                  dim=0)
#     #
#     #     # Same states, same action distributions.
#     #     _, _, _, dist = self.agent.actor_net(state_tensor)
#     #     # Same states, different actions.
#     #     q_0, _ = self.agent.critic_net(multi_state_tensor, self.as0)
#     #     # q_2, _ = self.agent.critic_net(multi_state_tensor, self.as2)
#     #     # q_3, _ = self.agent.critic_net(multi_state_tensor, self.as3)
#     #
#     #     # Dim 0
#     #     total_kld_0 = 0.0
#     #     for i in range(3125):
#     #         # For first dimension
#     #         q_s_0 = q_0[i * 5:i * 5 + 5]
#     #         # qs to distribution.
#     #         q_s_0 = F.softmax(q_s_0, dim=0)
#     #         a_s_0 = (dist.log_prob(self.as0[i * 5:i * 5 + 5]))
#     #         a_s_0 = torch.exp(a_s_0)
#     #         a_s_0 = F.softmax(a_s_0[:, 0], dim=0)
#     #         kld0 = F.kl_div(q_s_0, a_s_0)
#     #         total_kld_0 += kld0
#     #     self.logger.info(f"{total_kld_0.item()}")
#     #     return total_kld_0.item()
#     # state_tensor = torch.FloatTensor(next_state).to(
#     #     device=self.device)
#     # state_tensor = state_tensor.unsqueeze(dim=0)
#     # actions, _, _,_ = self.agent.actor_net.sample(state_tensor)
#
#     # self.observe_critic_actor()
#
#     # q1s, _ = self.agent.critic_net.sample(state_tensor, actions)
#     # # Normalize
#     # temp_min = torch.min(q1s)
#     # temp_max = torch.max(q1s)
#     # temp_scale = temp_max - temp_min
#     # norm_q1s = (q1s - temp_min) / temp_scale
#
#     # Reward Prediction.
#     # pred_mean, _ = self.agent.world_model.pred_rewards(
#     #     obs=state_tensor, actions=actions)
#     # pred_mean = pred_mean.item()
#     # reward_error += abs(pred_mean - reward)
#
#     # World model prediction
#     # pred_next_state, _, _, _ = self.agent.world_model.pred_next_states(
#     #     obs=state_tensor, actions=actions)
#     # pred_next_state = pred_next_state.detach().cpu().numpy().squeeze()
#     # dynamic_error += (np.mean((pred_next_state - next_state) ** 2))
#
#     # # uncert 1
#     # total_uncert1 += vi(pred_mean, pred_var)
#     # # uncert 2
#     # total_uncert2 += sampling(pred_mean, pred_var)
#     # # uncert 3
#     # total_uncert3 += mean_std(pred_mean, pred_var)
#
#
#
# # if __name__ == "__main__":
# #     main()
# #
# # import torch
# # import numpy as np
# # # import torch.nn.functional as F
# # # import math
# # # from scipy.spatial.transform import Rotation
# # # from arm.utils import matrix_to_euler_angles, matrix_to_quaternion
# # # from envs.DMCS import DMCSEnvironment
# #
# # from envs.UR10_Kinematic_Env import UR10_Kinematic_Env
# # from train_loop import Trainer
#
# # if __name__ == "__main__":
# #     env = UR10_Kinematic_Env()
#     # trainer = Trainer(env, action_dim=6)
#     # trainer.train()
#
# # lists = ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx', 'XYZ', 'XZY', 'YXZ', 'YZX', 'ZYX', 'ZXY']
# # lists2 = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZYX', 'ZXY']
# # for conb1 in lists:
# #     counter = 0
# #     for i in range(1000):
# #         euler_angles = 2 * (np.random.rand(3) - 0.5) * math.pi
# #
# #         rot = Rotation.from_euler(conb1, euler_angles)
# #         matrix = np.expand_dims(rot.as_matrix(), axis=0)
# #         matrix = torch.FloatTensor(matrix)
# #
# #         tensor_quat = matrix_to_quaternion(matrix)
# #         tensor_quat = tensor_quat.squeeze().numpy()
# #
# #         distance = np.linalg.norm(np.roll(rot.as_quat(canonical=True), shift=1) - tensor_quat)
# #
# #         if distance > 0.01:
# #             print("---------------")
# #             print(np.roll(rot.as_quat(), shift=1))
# #             print(tensor_quat)
# #
# #         # Successful
# #         if distance < 0.01:
# #             counter += 1
# #
# #     if counter < 10:
# #         print("---------------------")
# #         print(conb1)
# #
# #     print(counter)
#
# # ############    Verification of the Query and Step    ####################
# # evaluation = [[], [], []]
# #
# # for i in range(10):
# #     state = env.reset()
# #     for j in range(100):
# #         # Parital
# #         action = env.sample_action(1, 6)
# #         action = action[0]
# #         # Full for step.
# #         next_state, reward, dones, _, info = env.step(action)
# #         evaluation[0].append(info["quat_dist"])
# #         evaluation[1].append(info["euler_dist"])
# #         evaluation[2].append(info["euler_dist"] - info["quat_dist"])
# #
# #         # Partial state. Parital Action
# #         state = np.expand_dims(state, axis=0)
# #         action = np.expand_dims(action, axis=0)
# #         state_tensor = torch.FloatTensor(state)
# #         action_tensor = torch.FloatTensor(action)
# #         pred_next, rd, dns = env.tensor_query(state_tensor, action_tensor)
# #         pred_next = pred_next.numpy()
# #         rd = rd.numpy()
# #         dns = dns.numpy()
# #
# #         if dns[0] != dones:
# #             print("lllllll")
# #         # if distance > 0.001:
# #         #     print(distance)
# #         state = next_state
# #
# # eval_array = np.array(evaluation)
# # # Save the metrics
# # file_name = "data/Kinematic"
# # np.savetxt(file_name + "_eval_rewards.csv", eval_array, delimiter=",")
#
# # from envs.mujoco_env import Arm_Mujoco
#
# # kine = Forward_Kinematics()
# # theta = np.array(        [169.92, 121.48, 299.97, 211.52, 144.73, 34.86, 176.07])
# # for_kine = (theta-degree_offset) * (math.pi / 180)
# # pos, orient = kine.forward_kinematics(for_kine)
#
# # env = Arm_Mujoco()
#
# # ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1)
# # while True:
# #     data = ser.readline()# .decode().strip()
# #     if data:
# #         print(data)
#
#
# # from Arm import Arm
# # from Arm_Env import Arm_Env
# # import dynamixel_sdk as dxl
#
# # env = Arm_Env()
# # env.sensor.get_position()
#
# # env.arm.disable_all()
#
# # for i in range(10):
# #     state = env.reset()
# #     for k in range(10):
# #         action = env.sample_action()
# #         env.step(action)
#
# # arm.disable_all()
# # arm._enable_all()
#
# # fake_joints = np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]])
# # fake_joints = torch.FloatTensor(fake_joints)
# # forward(fake_joints)
#
# # port_handler = dxl.PortHandler("/dev/ttyUSB0")
# # port_handler.openPort()
# # port_handler.setBaudRate(1000000)
# # packet_handler = dxl.PacketHandler(protocol_version=1.0)
#
# # servo 0 position range: 400  - 800, home: 520
# # servo 1 position range: 3535 - 164, home: 2024
# # servo 2 position range: 20   - 990, home: 520
# # servo 3 position range: 120  - 880, home: 510
# # servo 4 position range: 0    - 1000, home: 520
# # servo 5 position range: 160  - 860, home: 520
# # servo 6 position range: 170  - 840, home: 520
#
# # lists = ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx', 'XYZ', 'XZY', 'YXZ', 'YZX', 'ZYX', 'ZXY']
# # lists2 = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZYX', 'ZXY']
# #
# # for conb2 in lists2:
# #     for conb1 in lists:
# #         counter = 0
# #         for i in range(1000):
# #             euler_angles = 2 * (np.random.rand(3) - 0.5) * math.pi
# #             rot = Rotation.from_euler(conb1, euler_angles)
# #
# #             matrix = np.expand_dims(rot.as_matrix(), axis=0)
# #             matrix = torch.FloatTensor(matrix)
# #             tensor_euler = matrix_to_euler_angles(matrix, conb2)
# #             tensor_euler = tensor_euler.squeeze().numpy()
# #             distance = np.linalg.norm(rot.as_euler(conb1) - tensor_euler)
# #
# #             if distance > 0.03:
# #                 counter += 1
# #
# #         if counter < 10:
# #             print("---------------------")
# #             print(conb1)
# #             print(conb2)
# #             print(counter)
#
#
