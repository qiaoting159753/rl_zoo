import numpy as np
import math
from datetime import datetime
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import trange


class Trainer:
    """
    A class that responsible for training and evaluating.

    """

    def __init__(self, env, agent, memory, device, use_dyna, use_critic_steve,
                 use_critic_mve, use_actor_mve, use_actor_pg, use_bound):
        # Should be Goal Conditioned.
        self.device = device
        self.max_steps = 1000000
        self.max_epi_steps = 1000
        self.num_eval = 5
        self.est_epi = int(
            math.ceil(self.max_steps / self.max_epi_steps) * 1.5)

        self.current_step = 0
        self.train_world_times = 1
        self.train_agent_times = 1

        self.use_dyna = use_dyna
        self.use_critic_steve = use_critic_steve
        self.use_critic_mve = use_critic_mve
        self.use_actor_mve = use_actor_mve
        self.use_actor_pg = use_actor_pg
        self.use_bound = use_bound

        self.date_and_time = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
        self.batch_size = 128
        self.evaluation_array = [[], [], [], [], []]
        self.env_name = "HalfCheetah-v4"
        # self.env = PusherEnv()
        self.env = env
        self.memory = memory
        self.agent = agent

    def evaluate(self):
        """

        :param max_epi_steps: Each episode, the maximum number of steps.
        :param num_eval: Number of episodes to evaluate the agent.
        """
        total_rewards = 0
        reward_error = 0
        dynamic_error = 0
        counter = 1

        for _ in range(self.num_eval):
            state, _ = self.env.reset()
            for _ in range(self.max_epi_steps):
                action = self.agent.select_action_from_policy(state,
                                                              evaluation=True)
                next_state, reward, terminate, truncate, _ = self.env.step(
                    action)
                total_rewards += reward

                # state_tensor = torch.FloatTensor(next_state).to(device=self.device)
                # state_tensor = state_tensor.unsqueeze(dim=0)

                # num_act_per = 10
                #
                # actions, _, _ = self.agent.actor_net.sample(obs=state_tensor,
                #                                             sample_times=100)
                # # This line have to goes after sample actions.
                # state_tensor = torch.repeat_interleave(state_tensor, 100,
                #                                        dim=0)
                #
                #
                # q1s, _ = self.agent.critic_net.sample(state_tensor, actions)
                #
                # # Normalize
                # temp_min = torch.min(q1s)
                # temp_max = torch.max(q1s)
                # temp_scale = temp_max - temp_min
                # norm_q1s = (q1s - temp_min) / temp_scale

                # Reward Prediction.
                # state_tensor = torch.FloatTensor(state).to(device).unsqueeze(
                #     dim=0)
                # action_tensor = torch.FloatTensor(action).to(device).unsqueeze(
                #     dim=0)
                # pred_mean, _ = self.agent.world_model.pred_rewards(
                #     obs=state_tensor, actions=action_tensor)
                # pred_mean = pred_mean.item()
                # reward_error += abs(pred_mean - reward)
                # World model prediction
                # pred_next_state, _, _, _ = self.agent.world_model.pred_next_states(
                #     obs=state_tensor, actions=action_tensor)
                # pred_next_state = pred_next_state.detach().cpu().numpy().squeeze()
                # dynamic_error += (np.mean((pred_next_state - next_state) ** 2))
                # # uncert 1
                # total_uncert1 += vi(pred_mean, pred_var)
                # # uncert 2
                # total_uncert2 += sampling(pred_mean, pred_var)
                # # uncert 3
                # total_uncert3 += mean_std(pred_mean, pred_var)

                counter += 1
                state = next_state
                done = terminate or truncate
                if done:
                    break
        avg_rewards = total_rewards / self.num_eval
        reward_error /= counter
        dynamic_error /= counter

        self.evaluation_array[0].append(avg_rewards)
        self.evaluation_array[1].append(reward_error)
        self.evaluation_array[2].append(dynamic_error)
        self.evaluation_array[3].append(self.current_step)
        self.evaluation_array[4].append(0)
        eval_array = np.array(self.evaluation_array)

        # Save the metrics
        param_list = "_" + str(self.use_bound) + "_" + str(self.use_dyna) + \
                     "_" + str(self.use_actor_mve) + "_" + \
                     str(self.use_actor_pg) + "_" + str(self.use_critic_mve) + \
                     "_" + str(self.use_critic_steve) + "_"

        file_name = self.env_name + param_list + self.date_and_time

        np.savetxt(file_name + "_eval_rewards.csv",
                   eval_array, delimiter=",")
        # Save the actor
        # torch.save(self.agent.actor.state_dict(),
        #            file_name + "_actor_params.pth")
        logging.info(msg=f'Evaluation: {total_rewards / counter}')

    def train_agent(self):
        """
        Train the agent
        :param max_epi_steps: Maximum number of steps for each episode
        """
        state, _ = self.env.reset()
        for _ in range(self.max_epi_steps):
            # Execute action and add to memory.
            if len(self.memory) < self.batch_size + 1:
                action = self.env.action_space.sample()
            else:
                action = self.agent.select_action_from_policy(state=state,
                                                              evaluation=False)
            next_state, reward, terminated, truncated, _ = self.env.step(
                action)

            done = terminated or truncated
            self.memory.add(state, action, reward, next_state, done)
            # Training the world model and the agent
            if len(self.memory) > self.batch_size:
                statistics = self.memory.get_statistics()
                self.agent.world_model.set_statistics(statistics)
                for _ in range(self.train_world_times):
                    transitions = self.memory.sample(
                        batch_size=self.batch_size)
                    self.agent.train_world_model(
                        statistics=statistics,
                        transitions=transitions)
                for _ in range(self.train_agent_times):
                    transitions = self.memory.sample(
                        batch_size=self.batch_size)
                    # if self.use_dyna:
                    #     self.agent.dyna_generate_and_train(transitions, self.env)
                    # else:
                    self.agent.train_policy(transitions)

            # Do evaluation for every 200
            self.current_step += 1
            if self.current_step % 1000 == 0:
                break
            if done:
                break
            # Move to the next state
            state = next_state

    def train_loop(self):
        """
        The main loop. Call Tranin or evaluation.

        """
        self.evaluate()
        with logging_redirect_tqdm():
            for _ in trange(self.est_epi):
                self.train_agent()
                self.evaluate()
