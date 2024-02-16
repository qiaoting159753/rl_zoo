import numpy as np
import math
from datetime import datetime
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

class Trainer:
    """
    A class that responsible for training and evaluating.

    """

    def __init__(self, generate_results, env, agent, memory, device, name,
                 use_mbrl, logger):

        # Should be Goal Conditioned.
        self.logger = logger
        self.use_mbrl = use_mbrl
        self.name = name
        self.generate_results = generate_results
        self.device = device

        self.batch_size = 128
        self.max_steps = 1000000
        self.max_epi_steps = 1000
        self.num_eval = 5
        self.est_epi = int(
            math.ceil(self.max_steps / self.max_epi_steps) * 1.5)
        self.train_world_times = 1
        self.train_agent_times = 1

        self.current_step = 0
        self.date_and_time = datetime.now().strftime('%y_%m_%d_%H_%M_%S')

        self.evaluation_array = [[], [], [], [], []]
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
            state = self.env.reset()
            for _ in range(self.max_epi_steps):
                action = self.agent.select_action_from_policy(state,
                                                              evaluation=True)
                next_state, reward, done, _ = self.env.step(action)
                total_rewards += reward

                # state_tensor = torch.FloatTensor(next_state).to(device=self.device)
                # state_tensor = state_tensor.unsqueeze(dim=0)

                # num_act_per = 10
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

        self.logger.info(f'Evaluation: {total_rewards / self.num_eval}')

        if self.generate_results:
            # Save the metrics
            file_name = (self.env.domain + "_" + self.env.task + "_" +
                         self.name + "_" + self.date_and_time)
            np.savetxt(file_name + "_eval_rewards.csv",
                       eval_array, delimiter=",")
            # Save the actor
            # torch.save(self.agent.actor.state_dict(),
            #            file_name + "_actor_params.pth")

    def train_agent(self):
        """
        Train the agent
        :param max_epi_steps: Maximum number of steps for each episode
        """
        state = self.env.reset()

        for _ in range(self.max_epi_steps):
            # Execute action and add to memory.
            if len(self.memory) < self.batch_size + 1:
                action = np.random.uniform(-1, 1, (self.env.action_num,))
            else:
                action = self.agent.select_action_from_policy(state=state,
                                                              evaluation=False)

            next_state, reward, done, _ = self.env.step(action)
            self.memory.add(state, action, reward, next_state, done)
            # Training the world model and the agent
            if len(self.memory) > self.batch_size:
                if self.use_mbrl and self.agent.type == "mbrl":
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
        with logging_redirect_tqdm():
            for i in trange(self.est_epi):
                self.evaluate(i)
                self.train_agent()
