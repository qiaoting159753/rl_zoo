import numpy as np
from datetime import datetime


class Runner:
    """
    A class that responsible for training and evaluating.

    """
    def __init__(self, env, agent, memory):
        # Should be Goal Conditioned.
        self.use_bound = False
        self.use_actor_mve = False

        self.use_actor_pg = False
        self.use_critic_mve = False
        self.use_critic_steve = True

        self.use_dyna = False
        self.horizon = 5

        self.date_and_time = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
        self.total_steps = 0
        self.batch_size = 128
        self.evaluation_array = [[], [], [], [], []]
        self.env_name = "HalfCheetah-v4"
        # self.env = PusherEnv()
        self.env = env
        self.memory = memory
        self.agent = agent

    def evaluate(self, max_epi_steps=1000, num_eval=5):
        """

        :param max_epi_steps: Each episode, the maximum number of steps.
        :param num_eval: Number of episodes to evaluate the agent.
        """
        total_rewards = 0
        reward_error = 0
        dynamic_error = 0
        counter = 1

        for _ in range(num_eval):
            state, _ = self.env.reset()
            for _ in range(max_epi_steps):
                action = self.agent.act(state)
                next_state, reward, terminate, truncate, _ = self.env.step(
                    action)
                total_rewards += reward

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

        avg_rewards = total_rewards / num_eval
        reward_error /= counter
        dynamic_error /= counter

        self.evaluation_array[0].append(avg_rewards)
        self.evaluation_array[1].append(reward_error)
        self.evaluation_array[2].append(dynamic_error)
        self.evaluation_array[3].append(self.total_steps)
        self.evaluation_array[4].append(0)
        eval_array = np.array(self.evaluation_array)

        # Save the metrics
        param_list = "_" + str(self.use_bound) + "_" + str(self.use_dyna) + \
                     "_" + str(self.use_actor_mve) + "_" + \
                     str(self.use_actor_pg) + "_" + str(self.use_critic_mve) + \
                     "_" + str(self.use_critic_steve) + "_"
        training_str = "2_2_sac_act_"

        file_name = self.env_name + param_list + training_str + self.date_and_time

        np.savetxt(file_name + "_eval_rewards.csv",
                   eval_array, delimiter=",")
        # Save the actor
        # torch.save(self.agent.actor.state_dict(),
        #            file_name + "_actor_params.pth")
        # logging.info(msg=f'Evaluation: {total_rewards / counter}')

    def train_agent(self, max_epi_steps=1000):
        """
        Train the agent
        :param max_epi_steps: Maximum number of steps for each episode
        """
        state, _ = self.env.reset()
        for _ in range(max_epi_steps):
            # Execute action and add to memory.
            if len(self.memory) < self.batch_size + 1:
                action = self.env.action_space.sample()
            else:
                action = self.agent.act(obs=state, sample=True)
            next_state, reward, terminated, truncated, _ = self.env.step(
                action)
            done = terminated or truncated
            self.memory.add(state, action, reward, next_state, done)
            # Training the world model and the agent
            if len(self.memory) > self.batch_size:
                statistics = self.memory.get_statistics()
                self.agent.world_model.set_statistics(statistics)
                for k in range(1):
                    for l in range(1):
                        # Step 1: Get Model-Free training.
                        transitions = self.memory.sample(
                            batch_size=self.batch_size)
                        # Step 2: Get the world model training.
                        self.agent.train_world_model(
                            statistics=statistics,
                            transitions=transitions)
                        # Step 3: Using the MVE
                        self.agent.train_policy(transitions)
                    # self.agent.dyna_generate_and_train(
                    # transitions=transitions)
            # Do evaluation for every 200
            self.total_steps += 1
            if self.total_steps % 1000 == 0:
                self.evaluate_agent()
                break
            # Move to the next state
            state = next_state
            if done:
                break