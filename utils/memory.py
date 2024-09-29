import random
import numpy as np


class SumTree(object):
    """
    A sum tree data structure for storing replay priorities.

    A sum tree is a complete binary tree whose leaves contain values called
    priorities. Internal nodes maintain the sum of the priorities of all leaf
    nodes in their subtree.

    For capacity = 4, the tree may look like this:

                +---+
                |2.5|
                +-+-+
                    |
            +-------+--------+
            |                |
        +-+-+            +-+-+
        |1.5|            |1.0|
        +-+-+            +-+-+
            |                |
        +----+----+      +----+----+
        |         |      |         |
    +-+-+     +-+-+  +-+-+     +-+-+
    |0.5|     |1.0|  |0.5|     |0.5|
    +---+     +---+  +---+     +---+

    This is stored in a list of numpy arrays:
    self.nodes = [ [2.5], [1.5, 1], [0.5, 1, 0.5, 0.5] ]

    For conciseness, we allocate arrays as powers of two, and pad the excess
    elements with zero values.

    This is similar to the usual array-based representation of a complete binary
    tree, but is a little more user-friendly.
    """

    def __init__(self, max_size: int):
        self.levels = [np.zeros(1)]
        # Tree construction
        # Double the number of nodes at each level
        level_size = 1
        while level_size < max_size:
            level_size *= 2
            self.levels.append(np.zeros(level_size))

    def sample_value(self, query_value: int = None) -> int:
        """Samples an element from the sum tree.

        Each element has probability p_i / sum_j p_j of being picked, where p_i is
        the (positive) value associated with node i (possibly unnormalized).

        Args:
            query_value: float in [0, 1], used as the random value to select a sample.
            If None, will select one randomly in [0, 1).

        Returns:
            int, a random element from the sum tree.
        """
        # Sample a value in range [0, R), where R is the value stored at the root.
        query_value = random.random() if query_value is None else query_value
        query_value *= self.levels[0][0]
        return self._retrieve([query_value])[0]

    def sample_simple(self, batch_size: int) -> list[int]:
        """
        Samples indices from the sum tree based on a given batch size.

        Batch binary search through sum tree.

        Sample a priority between 0 and the max priority and then search the tree for the corresponding index

        Args:
            batch_size (int): The number of indices to sample.

        Returns:
            numpy.ndarray: An array of sampled indices.
        """
        values = np.random.uniform(0, self.levels[0][0], size=batch_size)
        return self._retrieve(values)

    def sample_stratified(self, batch_size: int) -> list[int]:
        """Performs stratified sampling using the sum tree.

        Let R be the value at the root (total value of sum tree). This method will
        divide [0, R) into batch_size segments, pick a random number from each of
        those segments, and use that random number to sample from the sum_tree. This
        is as specified in Schaul et al. (2015).

        PER Paper: https://arxiv.org/pdf/1511.05952.pdf

        Args:
            batch_size: int, the number of strata to use.

        Returns:
            list of batch_size elements sampled from the sum tree.
        """

        bounds = np.linspace(0.0, 1.0, batch_size + 1)

        segments = [(bounds[i], bounds[i + 1]) for i in range(batch_size)]

        query_values = [
            random.uniform(segment[0], segment[1]) * self.levels[0][0]
            for segment in segments
        ]
        return self._retrieve(query_values)

    def _retrieve(self, values: np.ndarray) -> list[int]:
        """
        Retrieves the indices of the values in the sum tree that correspond to the given array of values.

        Args:
            values (np.ndarray): The array of values for which to retrieve the indices.

        Returns:
            list[int]: The indices of the values in the sum tree.

        """
        ind = np.zeros(len(values), dtype=int)
        for nodes in self.levels[1:]:
            ind *= 2
            left_sum = nodes[ind]
            # right_sum = nodes[ind + 1]

            is_greater = np.greater(values, left_sum)

            # If value > left_sum -> go right (+1), else go left (+0)
            ind += is_greater

            # If we go right, we only need to consider the values in the right tree
            # so we subtract the sum of values in the left tree
            values -= left_sum * is_greater

        return ind

    def set(self, ind: int, new_priority: float) -> None:
        """
        Set the priority of a node at a given index.

        Args:
            ind (int): The index of the node.
            new_priority (float): The new priority value.

        Returns:
            None
        """
        priority_diff = new_priority - self.levels[-1][ind]

        for nodes in self.levels[::-1]:
            np.add.at(nodes, ind, priority_diff)
            ind //= 2

    def batch_set(self, ind: list[int], new_priority: list[float]) -> None:
        """
        Batch update the priorities of multiple nodes in the sum tree.

        Args:
            ind (list[int]): The indices of the nodes to update.
            new_priority (list[float]): The new priorities to assign to the nodes.

        Returns:
            None
        """

        # Confirm we don't increment a node twice
        ind, unique_ind = np.unique(ind, return_index=True)
        priority_diff = new_priority[unique_ind] - self.levels[-1][ind]

        for nodes in self.levels[::-1]:
            # Best with numpy >= 1.26.4 for optimised add.at
            np.add.at(nodes, ind, priority_diff)
            ind //= 2


"""
Example Implemtnations:
https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py
https://github.com/sfujim/LAP-PAL/blob/master/continuous/utils.py

"""


class PrioritizedReplayBuffer:
    """
    A prioritized replay buffer implementation for reinforcement learning.

    This buffer stores experiences and allows for efficient sampling based on priorities.
    Experiences are stored in the order: state, action, reward, next_state, done, ...

    Args:
        max_capacity (int): The maximum capacity of the buffer.
        **priority_params: Additional parameters for priority calculation.

    Attributes:
        priority_params (dict): Additional parameters for priority calculation.
        max_capacity (int): The maximum capacity of the buffer.
        current_size (int): The current size of the buffer.
        memory_buffers (list): An array of buffers for each experience type.
        tree (SumTree): The SumTree data structure for efficient sampling based on priorities.
        tree_pointer (int): The location to add the next item into the tree.
        max_priority (float): The maximum priority value in the buffer.
        beta (float): The beta parameter for importance weight calculation.

    Methods:
        __len__(): Returns the current size of the buffer.
        add(state, action, reward, next_state, done, *extra): Adds a single experience to the buffer.
        sample_uniform(batch_size): Samples experiences uniformly from the buffer.
        sample_priority(batch_size): Samples experiences from the buffer based on priorities.
        sample_inverse_priority(batch_size): Samples experiences from the buffer based on inverse priorities.
        update_priorities(indices, priorities): Updates the priorities of the buffer at the given indices.
        flush(): Flushes the memory buffers and returns the experiences in order.
        sample_consecutive(batch_size): Randomly samples consecutive experiences from the memory buffer.
    """

    def __init__(
            self,
            max_capacity: int = int(1e6),
            min_priority: float = 1e-4,
            beta: float = 0.4,
            d_beta: float = 6e-7,
            **priority_params,
    ):
        self.max_capacity = max_capacity

        # size is the current size of the buffer
        self.current_size = 0

        # Functionally is an array of buffers for each experience type
        self.memory_buffers = []
        # 0 state = []
        # 1 action = []
        # 2 reward = []
        # 3 next_state = []
        # 4 done = []
        # 5 ... = [] e.g. log_prob = []
        # n ... = []

        # The SumTree is an efficient data structure for sampling based on priorities
        self.sum_tree = SumTree(self.max_capacity)
        self.inverse_tree = SumTree(self.max_capacity)

        # The location to add the next item into the tree - index for the SumTree
        self.tree_pointer = 0

        # Minimum priroity (aka epsilon), prevents zero probabilities
        self.min_priority = min_priority

        # Determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.init_beta = beta
        self.beta = self.init_beta
        self.d_beta = d_beta

        # Current max priority
        self.max_priority = 1.0

    def __len__(self) -> int:
        """
        Returns the current size of the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return self.current_size

    def add(self, state, action, reward, next_state, done, *extra) -> None:
        """
        Adds a single experience to the prioritized replay buffer.

        Data is expected to be stored in the order: state, action, reward, next_state, done, ...

        Args:
            state: The current state of the environment.
            action: The action taken in the current state.
            reward: The reward received for taking the action.
            next_state: The next state of the environment after taking the action.
            done: A flag indicating whether the episode is done after taking the action.
            *extra: Extra is a variable list of extra experience data to be added (e.g. log_prob).

        Returns:
            None
        """
        experience = [state, action, reward, next_state, done, *extra]

        # Iterate over the list of experiences (state, action, reward, next_state, done, ...) and add them to the buffer
        for index, exp in enumerate(experience):
            # Dynamically create the full memory size on first experience
            if index >= len(self.memory_buffers):
                # NOTE: This is a list of numpy arrays in order to use index extraction in sample O(1)
                memory = np.array([None] * self.max_capacity)
                self.memory_buffers.append(memory)

            # This adds to the latest position in the buffer
            self.memory_buffers[index][self.tree_pointer] = exp

        new_priority = self.max_priority
        self.sum_tree.set(self.tree_pointer, new_priority)

        self.tree_pointer = (self.tree_pointer + 1) % self.max_capacity
        self.current_size = min(self.current_size + 1, self.max_capacity)

    def sample_uniform(self, batch_size: int) -> tuple:
        """
        Samples experiences uniformly from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple containing the sampled experiences and their corresponding indices.
                - Experiences are returned in the order: state, action, reward, next_state, done, ...
                - The indices represent the indices of the sampled experiences in the buffer.
        """
        # If batch size is greater than size we need to limit it to just the data that exists
        batch_size = min(batch_size, self.current_size)
        indices = np.random.randint(self.current_size, size=batch_size)

        # Extracts the experiences at the desired indices from the buffer
        experiences = []
        for buffer in self.memory_buffers:
            # NOTE: we convert back to a standard list here
            experiences.append(buffer[indices].tolist())

        return (*experiences, indices.tolist())

    def _importance_sampling_prioritised_weights(
            self, indices: list[int], weight_normalisation="batch"
    ) -> np.ndarray:
        """
        Calculates the importance-sampling weights for prioritized replay and prioritises based on population max.

        PER Paper: https://arxiv.org/pdf/1511.05952.pdf

        Args:
            indices (list[int]): A list of indices representing the transitions to calculate weights for.
            weight_normalisation (str): The type of weight normalisation to use. Options are "batch" or "population".

        Returns:
            np.ndarray: An array of importance-sampling weights.

        Notes:
            - The importance-sampling weights are used to compensate for the non-uniform probabilities of sampling transitions.
            - The weights are calculated using the formula w_i = (1/N * 1/P(i))^β, where N is the current size of the replay buffer,
              P(i) is the priority of transition i, and β is a hyperparameter.
            - The weights are then normalized by dividing them by the maximum weight to ensure stability.
        """

        max_value = self.sum_tree.levels[0][0]

        priorities = self.sum_tree.levels[-1][indices]
        probabilities = priorities / max_value

        weights = (probabilities * self.current_size) ** (-self.beta)

        # Batch normalisation is the default and normalises the weights by the maximum weight in the batch
        if weight_normalisation == "batch":
            max_weight = weights.max()
        # Population normalisation normalises the weights by the maximum weight in the population (buffer)
        elif weight_normalisation == "population":
            p_min = (
                    self.sum_tree.levels[-1][: self.current_size].min()
                    / self.sum_tree.levels[0][0]
            )
            max_weight = (p_min * self.current_size) ** (-self.beta)

        weights /= max_weight

        return weights

    def sample_priority(
            self,
            batch_size: int,
            sampling: str = "stratified",
            weight_normalisation: str = "batch",
    ) -> tuple:
        """
        Samples experiences from the prioritized replay buffer.

        Stratifed vs Simple: https://www.sagepub.com/sites/default/files/upm-binaries/40803_5.pdf

        Args:
            batch_size (int): The number of experiences to sample.
            stratified (bool): Whether to use stratified priority sampling.
            weight_normalisation (str): The type of weight normalisation to use. Options are "batch" or "population".

        Returns:
            tuple: A tuple containing the sampled experiences, indices, and weights.
                - Experiences are returned in the order: state, action, reward, next_state, done, ...
                - The indices represent the indices of the sampled experiences in the buffer.
                - The weights represent the importance weights for each sampled experience.
        """
        # If batch size is greater than size we need to limit it to just the data that exists
        batch_size = min(batch_size, self.current_size)

        if sampling == "simple":
            indices = self.sum_tree.sample_simple(batch_size)
        elif sampling == "stratified":
            indices = self.sum_tree.sample_stratified(batch_size)
        else:
            raise ValueError(f"Unkown sampling scheme: {sampling}")

        weights = self._importance_sampling_prioritised_weights(
            indices, weight_normalisation=weight_normalisation
        )

        # We therefore exploit the flexibility of annealing the amount of importance-sampling
        # correction over time, by defining a schedule on the exponent β that reaches 1 only at the end of
        # learning. In practice, we linearly anneal β from its initial value β0 to 1. Note that the choice of this
        # hyperparameter interacts with choice of prioritization exponent α; increasing both simultaneously
        # prioritizes sampling more aggressively at the same time as correcting for it more strongly.
        self.beta = min(self.beta + self.d_beta, 1.0)

        # Extracts the experiences at the desired indices from the buffer
        experiences = []
        for buffer in self.memory_buffers:
            # NOTE: we convert back to a standard list here
            experiences.append(buffer[indices].tolist())

        return (
            *experiences,
            indices.tolist(),
            weights.tolist(),
        )

    def sample_inverse_priority(self, batch_size: int) -> tuple:
        """
        Samples experiences from the buffer based on inverse priorities.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple containing the sampled experiences, indices, and weights.
                - Experiences are returned in the order: state, action, reward, next_state, done, ...
                - The indices represent the indices of the sampled experiences in the buffer.
                - The weights represent the inverse importance weights for each sampled experience.

        """
        # If batch size is greater than size we need to limit it to just the data that exists
        batch_size = min(batch_size, self.current_size)

        top_value = self.sum_tree.levels[0][0]

        # TODO add inverse (1 - prob into SumTree instead)
        # Inverse based on paper for LA3PD - https://arxiv.org/abs/2209.00532
        reversed_priorities = top_value / (
                self.sum_tree.levels[-1][: self.current_size] + 1e-6
        )

        self.inverse_tree.batch_set(np.arange(self.current_size), reversed_priorities)

        indices = self.inverse_tree.sample_simple(batch_size)

        # Extracts the experiences at the desired indices from the buffer
        experiences = []
        for buffer in self.memory_buffers:
            # NOTE: we convert back to a standard list here
            experiences.append(buffer[indices].tolist())

        return (
            *experiences,
            indices.tolist(),
            reversed_priorities[indices].tolist(),
        )

    def update_priorities(self, indices: list[int], priorities: list[float]) -> None:
        """
        Update the priorities of the replay buffer at the given indices.

        Parameters:
        - indices (array-like): The indices of the replay buffer to update.
        - priorities (array-like): The new priorities to assign to the specified indices.

        Returns:
        None
        """
        self.max_priority = max(priorities.max(), self.max_priority)
        self.sum_tree.batch_set(indices, priorities)

    def flush(self) -> list[tuple]:
        """
        Flushes the memory buffers and returns the experiences in order.

        Returns:
            experiences (list): The full memory buffer in order.
        """
        experiences = []
        for buffer in self.memory_buffers:
            # NOTE: we convert back to a standard list here
            experiences.append(buffer[0: self.current_size].tolist())
        # self.clear()
        return experiences

    def sample_consecutive(self, batch_size: int) -> tuple:
        """
        Randomly samples consecutive experiences from the memory buffer.

        Args:
            batch_size (int): The number of consecutive experiences to sample.

        Returns:
            tuple: A tuple containing the sampled experiences_t and experiences_t+1 and their corresponding indices.
                - Experiences are returned in the order: state_i, action_i, reward_i, next_state_i, done_i, ..._i, state_i+1, action_i+1, reward_i+1, next_state_i+1, done_i+1, ..._+i
                - The indices represent the indices of the sampled experiences in the buffer.

        """
        # If batch size is greater than size we need to limit it to just the data that exists
        batch_size = min(batch_size, self.current_size)

        candididate_indices = list(range(self.current_size - 1))

        # A list of candidate indices includes all indices.
        sampled_indices = []  # randomly sampled indices that is okay.
        # In this way, the sampling time depends on the batch size rather than buffer size.

        # Add in only experiences that are not done and not already sampled.
        while len(sampled_indices) < batch_size:
            # Sample size based on how many still needed.
            idxs = random.sample(candididate_indices, batch_size - len(sampled_indices))
            for i in idxs:
                # Check the experience is not done and not already sampled.
                done = self.memory_buffers[4][i]
                if (not done) and (i not in sampled_indices):
                    sampled_indices.append(i)

        sampled_indices = np.array(sampled_indices)

        experiences = []
        for buffer in self.memory_buffers:
            # NOTE: we convert back to a standard list here
            experiences.append(buffer[sampled_indices].tolist())

        next_sampled_indices = sampled_indices + 1

        for buffer in self.memory_buffers:
            # NOTE: we convert back to a standard list here
            experiences.append(buffer[next_sampled_indices].tolist())

        return (*experiences, sampled_indices.tolist())

    def get_statistics(self) -> dict[str, np.ndarray]:
        """
        Calculate statistics of the replay buffer.

        Returns:
            statistics (dict): A dictionary containing the following statistics:
                - observation_mean: Mean of the observations in the replay buffer.
                - observation_std: Standard deviation of the observations in the replay buffer.
                - delta_mean: Mean of the differences between consecutive observations.
                - delta_std: Standard deviation of the differences between consecutive observations.
        """
        states = np.array(self.memory_buffers[0][: self.current_size].tolist())
        next_states = np.array(self.memory_buffers[3][: self.current_size].tolist())
        diff_states = next_states - states

        # Add a small number to avoid zeros.
        observation_mean = np.mean(states, axis=0) + 0.00001
        observation_std = np.std(states, axis=0) + 0.00001
        delta_mean = np.mean(diff_states, axis=0) + 0.00001
        delta_std = np.std(diff_states, axis=0) + 0.00001

        statistics = {
            "observation_mean": observation_mean,
            "observation_std": observation_std,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
        }
        return statistics

    def clear(self) -> None:
        """
        Clears the prioritised replay buffer.

        Resets the pointer, size, memory buffers, sum tree, max priority, and beta values.
        """
        self.tree_pointer = 0
        self.current_size = 0
        self.memory_buffers = []

        self.sum_tree = SumTree(self.max_capacity)
        self.max_priority = self.min_priority
        self.beta = self.init_beta
