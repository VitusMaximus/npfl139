#!/usr/bin/env python3
# f5419161-0138-4909-8252-ba9794a63e53
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
import argparse

import gymnasium as gym
import numpy as np
import torch
from collections import deque
import npfl139
npfl139.require_version("2526.9")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--cards", default=16, type=int, help="Number of cards in the memory game.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=32, type=int, help="Number of episodes to train on.")
parser.add_argument("--evaluate_each", default=1000, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of episodes.")
parser.add_argument("--hidden_layer", default=None, type=int, help="Hidden layer size; default 8*`cards`")
parser.add_argument("--memory_cells", default=None, type=int, help="Number of memory cells; default 2*`cards`")
parser.add_argument("--memory_cell_size", default=None, type=int, help="Memory cell size; default 2*`cards`")


class Agent:
    device = torch.device("cpu")
    # Use the following line instead to use GPU if available.
    # device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self.args = args
        self.env = env

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # TODO: Create suitable layers.
                
                self._key_generator = torch.nn.Sequential(
                    torch.nn.Linear(sum(env.observation_space.nvec), args.hidden_layer),
                    torch.nn.ReLU(),
                    torch.nn.Linear(args.hidden_layer, args.memory_cell_size),
                    torch.nn.Tanh()
                )

                self._policy = torch.nn.Sequential(
                    torch.nn.Linear(sum(env.observation_space.nvec) + args.memory_cell_size, args.hidden_layer),
                    torch.nn.ReLU(),
                    torch.nn.Linear(args.hidden_layer, env.action_space.n)
                )


            def forward(self, memory, observation):
                # Encode the input observation, which is a (card, observation) pair,
                # by representing each element as one-hot and concatenating them, resulting
                # in a vector of length `sum(env.observation_space.nvec)`.
                encoded_input = torch.cat([torch.nn.functional.one_hot(torch.relu(observation[:, i]), dim).float()
                                           for i, dim in enumerate(env.observation_space.nvec)], dim=-1)

                # TODO: Generate a read key for memory read from the encoded input, by using
                # a ReLU-activated hidden layer of size `args.hidden_layer` followed by
                # a fully connected layer with `env.memory_cell_size` units.

                read_key = self._key_generator(encoded_input)

                # TODO: Read the memory using the generated read key. Notably, compute cosine
                # similarity of the key and every memory row, apply softmax to generate
                # a weight distribution over the rows, and finally take a weighted average of
                # the memory rows.

                memory_similarity = torch.cosine_similarity(read_key.unsqueeze(1), memory, dim=-1)
                weight_distribution = torch.softmax(memory_similarity, dim=-1)
                read_value = torch.sum(weight_distribution.unsqueeze(-1) * memory, dim=1)

                # TODO: Using concatenated encoded input and the read value, produce policy logits
                # by applying a ReLU-activated hidden layer of size `args.hidden_layer` followed
                # by an output linear layer with `env.action_space.n` units.

                policy_input = torch.cat([encoded_input, read_value], dim=-1)
                logits = self._policy(policy_input)

                # TODO: Perform a memory write. First generate a write key from the encoded input
                # by applying a ReLU-activated hidden layer of size `args.hidden_layer` followed by
                # a linear layer with `env.memory_cell_size` units, and then write it memory;
                # specifically, prepend the write key as a first memory row and drop the last memory
                # row to keep the memory size constant.

                updated_memory = torch.cat([encoded_input.unsqueeze(1), memory[:, :-1, :]], dim=1)

                # TODO: Return the updated memory and the policy
                return updated_memory, logits

        # Create the agent
        self._model = Model().to(self.device)

        # TODO: Create an optimizer and a loss function.
        self._optimizer = torch.optim.Adam(self._model.parameters())
        self._loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def zero_memory(self):
        # TODO: Return an empty memory. It should be a tensor
        # with shape `[self.args.memory_cells, self.args.memory_cell_size]` on `self.device`.
        return torch.zeros((self.args.memory_cells, self.args.memory_cell_size), device=self.device)

    @npfl139.typed_torch_function(device, torch.int64, torch.int64)
    def _train(self, observations, targets):
        # TODO: Given a batch of sequences of `observations` (each being a (card, symbol) pair),
        # train the network to predict the required `targets`.
        #
        # Specifically, start with a batch of empty memories, and run the agent
        # sequentially as many times as necessary, using `targets` as gold labels.
        #
        # Note that the sequences can be of different length, so you need to pad them
        # to same length and then somehow indicate the length of the individual episodes
        # (one possibility is to add another parameter to `_train`).
        self._model.train()
        self._optimizer.zero_grad()
        memories = torch.zeros((observations.shape[0], self.args.memory_cells, self.args.memory_cell_size), device=self.device)

        loss_sum = 0.0
        for t in range(observations.shape[1]):
            memories, logits = self._model(memories, observations[:, t])
            loss_sum += self._loss_fn(logits, targets[:, t])

        loss = loss_sum / observations.shape[1]
        loss.backward()
        self._optimizer.step()
            
    
        
    def train(self, episodes):
        # TODO: Given a list of episodes, prepare the arguments
        # of the self._train method, and execute it.
        B = len(episodes)
        T = max(len(episode) -1 for episode in episodes)
        
        observations = torch.zeros((B, T, 2), dtype=torch.int64, device=self.device)
        targets = torch.full((B, T),fill_value=-100, dtype=torch.int64, device=self.device)

        for b, episode in enumerate(episodes):
            for t, (observation, action) in enumerate(episode[:-1]):
                observations[b, t] = torch.tensor(observation, dtype=torch.int64)
                targets[b, t] = torch.tensor(action, dtype=torch.int64)

        self._train(observations, targets)

    @npfl139.typed_torch_function(device, torch.float32, torch.int64)
    def predict(self, memory, observation):
        self._model.eval()
        with torch.no_grad():
            memory, logits = self._model(memory, observation)
            return memory, torch.softmax(logits, dim=-1)


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Post-process arguments to default values if not overridden on the command line.
    if args.hidden_layer is None:
        args.hidden_layer = 8 * args.cards
    if args.memory_cells is None:
        args.memory_cells = 2 * args.cards
    if args.memory_cell_size is None:
        args.memory_cell_size = 2 * args.cards

    # Construct the agent.
    agent = Agent(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        observation, memory = env.reset(start_evaluation=start_evaluation, logging=logging)[0], agent.zero_memory()
        rewards, done = 0, False
        while not done:
            # TODO: Find out which action to use.
            memory, action_probs = agent.predict(torch.tensor(memory).unsqueeze(0), torch.tensor(observation, device=agent.device).unsqueeze(0))
            action = np.argmax(action_probs).item()
            memory = memory.squeeze(0)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    mean_returns = deque(maxlen=10)
    # Training
    training = True
    while training:
        # Generate required number of episodes
        for _ in range(args.evaluate_each // args.batch_size):
            episodes = []
            for _ in range(args.batch_size):
                episodes.append(env.expert_episode())

            # Train the agent
            agent.train(episodes)

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        mean_return = np.mean(returns)
        mean_returns.append(mean_return)

        if np.mean(mean_returns) >= 1.0:
            training = False

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        gym.make("npfl139/MemoryGame-v0", cards=main_args.cards), main_args.seed, main_args.render_each)

    main(main_env, main_args)
