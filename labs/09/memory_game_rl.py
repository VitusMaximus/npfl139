#!/usr/bin/env python3
# f5419161-0138-4909-8252-ba9794a63e53
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.9")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--cards", default=4, type=int, help="Number of cards in the memory game.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=16, type=int, help="Number of episodes to train on.")
parser.add_argument("--gradient_clipping", default=1.0, type=float, help="Gradient clipping.")
parser.add_argument("--entropy_regularization", default=0.1, type=float)
parser.add_argument("--evaluate_each", default=1000, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of episodes.")
parser.add_argument("--hidden_layer", default=None, type=int, help="Hidden layer size; default 8*`cards`")
parser.add_argument("--memory_cells", default=None, type=int, help="Number of memory cells; default 2*`cards`")
parser.add_argument("--memory_cell_size", default=None, type=int, help="Memory cell size; default 3/2*`cards`")

parser.add_argument("--gamma", default=1.0, type=float, help="Discount factor.")
parser.add_argument("--learning_rate", default=5e-3, type=float)
parser.add_argument("--baseline_momentum", default=0.01, type=float)
parser.add_argument("--target_return", default=0.25, type=float)
parser.add_argument("--max_episodes", default=1000000, type=int)


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
                self._key_generator = torch.nn.Sequential(
                    torch.nn.Linear(sum(env.observation_space.nvec), args.hidden_layer),
                    torch.nn.ReLU(),
                    torch.nn.Linear(args.hidden_layer, args.memory_cell_size),
                    torch.nn.Tanh(),
                )

                self._policy = torch.nn.Sequential(
                    torch.nn.Linear(sum(env.observation_space.nvec) + args.memory_cell_size, args.hidden_layer),
                    torch.nn.ReLU(),
                    torch.nn.Linear(args.hidden_layer, env.action_space.n),
                )

            def forward(self, memory, observation):
                encoded_input = torch.cat([torch.nn.functional.one_hot(torch.relu(observation[:, i]), dim).float()
                                           for i, dim in enumerate(env.observation_space.nvec)], dim=-1)

                read_key = self._key_generator(encoded_input)
                memory_similarity = torch.cosine_similarity(read_key.unsqueeze(1), memory, dim=-1)
                weight_distribution = torch.softmax(memory_similarity, dim=-1) 
                read_value = torch.sum(weight_distribution.unsqueeze(-1) * memory, dim=1)
                policy_input = torch.cat([encoded_input, read_value], dim=-1)
                logits = self._policy(policy_input)
                updated_memory = torch.cat([encoded_input.unsqueeze(1), memory[:, :-1, :]], dim=1)
                return updated_memory, logits

        self._model = Model().to(self.device)

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)
        self._loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
        self._baseline = torch.zeros(2 * args.cards, device=self.device)

    def zero_memory(self):
        return torch.zeros((self.args.memory_cells, self.args.memory_cell_size), device=self.device)

    @npfl139.typed_torch_function(device, torch.int64, torch.int64, torch.float32)
    def _train(self, observations, actions, returns):
        self._model.train()
        self._optimizer.zero_grad()

        # Per-timestep EMA baseline (momentum 0.01) over valid (non-padded) entries.
        mask = (actions != -1).float()
        with torch.no_grad():
            for t in range(actions.shape[1]):
                count = mask[:, t].sum()
                if count > 0:
                    mean = (returns[:, t] * mask[:, t]).sum() / count
                    self._baseline[t] = (1 - self.args.baseline_momentum) * self._baseline[t] \
                                        + self.args.baseline_momentum * mean
        advantages = (returns - self._baseline[:actions.shape[1]].unsqueeze(0)).detach()

        memories = torch.zeros((observations.shape[0], self.args.memory_cells, self.args.memory_cell_size), device=self.device)

        loss_sum = 0.0
        for t in range(observations.shape[1]):
            memories, logits = self._model(memories, observations[:, t])
            nll = self._loss_fn(logits, actions[:, t])
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = -(log_probs.exp() * log_probs).sum(-1) * mask[:, t]
            loss_sum += (advantages[:, t] * nll - self.args.entropy_regularization * entropy).mean()

        loss = loss_sum / observations.shape[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.args.gradient_clipping)
        self._optimizer.step()

    def train(self, episodes, returns):
        observations, actions = [], []
        for episode in episodes:
            obs, acs, _ = zip(*episode)
            observations.append(torch.stack(obs, 0))
            actions.append(torch.tensor(acs, dtype=torch.int64))

        self._train(
            torch.nn.utils.rnn.pad_sequence(observations, batch_first=True, padding_value=0),
            torch.nn.utils.rnn.pad_sequence(actions, batch_first=True, padding_value=-1),
            torch.nn.utils.rnn.pad_sequence(returns, batch_first=True, padding_value=0.0),
        )

    @npfl139.typed_torch_function(device, torch.float32, torch.int64)
    def predict(self, memory, observation):
        self._model.eval()
        with torch.no_grad():
            memory, logits = self._model(memory, observation)
            return memory, torch.softmax(logits, dim=-1)


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()

    if args.hidden_layer is None:
        args.hidden_layer = 8 * args.cards
    if args.memory_cells is None:
        args.memory_cells = 2 * args.cards
    if args.memory_cell_size is None:
        args.memory_cell_size = 3 * args.cards // 2
    assert sum(env.observation_space.nvec) == args.memory_cell_size

    agent = Agent(env, args)
    rng = np.random.default_rng(args.seed)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        observation, memory = env.reset(start_evaluation=start_evaluation, logging=logging)[0], agent.zero_memory()
        rewards, done = 0, False
        while not done:
            memory, action_probs = agent.predict(
                torch.tensor(memory).unsqueeze(0),
                torch.tensor(observation, device=agent.device).unsqueeze(0),
            )
            action = int(np.argmax(action_probs.squeeze(0)))
            memory = memory.squeeze(0)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    recent_eval_returns = deque(maxlen=5)
    total_episodes = 0
    training = True
    while training:
        for _ in range(max(1, args.evaluate_each // args.batch_size)):
            episodes, returns = [], []
            for _ in range(args.batch_size):
                observation, memory, episode, done = env.reset()[0], agent.zero_memory(), [], False
                while not done:
                    memory, action_probs = agent.predict(
                        torch.tensor(memory).unsqueeze(0),
                        torch.tensor(observation, device=agent.device).unsqueeze(0),
                    )
                    probs = np.asarray(action_probs.squeeze(0), dtype=np.float64)
                    probs = probs / probs.sum()
                    action = int(rng.choice(len(probs), p=probs))
                    memory = memory.squeeze(0)

                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode.append([torch.tensor(observation, dtype=torch.int64), action, reward])
                    observation = next_observation

                _returns = torch.zeros(len(episode), dtype=torch.float32)
                _returns[-1] = episode[-1][-1]
                for t in range(len(episode) - 2, -1, -1):
                    _returns[t] = episode[t][-1] + args.gamma * _returns[t + 1]

                returns.append(_returns)
                episodes.append(episode)
                total_episodes += 1

            agent.train(episodes, returns)

        eval_returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        recent_eval_returns.append(float(np.mean(eval_returns)))

        if (len(recent_eval_returns) == recent_eval_returns.maxlen
                and np.mean(recent_eval_returns) >= args.target_return):
            training = False
        if total_episodes >= args.max_episodes:
            training = False

    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    main_env = npfl139.EvaluationEnv(
        gym.make("npfl139/MemoryGame-v0", cards=main_args.cards), main_args.seed, main_args.render_each)

    main(main_env, main_args)
