#!/usr/bin/env python3
# f5419161-0138-4909-8252-ba9794a63e53
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72

import argparse
import json

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.10")

from collections import deque

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="npfl139/SingleCollect-v0", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--clip_epsilon", default=0.2, type=float, help="Clipping epsilon.")
parser.add_argument("--entropy_regularization", default=0.01, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=8, type=int, help="Workers during experience collection.")
parser.add_argument("--epochs", default=4, type=int, help="Epochs to train each iteration.")
parser.add_argument("--evaluate_each", default=10, type=int, help="Evaluate each given number of iterations.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--trace_lambda", default=0.95, type=float, help="Traces factor lambda.")
parser.add_argument("--worker_steps", default=128, type=int, help="Steps for each worker to perform.")


class Agent:
    # Use GPU if available.
    device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self._args = args

        # TODO: Create an actor using a single hidden layer with `args.hidden_layer_size`
        # units and ReLU activation, produce a policy with `env.action_space.n` discrete actions.
        self._actor = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n),
            torch.nn.Softmax(dim=-1)
        ).to(self.device)

        # TODO: Create a critic (value predictor) consisting of a single hidden layer with
        # `args.hidden_layer_size` units and ReLU activation, and an output layer with a single output.
        self._critic = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1)
        ).to(self.device)

        self._critic_loss = torch.nn.MSELoss()

        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=args.learning_rate)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=args.learning_rate)

    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, action_probs: torch.Tensor,
              advantages: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Perform a single training step of the PPO algorithm.
        # For the policy model, the sum is the sum of:
        # - the PPO loss, where `self._args.clip_epsilon` is used to clip the probability ratio
        # - the entropy regularization with coefficient `self._args.entropy_regularization`.
        #   You can compute it for example using the `torch.distributions.Categorical` class.
        self._actor.train()
        self._actor_optimizer.zero_grad()

        batch_indices = torch.arange(action_probs.shape[0], device=self.device)
        action_probs_new = self._actor(states)[batch_indices, actions]

        r = action_probs_new / action_probs

        ppo_loss = torch.min(r * advantages, torch.clip(r, 1 - self._args.clip_epsilon, 1 + self._args.clip_epsilon) * advantages).mean()
        entropy = torch.distributions.Categorical(action_probs_new).entropy().mean()
        actor_loss = -ppo_loss - self._args.entropy_regularization * entropy

        actor_loss.backward()
        self._actor_optimizer.step()



        # TODO: The critic model is trained in a standard way, by using the MSE
        # error between the predicted value function and target returns.
        self._critic.train()
        self._critic_optimizer.zero_grad()
        predicted_values = self._critic(states).squeeze(-1)
        critic_loss = self._critic_loss(predicted_values, returns)
        critic_loss.backward()
        self._critic_optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return predicted action probabilities.
        self._actor.eval()
        return self._actor(states).cpu()
    
    @npfl139.typed_torch_function(device, torch.float32)
    def sample_actions(self, states: torch.Tensor) -> np.ndarray:
        self._actor.eval()
        action_probs = self._actor(states)
        action_dist = torch.distributions.Categorical(action_probs)
        return action_dist.sample().cpu()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return estimates of value function.
        self._critic.eval()
        return self._critic(states).squeeze(-1).cpu()

    # Serialization methods.
    def save_actor(self, path: str) -> None:
        torch.save(self._actor.state_dict(), path)

    def load_actor(self, path: str) -> None:
        self._actor.load_state_dict(torch.load(path, map_location=self.device))

    @staticmethod
    def save_args(path: str, args: argparse.Namespace) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(vars(args), file, ensure_ascii=False, indent=2)

    @staticmethod
    def load_args(path: str) -> argparse.Namespace:
        with open(path, "r", encoding="utf-8-sig") as file:
            args = json.load(file)
        return argparse.Namespace(**args)


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Construct the agent.
    agent = Agent(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            # TODO: Predict an action by using a greedy policy.
            action = agent.predict_actions(state[None])[0].argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create an asynchronous vector environment for training.
    vector_env = gym.make_vec(args.env, args.envs, gym.VectorizeMode.ASYNC,
                              vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})

    # Training
    #mean_returns = deque(maxlen=10)

    state = vector_env.reset(seed=args.seed)[0]
    training, iteration = True, 0
    while training:
        # Collect experience. Notably, we collect the following quantities
        # as tensors with the first two dimensions `[args.worker_steps, args.envs]`.
        states, actions, action_probs, rewards, dones, values = [], [], [], [], [], []
        for _ in range(args.worker_steps):
            # TODO: Choose `action`, which is a vector of `args.envs` actions, each
            # sampled from the corresponding policy generated by the `agent.predict`
            # executed on the vector `state`.
            action = agent.sample_actions(state)

            # Perform the environment interaction.
            next_state, reward, terminated, truncated, _ = vector_env.step(action)
            done = terminated | truncated

            # TODO: Compute and collect the required quantities.
            states.append(state)
            actions.append(action)
            action_probs.append(agent.predict_actions(state)[torch.arange(args.envs), action])
            rewards.append(reward)
            dones.append(done)
            values.append(agent.predict_values(state))

            state = next_state

        # TODO: Estimate `advantages` and `returns` (they differ only by the value function estimate)
        # using lambda-return with coefficients `args.trace_lambda` and `args.gamma`.
        # You need to handle both the cases that (a) the last episode is probably unfinished, and
        # (b) there are multiple episodes in the collected data.
        advantages = np.zeros((args.worker_steps, args.envs), dtype=np.float32)
        returns = np.zeros_like(advantages)
        
        #G = np.zeros((args.envs,), dtype=np.float32)
        #One_step_G = np.zeros((args.envs,), dtype=np.float32)
        A = np.zeros((args.envs,), dtype=np.float32)
        next_values = agent.predict_values(state)

        for t in reversed(range(args.worker_steps)):

            #One_step_G = rewards[t] + (1 - dones[t]) * args.gamma * next_values
            #G = args.trace_lambda * (rewards[t] + (1 - dones[t]) * args.gamma * G) + (1 - args.trace_lambda) * One_step_G
            #returns[:, t] = G

            delta_t = rewards[t] + (1 - dones[t]) * args.gamma * next_values - values[t]
            A = delta_t + (1 - dones[t]) * args.gamma * args.trace_lambda * A
            advantages[t, :] = A
            returns[t, :] = A + values[t]

            next_values = values[t]


        # TODO: Train for `args.epochs` using the collected data. In every epoch,
        # you should randomly sample batches of size `args.batch_size` from the collected data.
        # A possible approach is to create a dataset of `(states, actions, action_probs, advantages, returns)`
        # quintuples using a single `torch.utils.data.StackDataset` and then use a dataloader.

        for _ in range(args.epochs):
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(np.stack(states).reshape(-1, *np.stack(states).shape[2:]), device=agent.device),
                torch.tensor(np.stack(actions).reshape(-1), device=agent.device),
                torch.tensor(np.stack(action_probs).reshape(-1), device=agent.device),
                torch.tensor(advantages.reshape(-1), device=agent.device),
                torch.tensor(returns.reshape(-1), device=agent.device)
            )
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            for batch in dataloader:
                agent.train(*batch)

        # Periodic evaluation
        iteration += 1
        if iteration % args.evaluate_each == 0:
            returns = [evaluate_episode() for _ in range(args.evaluate_for)]
            if np.mean(returns) >= 520:
                training = False
            #mean_returns.append(np.mean(returns))
            #if np.mean(mean_returns) >= 510:
            #    training = False

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)

    main(main_env, main_args)
