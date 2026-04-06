#!/usr/bin/env python3
# f5419161-0138-4909-8252-ba9794a63e53
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.7")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=0.001, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=16, type=int, help="Number of parallel environments.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--critic_learning_rate", default=0.001, type=float, help="Learning rate for critic.")
parser.add_argument("--actor_learning_rate", default=0.0003, type=float, help="Learning rate for actor.")
parser.add_argument("--tiles", default=16, type=int, help="Tiles to use.")
parser.add_argument("--reward_threshold", default=90, type=float, help="Reward threshold for solving the environment.")


class Actor(torch.nn.Module):
    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        super().__init__()
        self._actor_trunk = torch.nn.Sequential(
            torch.nn.EmbeddingBag(max(env.observation_space.nvec) + 1, args.hidden_layer_size, mode="sum"),
            #torch.nn.Linear(args.tiles, args.hidden_layer_size),
            torch.nn.ReLU(),
        )
        self._actor_mus = torch.nn.Linear(args.hidden_layer_size, env.action_space.shape[0])
        self._actor_sds = torch.nn.Linear(args.hidden_layer_size, env.action_space.shape[0])

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk_output = self._actor_trunk(states)
        mus = torch.tanh(self._actor_mus(trunk_output))
        sds = torch.nn.functional.softplus(self._actor_sds(trunk_output)) + 1e-3
        return mus, sds

class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Analogously to paac, your model should contain two components:
        # - an actor, which predicts distribution over the actions, and
        # - a critic, which predicts the value function.
        #
        # The given states are tile encoded, so they are integer indices of
        # tiles intersecting the state. Therefore, you should convert them
        # to dense encoding (one-hot-like, with `args.tiles` ones); or you can
        # even use the `torch.nn.EmbeddingBag` layer.
        #
        # The actor computes `mus` and `sds`, each of shape `[batch_size, actions]`.
        # Compute each independently using states as input, adding a fully connected
        # layer with `args.hidden_layer_size` units and a ReLU activation. Then:
        # - For `mus`, add a fully connected layer with `actions` outputs.
        #   To avoid `mus` moving from the required range, you should apply
        #   properly scaled `torch.tanh` activation.
        # - For `sds`, add a fully connected layer with `actions` outputs
        #   and `torch.exp` or `torch.nn.functional.softplus` activation.
        #
        # The critic should be a usual one, passing states through one hidden
        # layer with `args.hidden_layer_size` ReLU units and then predicting
        # the value function.

        self._entropy_regularization = args.entropy_regularization

        self._actor = Actor(env, args).to(self.device)

        self._critic = torch.nn.Sequential(
            torch.nn.EmbeddingBag(max(env.observation_space.nvec) + 1, args.hidden_layer_size, mode="sum"),
            #torch.nn.Linear(args.tiles, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1),
        ).to(self.device)

        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=args.actor_learning_rate)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=args.critic_learning_rate)

        self._critic_loss = torch.nn.MSELoss()

    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.int64, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Run the model on given `states` and compute `sds`, `mus` and predicted values.
        # Then create `action_distribution` using `torch.distributions.Normal` class and
        # the computed `mus` and `sds`.
        #
        # TODO: Train the actor using the sum of the following two losses:
        # - REINFORCE loss, i.e., the negative log likelihood of the `actions` in the
        #   `action_distribution` (using the `log_prob` method). You then need to sum
        #   the log probabilities of the action components in a single batch example.
        #   Finally, multiply the resulting vector by `(returns - baseline)`
        #   and compute its mean. Be sure to let the gradient flow only where it should.
        # - negative value of the distribution entropy (use `entropy` method of
        #   the `action_distribution`) weighted by `args.entropy_regularization`.
        #
        # Train the critic using mean square error of the `returns` and predicted values.
        
        self._critic.train()
        self._actor.train()
        
        self._critic_optimizer.zero_grad()
        self._actor_optimizer.zero_grad()

        critic_values = self._critic(states).squeeze(-1)
        critic_loss = self._critic_loss(critic_values, returns)
        critic_loss.backward()
        self._critic_optimizer.step()

        values = critic_values.detach()
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        mus, sds = self._actor(states)
        action_distribution = torch.distributions.Normal(mus, sds)
        log_probs = action_distribution.log_prob(actions)
        nll = -log_probs.sum(dim=1)

        entropy = action_distribution.entropy().sum(dim=1) * self._entropy_regularization
        actor_loss = (advantages * nll - entropy).mean()
        actor_loss.backward()
        self._actor_optimizer.step()        


    @npfl139.typed_torch_function(device, torch.int64)
    def predict_actions(self, states: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Return predicted action distributions (mus and sds).
        self._actor.eval()
        with torch.no_grad():
            mus, sds = self._actor(states)
        return mus.cpu(), sds.cpu()

    @npfl139.typed_torch_function(device, torch.int64)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return predicted state-action values.
        self._critic.eval()
        with torch.no_grad():
            values = self._critic(states).squeeze(-1)
        return values.cpu()


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
            # TODO: Predict an action using the greedy policy.
            action = agent.predict_actions(state[None])[0][0]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create the vectorized environment, using the SAME_STEP autoreset mode.
    vector_env = gym.make_vec("MountainCarContinuous-v0", args.envs, gym.VectorizeMode.ASYNC,
                              wrappers=[lambda env: npfl139.DiscreteMountainCarWrapper(env, tiles=args.tiles)],
                              vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    states = vector_env.reset(seed=args.seed)[0]
    last_100_ep_returns = deque(maxlen=10)
    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # TODO: Predict action distribution using `agent.predict_actions`
            # and then sample it using for example `np.random.normal`. Do not
            # forget to clip the actions to the `env.action_space.{low,high}`
            # range, for example using `np.clip`.
            mus, sds = agent.predict_actions(states)
            actions = np.random.normal(mus, sds)
            actions = np.clip(actions, env.action_space.low, env.action_space.high)

            # Perform steps in the vectorized environment
            next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
            dones = terminated | truncated

            # TODO(paac): Compute estimates of returns by one-step bootstrapping
            next_values = agent.predict_values(next_states) # [envs]
            returns = rewards + args.gamma * next_values * (1 - dones)
            # TODO(paac): Train agent using current states, chosen actions and estimated returns.
            agent.train(states, actions, returns)

            states = next_states

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        last_100_ep_returns.append(np.mean(returns))

        if np.mean(last_100_ep_returns) >= args.reward_threshold:
            break

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteMountainCarWrapper(gym.make("MountainCarContinuous-v0"), tiles=main_args.tiles),
        main_args.seed, main_args.render_each)

    main(main_env, main_args)
