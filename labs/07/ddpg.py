#!/usr/bin/env python3
import argparse
import collections
import copy

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.7")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="Pendulum-v1", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=50, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--noise_sigma", default=0.2, type=float, help="UB noise sigma.")
parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
parser.add_argument("--replay_buffer_size", default=100_000, type=int, help="Replay buffer size")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")


class Actor(torch.nn.Module):
    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        super().__init__()
        
        self.register_buffer("_action_low", torch.from_numpy(env.action_space.low))
        self.register_buffer("_action_high", torch.from_numpy(env.action_space.high))

        self._model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.shape[0]),
            torch.nn.Tanh(),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        actions = self._model(states)
        return actions * (self._action_high - self._action_low) / 2.0 + (self._action_high + self._action_low) / 2.0


class Critic(torch.nn.Module):
    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        super().__init__()

        self._model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0] + env.action_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        input = torch.cat([states, actions], dim=1)
        return self._model(input).squeeze(1)



class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create:
        # - An actor, which starts with states and returns actions.
        #   Usually, one or two hidden layers are employed. As in the
        #   paac_continuous, to keep the actions in the required range, you
        #   should apply properly scaled `torch.tanh` activation.
        #
        # - A target actor as the copy of the actor using `copy.deepcopy`.
        #
        # - A critic, starting with given states and actions, producing predicted
        #   returns. The states and actions are usually concatenated and fed through
        #   two more hidden layers, before computing the returns with the last output layer.
        #
        # - A target critic as the copy of the critic using `copy.deepcopy`.
        self._args = args
        
        self._actor = Actor(env, args).to(self.device)
        self._target_actor = copy.deepcopy(self._actor).to(self.device)

        self._critic = Critic(env, args).to(self.device)
        self._target_critic = copy.deepcopy(self._critic).to(self.device)

        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=args.learning_rate)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=args.learning_rate)

        self._critic_loss = torch.nn.MSELoss()

    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Separately train:
        # - the actor using the DPG loss,
        # - the critic using MSE loss.
        #
        # Furthermore, update the target actor and critic networks by exponential moving average
        # with momentum `args.target_tau`. An implementation for EMA update is provided as
        #   npfl139.update_params_by_ema(target: torch.nn.Module, source: torch.nn.Module, tau: float)
        
        self._critic.train()
        self._actor.train()

        self._critic_optimizer.zero_grad()
        self._actor_optimizer.zero_grad()

        critic_values: torch.Tensor = self._critic(states, actions)
        critic_loss: torch.Tensor = self._critic_loss(critic_values, returns)
        critic_loss.backward()
        self._critic_optimizer.step()

        self._critic.requires_grad_(False)

        predicted_actions: torch.Tensor = self._actor(states)
        actor_loss: torch.Tensor = -self._critic(states, predicted_actions).mean()
        actor_loss.backward()
        self._actor_optimizer.step()

        self._critic.requires_grad_(True)


        npfl139.update_params_by_ema(self._target_actor, self._actor, self._args.target_tau)
        npfl139.update_params_by_ema(self._target_critic, self._critic, self._args.target_tau)



    @npfl139.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return predicted actions by the actor.
        self._actor.eval()
        with torch.inference_mode():
            actions = self._actor(states)
        return actions.cpu()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return predicted returns -- predict actions by the target actor
        # and evaluate them using the target critic.
        self._target_actor.eval()
        self._target_critic.eval()
        with torch.inference_mode():
            actions = self._target_actor(states)
            values = self._target_critic(states, actions)
        return values.cpu()


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Construct the agent.
    agent = Agent(env, args)

    # Replay memory of a specified maximum size.
    replay_buffer = npfl139.ReplayBuffer(args.replay_buffer_size, args.seed)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            # TODO: Predict an action by calling `agent.predict_actions`.
            action = agent.predict_actions(state[None])[0]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    noise = OrnsteinUhlenbeckNoise(env.action_space.shape[0], 0, args.noise_theta, args.noise_sigma)
    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            state, done = env.reset()[0], False
            noise.reset()
            while not done:
                # TODO: Predict actions by calling `agent.predict_actions`
                # and adding the Ornstein-Uhlenbeck noise. As in paac_continuous,
                # clip the actions to the `env.action_space.{low,high}` range.
                action = agent.predict_actions(state[None])[0]
                action += noise.sample()
                action = np.clip(action, env.action_space.low, env.action_space.high)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay_buffer.append(Transition(state, action, reward, done, next_state))
                state = next_state

                if len(replay_buffer) < 4 * args.batch_size:
                    continue
                states, actions, rewards, dones, next_states = replay_buffer.sample(args.batch_size)
                # TODO: Perform the training

                next_values = agent.predict_values(next_states)
                returns = rewards + args.gamma * next_values * (1 - dones)
                agent.train(states, actions, returns)

        # Periodic evaluation
        returns = [evaluate_episode(logging=False) for _ in range(args.evaluate_for)]
        print(f"Evaluation after episode {env.episode}: {np.mean(returns):.2f}")

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)

    main(main_env, main_args)
