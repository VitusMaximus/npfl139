#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.6")

import json

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="LunarLander-v3", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=0.01, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=16, type=int, help="Number of parallel environments.")
parser.add_argument("--evaluate_each", default=1000, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--critic_learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--actor_learning_rate", default=0.0003, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="paac_actor.pt", type=str, help="Path to the actor model.")
parser.add_argument("--reward_threshold", default=260, type=float, help="Reward threshold for solving the environment.")



class Agent:
    #device = torch.device("cpu")
    # Use the following line instead to use GPU if available.
    device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Similarly to reinforce with baseline, define two components:
        # - an actor, which predicts distribution over the actions, and
        # - a critic, which predicts the value function.
        #
        # Use independent networks for both of them, each with
        # `args.hidden_layer_size` neurons in one ReLU hidden layer,
        # and train them using Adam with given `args.learning_rate`.
        self.entropy_regularization = args.entropy_regularization



        self._actor = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n),
            torch.nn.LogSoftmax(dim=-1)
        ).to(self.device)

        self._critic = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1)
        ).to(self.device)

        self._critic_loss = torch.nn.MSELoss()
        self._actor_loss = torch.nn.NLLLoss(reduction="none")

        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=args.critic_learning_rate)
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=args.actor_learning_rate)

        #self._critic_scheduler = torch.optim.lr_scheduler.LinearLR(self._critic_optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.episodes - 3_000)
        #self._actor_scheduler = torch.optim.lr_scheduler.LinearLR(self._actor_optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.episodes - 3_000)


    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Train the policy network using policy gradient theorem
        # and the value network using MSE.
        #
        # The `args.entropy_regularization` might be used to include actor
        # entropy regularization -- the assignment can be solved even without
        # it, but my reference solution learns quicklier when using it.
        # In any case, `torch.distributions.Categorical` is a suitable distribution
        # offering the `.entropy()` method.
        self._critic.train()
        self._actor.train()

        critic_values = self._critic(states).squeeze(-1)
        critic_loss = self._critic_loss(critic_values, returns)

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        with torch.no_grad():
            values = self._critic(states).squeeze(-1)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs = self._actor(states)
        nll = self._actor_loss(log_probs, actions)
        entropy = torch.distributions.Categorical(probs=log_probs.exp()).entropy()
        actor_loss = (advantages * nll - self.entropy_regularization * entropy).mean()
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()
        #self._critic_scheduler.step()
        #self._actor_scheduler.step()


    @npfl139.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return predicted action probabilities.
        self._actor.eval()
        with torch.no_grad():
            return self._actor(states).exp().cpu()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return estimates of the value function.
        self._critic.eval()
        with torch.no_grad():
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
    agent = Agent(env, args if not args.recodex else Agent.load_args(args.model_path + ".json"))

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            # TODO: Predict an action using the greedy policy.
            action = agent.predict_actions(state[None])[0].argmax()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # ReCodEx evaluation.
    if args.recodex:
        agent.load_actor(args.model_path)
        while True:
            evaluate_episode(start_evaluation=True)

    # Create the vectorized environment, using the SAME_STEP autoreset mode.
    vector_env = gym.make_vec(args.env, args.envs, gym.VectorizeMode.ASYNC,
                              vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    states = vector_env.reset(seed=args.seed)[0]

    training = True
    while training:

        # Training
        for _ in range(args.evaluate_each):
            # TODO: Choose actions using `agent.predict_actions`.
            probs = agent.predict_actions(states) # [envs, actions]
            actions = torch.multinomial(torch.from_numpy(probs), num_samples=1).squeeze(-1).numpy() # [envs]

            # Perform steps in the vectorized environment
            next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
            dones = terminated | truncated

            # TODO: Compute estimates of returns by one-step bootstrapping
            next_values = agent.predict_values(next_states) # [envs]
            returns = rewards + args.gamma * next_values * (1 - dones)

            # TODO: Train agent using current states, chosen actions and estimated returns.
            agent.train(states, actions, returns)

            states = next_states

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        
        if returns.mean() >= args.reward_threshold:
            break

        

    # Save the agent
    agent.save_actor(args.model_path)
    agent.save_args(args.model_path + ".json", args)

    # Final evaluation
    print("Final evaluation:")
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)

    main(main_env, main_args)
