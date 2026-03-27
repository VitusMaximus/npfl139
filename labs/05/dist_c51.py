#!/usr/bin/env python3
# f5419161-0138-4909-8252-ba9794a63e53
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72

import argparse
import collections
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.5")

import copy

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--verify", default=False, action="store_true", help="Verify the loss computation")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--atoms", default=51, type=int, help="Number of atoms.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=1.0, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=500, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=1_000, type=int, help="Target update frequency.")
parser.add_argument("--start_buffer_size", default=1000, type=int, help="Minimum replay buffer size to start training.")
parser.add_argument("--episodes", default=700, type=int, help="Training episodes.")



class Network:
    device = torch.device("cpu")
    # Use the following line instead to use GPU if available.
    # device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model and store it as `self._model`. The model
        # should compute `args.atoms` logits for each action, so for input of shape
        # `[batch_size, *env.observation_space.shape]`, the output should have
        # the shape `[batch_size, env.action_space.n, args.atoms]`. The module
        # `torch.nn.Unflatten` might come handy.
        self.args = args

        self._model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n * args.atoms),
            torch.nn.Unflatten(1, (int(env.action_space.n), args.atoms))
        )

        # Create `self._model.atoms` as uniform grid from 0 to 500 with `args.atoms` elements.
        # We create them as a buffer in `self._model` so they are automatically moved with `.to`.
        self._model.register_buffer("atoms", torch.linspace(0, 500, args.atoms))

        self._model.to(self.device)

        self._target_model = copy.deepcopy(self._model).to(self.device)
        self._target_model.eval()
        for p in self._target_model.parameters():
            p.requires_grad = False

        self._since_update = 0

        # Store the discount factor.
        self.gamma = args.gamma

        # TODO(q_network): Define a suitable optimizer from `torch.optim`.
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)

    @staticmethod
    def compute_loss(
        states_logits: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor,
        next_states_logits: torch.Tensor, atoms: torch.Tensor, gamma: float,
    ) -> torch.Tensor:
        # TODO: Implement the loss computation according to the C51 algorithm.
        # - The `states_logits` are current state logits of shape `[batch_size, actions, atoms]`.
        # - The `actions` are the integral actions taken in the states, of shape `[batch_size]`.
        # - The `rewards` are the rewards obtained after taking the actions, of shape `[batch_size]`.
        # - The `dones` are `torch.float32` indicating whether the episode ended, of shape `[batch_size]`.
        # - The `next_states_logits` are logits of the next states, of shape `[batch_size, actions, atoms]`.
        #   Because they should not be backpropagated through, use an appropriate `.detach()` call.
        # - The `atoms` are the atom values. Your implementation must handle any number of atoms. The
        #   `atoms[0]` is V_MIN (the minimum atom value), `atoms[-1]` is V_MAX (the maximum atom value),
        #   and use `atoms[1] - atoms[0]` as the distance between two consecutive atoms. You can
        #   assume that one of the atoms is always 0.
        # The resulting loss should be the mean of the cross-entropy losses of the individual batch examples.
        #
        # Your implementation most likely needs to be vectorized to pass ReCodEx time limits. Note that you
        # can add given values to a vector of (possibly repeating) tensor indices using `scatter_add_`.
        p_next = torch.softmax(next_states_logits.detach(), dim=2)
        Q_next = (p_next * atoms).sum(dim=2)
        best_actions = Q_next.argmax(dim=1)

        Tz = (rewards[:, None] + gamma * atoms[None, :] * (1 - dones[:, None])).clamp(atoms[0], atoms[-1])
        b = (Tz - atoms[0]) / (atoms[1] - atoms[0])
        l = b.floor().clamp(0, len(atoms) - 1).long()
        u = b.ceil().clamp(0, len(atoms) - 1).long()

        Q_best_next = p_next[torch.arange(next_states_logits.shape[0]), best_actions] # [batch_size, atoms]

        m = torch.zeros_like(Q_best_next)

        same = (u == l)
        if same.any():
            m.scatter_add_(1, l, Q_best_next * same.float())

        
        m.scatter_add_(1, l, Q_best_next * (u.float() - b))
        m.scatter_add_(1, u, Q_best_next * (b - l.float()))
        
        log_p = torch.log_softmax(states_logits[torch.arange(states_logits.shape[0]), actions], dim=1)

        return - torch.sum(m * log_p, dim=1).mean()

    # The training function defers the computation to the `compute_loss` method.
    #
    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
              dones: torch.Tensor, next_states: torch.Tensor) -> None:
        self._since_update += 1

        if self._since_update >= self.args.target_update_freq:
            print("Updating target network...")
            self._target_model.load_state_dict(self._model.state_dict())
            self._since_update = 0

        self._model.train()
        # Pass all arguments to the `compute_loss` method.
        loss = self.compute_loss(
            self._model(states), actions, rewards, dones, self._target_model(next_states), self._model.atoms, self.gamma)
        self._optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            # TODO: Return all predicted Q-values for the given states
            
            logits = self._model(states)
            probs = torch.softmax(logits, dim=2)
            return (probs * self._model.atoms).sum(dim=2)

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    def copy_weights_from(self, other: "Network") -> None:
        self._model.load_state_dict(other._model.state_dict())


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> Callable | None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # When the `args.verify` is set, just return the loss computation function for validation.
    if args.verify:
        return Network.compute_loss

    # Construct the network
    network = Network(env, args)

    # Replay memory; the `max_length` parameter is its maximum capacity.
    replay_buffer = npfl139.ReplayBuffer(max_length=1_000_000)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon
    episodes = 0
    training = True
    while training:
        # Perform episode
        if episodes % 10 == 0:
            print(f"Epsilon: {epsilon:.6f}")
        episodes += 1
        state, done = env.reset()[0], False
        while not done:
            # TODO(q_network): Choose an action.
            # You can compute the q_values of a given state by
            #   q_values = network.predict(state[np.newaxis])[0]
            q_values = network.predict(state[np.newaxis])[0]
            action = q_values.argmax() if np.random.rand() >= epsilon else env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, terminated, next_state))

            # TODO: If the `replay_buffer` is large enough, perform training by
            # sampling a batch of `args.batch_size` uniformly randomly chosen transitions
            # and calling `network.train(states, actions, rewards, dones, next_states)`.
            #
            # The `replay_buffer` offers a method with signature
            #   sample(self, size, replace=True) -> NamedTuple
            # which returns uniformly selected batch of `size` transitions, either with
            # replacement (which is faster, and hence the default) or without.
            # The returned batch is a `Transition` named tuple, each field being
            # a NumPy array containing a batch of corresponding transition components.
            if len(replay_buffer) >= args.start_buffer_size:
                batch = replay_buffer.sample(args.batch_size)
                network.train(batch.state, batch.action, batch.reward, batch.done, batch.next_state)

            state = next_state

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])
        
        if episodes >= args.episodes:
            training = False

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO(q_network): Choose (greedy) action
            q_values = network.predict(state[np.newaxis])[0]
            action = q_values.argmax()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)

    result = main(main_env, main_args)
    if main_args.verify:
        np.testing.assert_allclose(result(
            states_logits=torch.tensor([[[-1.5, 1.2, -1.2], [-0.0, -1.8, -0.1]],
                                        [[-0.2, -0.3, 1.3], [0.5, -1.1, -0.7]],
                                        [[-0.1, 1.9, -0.0], [-0.3, -1.1, -0.1]]]),
            actions=torch.tensor([0, 1, 0]),
            rewards=torch.tensor([0.5, -0.2, 0.7]), dones=torch.tensor([1., 0., 0.]),
            next_states_logits=torch.tensor([[[1.1, 0.2, 0.3], [0.3, 1.1, 1.3]],
                                             [[-0.4, -0.5, -0.6], [2.0, 1.2, 0.4]],
                                             [[-0.3, -0.9, 2.3], [0.7, 0.7, -0.3]]]),
            atoms=torch.tensor([-2., -1., 0.]),
            gamma=0.3).numpy(force=True), 2.170941, atol=1e-5)

        np.testing.assert_allclose(result(
            states_logits=torch.tensor([[[0.1, 1.4, -0.5, -0.8], [0.3, -0.0, -0.2, -0.2]],
                                        [[1.2, -0.8, -1.4, -1.5], [0.1, -0.6, -2.1, -0.3]]]),
            actions=torch.tensor([0, 1]),
            rewards=torch.tensor([0.5, 0.6]), dones=torch.tensor([0., 0.]),
            next_states_logits=torch.tensor([[[0.8, 1.2, -1.2, 0.7], [0.3, 0.4, -1.2, 0.4]],
                                             [[-0.2, 1.0, -1.5, 0.2], [0.2, 0.5, 0.4, -0.9]]]),
            atoms=torch.tensor([-3., 0., 3., 6.]),
            gamma=0.2).numpy(force=True), 1.43398, atol=1e-5)
