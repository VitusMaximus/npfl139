#!/usr/bin/env python3
from npfl139 import replay_buffer
import collections
from collections import deque
import argparse

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

import npfl139
npfl139.require_version("2526.5")

import torch
import copy
import numpy as np
import regex as re


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--frame_skip", default=4, type=int, help="Frame skip.")
parser.add_argument("--frame_stack", default=4, type=int, help="Frame stack.")
parser.add_argument("--game", default="Pong", type=str, help="Game to play.")
parser.add_argument("--grayscale", default=True, action=argparse.BooleanOptionalAction, help="Grayscale obs.")
parser.add_argument("--screen_size", default=84, type=int, help="Screen size.")

parser.add_argument("--atoms", default=51, type=int, help="Number of atoms.")
parser.add_argument("--V_max", default=10, type=int, help="Maximum value.")
parser.add_argument("--V_min", default=-10, type=int, help="Minimum value.")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--epsilon", default=1.0, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=400_000, type=int, help="Training steps.")

parser.add_argument("--target_update_freq", default=5_000, type=int, help="Target network update frequency.")
parser.add_argument("--replay_buffer_size", default=100_000, type=int, help="Replay buffer size.")
parser.add_argument("--replay_start_size", default=5_000, type=int, help="Start training after this many steps.")
parser.add_argument("--num_envs", default=8, type=int, help="Number of parallel environments.")



class Network:
    device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self.args = args

        self._model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=args.frame_stack, out_channels=32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, args.atoms * env.action_space.n),
            torch.nn.Unflatten(1, (int(env.action_space.n), args.atoms))
        )

        self._model.to(Network.device)

        self._model.register_buffer("atoms", torch.linspace(args.V_min, args.V_max, args.atoms).to(self.device))

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)

        self._target_model = copy.deepcopy(self._model).to(self.device)
        self._target_model.eval()
        for p in self._target_model.parameters():
            p.requires_grad = False

        self._since_update = 0

        self.gamma = args.gamma


    @staticmethod
    def compute_loss(
        states_logits: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor,
        next_states_logits: torch.Tensor, atoms: torch.Tensor, gamma: float,
    ) -> torch.Tensor:
        p_next = torch.softmax(next_states_logits.detach(), dim=2)
        q_next = torch.sum(p_next * atoms, dim=2)
        best_actions = q_next.argmax(dim=1)

        Tz = (rewards[:, None] + gamma * atoms[None, :] * (1 - dones[:, None])).clamp(atoms[0], atoms[-1])
        b = (Tz - atoms[0]) / (atoms[1] - atoms[0])
        l = b.floor().clamp(0, len(atoms) - 1).long()
        u = b.ceil().clamp(0, len(atoms) - 1).long()
        
        Q_best_next = p_next[torch.arange(next_states_logits.shape[0]), best_actions]

        m = torch.zeros_like(Q_best_next)
        
        same = (u == l)
        if same.any():
            m.scatter_add_(1, l, Q_best_next * same.float())
        
        m.scatter_add_(1, l, Q_best_next * (u.float() - b))
        m.scatter_add_(1, u, Q_best_next * (b - l.float()))
        
        log_p = torch.log_softmax(states_logits[torch.arange(states_logits.shape[0]), actions], dim=1)

        return - torch.sum(m * log_p, dim=1).mean()
        
    
    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
              dones: torch.Tensor, next_states: torch.Tensor) -> None:
        states = states / 255.0
        next_states = next_states / 255.0
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
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=10.0)
        with torch.no_grad():
            self._optimizer.step()


    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        states = states / 255.0
        self._model.eval()
        with torch.no_grad():
            logits = self._model(states)
            probs = torch.softmax(logits, dim=2).to(self.device)
            q_values = torch.sum(probs * self._model.atoms, dim=2)
            return q_values
            
class Observer:
    def __init__(self, num_envs):
        self.episode_returns = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs)
        self.returns = deque(maxlen=100)
        self.steps = 0

    def step(self, rewards, dones, epsilon):
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.steps += 1

        if dones.any():
            self.returns.extend(self.episode_returns[dones])
            self.episode_returns[dones] = 0
            self.episode_lengths[dones] = 0
            
        if self.steps % 100 == 0:
            print(f"Step: {self.steps}, Returns: {np.mean(self.returns):.3f}, Epsilon: {epsilon:.4f}")


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    env = gym.wrappers.AtariPreprocessing(
        env, frame_skip=args.frame_skip, grayscale_obs=args.grayscale, screen_size=args.screen_size)
    env = gym.wrappers.FrameStackObservation(env, stack_size=args.frame_stack)

    network = Network(env, args)

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # Load the agent
        network._model.load_state_dict(torch.load("atari_gamer.pt", map_location=Network.device, weights_only=True))

        # Final evaluation
        while True:
            state, done = env.reset(options={"start_evaluation": True})[0], False
            while not done:
                # Choose a greedy action
                q_values = network.predict(np.expand_dims(state, 0))
                action = q_values.argmax(axis=1)[0]
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    # TODO: Train an agent using for example some distributed-RL algorithm.
    #
    # If you want to create N multithreaded parallel environments, use
    vector_env = ale_py.AtariVectorEnv(
        game=re.sub(r"(?<=[a-z])(?=[A-Z])", "_", args.game).lower(),  # use snake_case for the game name
        num_envs=args.num_envs,  # the requred number of parallel environments,
        frameskip=args.frame_skip, stack_num=args.frame_stack, grayscale=args.grayscale,
        img_height=args.screen_size, img_width=args.screen_size,
        use_fire_reset=False, reward_clipping=False, repeat_action_probability=0.25,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    # There are several Autoreset modes available, see https://farama.org/Vector-Autoreset-Mode.
    # In some situations, the SAME_STEP might be more practical than the default NEXT_STEP mode.


    replay_buffer = npfl139.ReplayBuffer(max_length = args.replay_buffer_size)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    observer = Observer(args.num_envs)

    states, _ = vector_env.reset()
    
    
    epsilon = args.epsilon
    steps = 0
    training = True
    while training:
        steps += 1

        q_values = network.predict(states)
        
        greedy_actions = q_values.argmax(axis=1)
        random_actions = np.random.randint(0, env.action_space.n, size=vector_env.num_envs)

        actions = np.where(np.random.random(vector_env.num_envs) < epsilon, random_actions, greedy_actions)
        
        next_states, rewards, terminated, truncated, info = vector_env.step(actions)
        dones = terminated | truncated

        buffer_next_states = next_states.copy()
        if "final_observation" in info and dones.any():
            for i in range(vector_env.num_envs):
                if dones[i] and info["final_observation"][i] is not None:
                    buffer_next_states[i] = info["final_observation"][i]

        observer.step(rewards, dones, epsilon)   

        replay_buffer.extend([Transition(states[i], actions[i], rewards[i], buffer_next_states[i], dones[i]) for i in range(vector_env.num_envs)])

        if len(replay_buffer) > args.replay_start_size:
            for _ in range(4):
                batch = replay_buffer.sample(args.batch_size)
                network.train(batch.state, batch.action, batch.reward, batch.done, batch.next_state)

        states = next_states

        if epsilon > args.epsilon_final:
            epsilon = np.interp(steps, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])
            
        # Stop and save if the moving average return is > 6
        if len(observer.returns) == observer.returns.maxlen and np.mean(observer.returns) >= 16:
            print("Target score reached! Saving model and exiting.")
            torch.save(network._model.state_dict(), "atari_gamer.pt")
            break
        


        
        
        




if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    assert main_args.render_each in [0, 1], "Option render_each can be only 0 or 1 for Atari games"

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        gym.make(f"ALE/{main_args.game}-v5", frameskip=1, render_mode="human" if main_args.render_each else None),
        main_args.seed)

    main(main_env, main_args)
