#!/usr/bin/env python3
import argparse
import collections

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.4")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--continuous", default=0, type=int, help="Use continuous actions.")
parser.add_argument("--frame_skip", default=4, type=int, help="Frame skip.")
parser.add_argument("--frame_stack", default=4, type=int, help="Frame stack.")

parser.add_argument("--agent_path", default="car_racing_agent.pt", type=str, help="Path to the saved model.")

parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--replay_buffer_size", default=500_000, type=int, help="Replay buffer size.")
parser.add_argument("--replay_start_size", default=100_000, type=int, help="Minimum replay buffer size before training.")
parser.add_argument("--epsilon", default=1.0, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=800_000, type=int, help="Training steps.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.00025, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=10_000, type=int, help="Target update frequency.")
parser.add_argument("--steps", default=1_000_000, type=int, help="Training steps.")

parser.add_argument("--num_envs", default=16, type=int, help="Number of parallel environments.")
parser.add_argument("--downsample", default=1, type=int, help="Downsample factor for observations.")    # 1 better



class Permute(torch.nn.Module):
    def __init__(self, *dims):
        super().__init__()   
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class Network:
    # Use GPU if available.
    device = torch.device(torch.accelerator.current_device_index() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:

        self._model = torch.nn.Sequential(
            Permute(0, 3, 1, 2),
            torch.nn.Conv2d(env.observation_space.shape[2] * args.frame_stack, 32, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((2, 2)),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 2 * 2, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n)
        ).to(self.device)

        x = torch.zeros(1, 96//args.downsample, 96//args.downsample, 12, device=self.device)
        for layer in self._model:
            x = layer(x)
            print(layer.__class__.__name__, x.shape)


        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)

        self._loss = torch.nn.MSELoss()



    @npfl139.typed_torch_function(device, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, q_values: torch.Tensor) -> None:
        self._model.train()
        predictions = self._model(states)
        loss = self._loss(predictions, q_values)
        self._optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()


    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            return self._model(states)
        

    def copy_weights_from(self, other: "Network") -> None:
        self._model.load_state_dict(other._model.state_dict())


class CarRacingEnv:
    def __init__(self, env, args: argparse.Namespace):
        self._env = env
        self._frame_stack = args.frame_stack
        self._frames = collections.deque(maxlen=args.frame_stack)
        self._downsample = args.downsample

    def _process_obs(self, obs):
        if self._downsample > 1:
            obs = obs[::self._downsample, ::self._downsample, :]  # Downsample the observation
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._frames.append(self._process_obs(obs))
        return np.concatenate(self._frames, axis=2), reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        for _ in range(self._frame_stack):
            self._frames.append(self._process_obs(obs))
        return np.concatenate(self._frames, axis=2), info
    
class VecCarRacingEnv:
    def __init__(self, env, args: argparse.Namespace):
        self._env = env
        self._frame_stack = args.frame_stack
        self._frames = [collections.deque(maxlen=args.frame_stack) for _ in range(env.num_envs)]
        self._downsample = args.downsample

    def _process_obs(self, obs):
        if self._downsample > 1:
            obs = obs[::self._downsample, ::self._downsample, :]  # Downsample the observation
        return obs

    def step(self, actions):
        obs, rewards, terminateds, truncateds, infos = self._env.step(actions)
        for i in range(self._env.num_envs):
            self._frames[i].append(self._process_obs(obs[i]))
        stacked_obs = np.stack([np.concatenate(frames, axis=2) for frames in self._frames], axis=0)
        return stacked_obs, rewards, terminateds, truncateds, infos

    def reset(self, **kwargs):
        obs, infos = self._env.reset(**kwargs)
        for i in range(self._env.num_envs):
            for _ in range(self._frame_stack):
                self._frames[i].append(self._process_obs(obs[i]))
        stacked_obs = np.stack([np.concatenate(frames, axis=2) for frames in self._frames], axis=0)
        return stacked_obs, infos
    

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    

    # If you want, you can wrap even the `npfl139.EvaluationEnv` with additional wrappers, like
    #   env = gym.wrappers.ResizeObservation(env, (64, 64))
    # or
    #   env = gym.wrappers.GrayscaleObservation(env)
    # However, if you do that, you can no longer call just `env.reset(start_evaluation=True)`;
    # instead, you need to pass the `start_evaluation` to the inner environment using
    #   env.reset(options={"start_evaluation": True})

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # TODO: Load the agent
        network: Network = torch.load(args.agent_path)

        cr_env = CarRacingEnv(env, args)

        # Final evaluation
        while True:
            state, done = cr_env.reset(start_evaluation=True)[0], False
            while not done:
                # TODO: Choose a greedy action
                action = network.predict(state[np.newaxis])[0].argmax()
                state, reward, terminated, truncated, _ = cr_env.step(action)
                done = terminated or truncated
        return

    # TODO: Implement a suitable RL algorithm and train the agent.
    #
    # If you want to create N multiprocessing parallel environments, use
    #   vector_env = gym.make_vec("npfl139/CarRacingFS-v3", N, gym.VectorizeMode.ASYNC,
    #                             frame_skip=args.frame_skip, continuous=args.continuous)
    #   vector_env.reset(seed=args.seed)  # The individual environments get incremental seeds
    #
    # There are several Autoreset modes available, see https://farama.org/Vector-Autoreset-Mode.
    # To change the autoreset mode to SAME_STEP from the default NEXT_STEP, pass
    #   vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}
    # as an additional argument to the above `gym.make_vec`.
    vec_env = gym.make_vec("npfl139/CarRacingFS-v3", args.num_envs, gym.VectorizeMode.ASYNC,
                            frame_skip=args.frame_skip, continuous=args.continuous)
    vec_env.reset(seed=args.seed)
    vec_env = VecCarRacingEnv(vec_env, args)

    network = Network(env, args)
    target_network = Network(env, args)
    target_network.copy_weights_from(network)
    since_update = 0

    replay_buffer = npfl139.ReplayBuffer(max_length=args.replay_buffer_size)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon


    training = True

    states, _ = vec_env.reset(seed=args.seed)
    step = 0
    running_returns = np.zeros(vec_env._env.num_envs, dtype=np.float32)
    completed_returns = collections.deque(maxlen=100)
    completed_episodes = 0
    while training:
        if step % 1000 == 0:
            print(f"Epsilon: {epsilon:.6f}")
        step += 1

        q = network.predict(states)
        greedy_actions = q.argmax(axis=1)
        random_actions = np.random.randint(env.action_space.n, size=vec_env._env.num_envs)
        actions = np.where(np.random.rand(vec_env._env.num_envs) >= epsilon, greedy_actions, random_actions)

        next_states, rewards, terminated, truncated, _ = vec_env.step(actions)
        dones = terminated | truncated

        running_returns += rewards
        if np.any(dones):
            for i in np.where(dones)[0]:
                completed_returns.append(float(running_returns[i]))
                completed_episodes += 1
                running_returns[i] = 0.0

            if completed_episodes % 10 == 0:
                print(
                    f"Episodes: {completed_episodes}, Step: {step}, " +
                    f"mean return (last {len(completed_returns)}): {np.mean(completed_returns):.2f}",
                    flush=True,
                )

        for i in range(vec_env._env.num_envs):
            replay_buffer.append(Transition(states[i], actions[i], rewards[i], dones[i], next_states[i]))


        if len(replay_buffer) >= args.replay_start_size:
            batch = replay_buffer.sample(args.batch_size)

            current_q = network.predict(batch.state)
            next_q = target_network.predict(batch.next_state)

            target = current_q.copy()
            target[np.arange(args.batch_size), batch.action] = batch.reward + (1 - batch.done) * args.gamma * np.max(next_q, axis=1)

            network.train(batch.state, target)

            since_update += 1
            if since_update >= args.target_update_freq:
                print("Updating target network.")
                target_network.copy_weights_from(network)
                since_update = 0

        
        states = next_states

        if args.epsilon_final_at:
            epsilon = np.interp(step + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

        if step >= args.steps:
            training = False

    torch.save(network, args.agent_path)

        

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        gym.make("npfl139/CarRacingFS-v3", frame_skip=main_args.frame_skip, continuous=main_args.continuous),
        main_args.seed, main_args.render_each, evaluate_for=15, report_each=1)

    main(main_env, main_args)
