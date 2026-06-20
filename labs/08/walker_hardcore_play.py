#!/usr/bin/env python3

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import npfl139


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="BipedalWalker-v3", type=str, help="Environment.")
parser.add_argument("--model_path", default="SAC_walker.pt", type=str, help="Path to the saved actor.")
parser.add_argument("--episodes", default=5, type=int, help="Number of episodes to run.")
parser.add_argument("--render_each", default=1, type=int, help="Render every N-th episode.")
parser.add_argument("--record_dir", default="labs/08", type=str, help="Directory for recorded GIFs.")
parser.add_argument("--fps", default=50, type=int, help="Frames per second for recorded video.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")


class Actor(torch.nn.Module):
    def __init__(self, env: npfl139.EvaluationEnv, hidden_layer_size: int) -> None:
        super().__init__()

        self._trunk = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_size, hidden_layer_size),
            torch.nn.ReLU(),
        )
        self._mus = torch.nn.Linear(hidden_layer_size, env.action_space.shape[0])
        self._sds = torch.nn.Linear(hidden_layer_size, env.action_space.shape[0])

        self._log_alpha = torch.nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))

        self.register_buffer("action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2))
        self.register_buffer("action_offset", torch.tensor((env.action_space.high + env.action_space.low) / 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self._trunk(inputs)
        mus = self._mus(hidden)
        return torch.tanh(mus) * self.action_scale + self.action_offset


class Agent:
    device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self._actor = Actor(env, args.hidden_layer_size).to(self.device)

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        self._actor.eval()
        with torch.inference_mode():
            return self._actor(states).cpu()

    def load_actor(self, path: str) -> None:
        self._actor.load_state_dict(torch.load(path, map_location=self.device))


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()

    agent = Agent(env, args)
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parents[2] / model_path
    agent.load_actor(str(model_path))

    returns = []
    for episode in range(1, args.episodes + 1):
        logging = args.render_each > 0 and episode % args.render_each == 0
        state = env.reset(options={"logging": logging})[0]
        done = False
        total_return = 0.0

        while not done:
            action = agent.predict_actions(state[None])[0]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_return += reward

        returns.append(total_return)
        print(f"Episode {episode}: return {total_return:.2f}")

    print(f"Mean return over {args.episodes} episodes: {np.mean(returns):.2f}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    if main_args.record_dir:
        record_dir = Path(main_args.record_dir)
        if not record_dir.is_absolute():
            record_dir = Path(__file__).resolve().parents[2] / record_dir
        record_dir.mkdir(parents=True, exist_ok=True)

        main_env = gym.make(main_args.env, render_mode="rgb_array")

        npfl139.startup(main_args.seed, main_args.threads)
        npfl139.global_keras_initializers()

        agent = Agent(main_env, main_args)
        model_path = Path(main_args.model_path)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[2] / model_path
        agent.load_actor(str(model_path))

        returns = []
        for episode in range(1, main_args.episodes + 1):
            state = main_env.reset(seed=main_args.seed + episode)[0]
            done = False
            total_return = 0.0
            # Open a writer for this episode (mp4)
            output_path = record_dir / f"walker-episode-{episode:03d}.mp4"
            writer = imageio.get_writer(str(output_path), fps=main_args.fps)

            try:
                # initial frame
                frame = main_env.render()
                if frame is not None:
                    writer.append_data(frame)

                while not done:
                    action = agent.predict_actions(state[None])[0]
                    state, reward, terminated, truncated, _ = main_env.step(action)
                    done = terminated or truncated
                    total_return += reward
                    frame = main_env.render()
                    if frame is not None:
                        writer.append_data(frame)

            finally:
                writer.close()
                returns.append(total_return)
                print(f"Episode {episode}: return {total_return:.2f}")

        print(f"Mean return over {main_args.episodes} episodes: {np.mean(returns):.2f}")
        main_env.close()
    else:
        main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)
        main(main_env, main_args)