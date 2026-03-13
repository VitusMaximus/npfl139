#!/usr/bin/env python3
# f5419161-0138-4909-8252-ba9794a63e53
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
import argparse

import gymnasium as gym
import numpy as np

import npfl139
npfl139.require_version("2526.4")
parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.3, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.000001, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=6000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
parser.add_argument("--n", default=8, type=int, help="Number of steps for n-step SARSA.")
parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.")

def mk_tile_state(vec:np.ndarray, indices:np.ndarray):
    vec *= 0
    vec[indices] = 1

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)

    # Implement Q-learning RL algorithm, using linear approximation.
    W = np.zeros([env.observation_space.nvec[-1], 2])
    epsilon = args.epsilon
    alpha = args.alpha / args.tiles
    n = args.n
    training = True

    while training:
        state, _ = env.reset()
        tile_states = [np.zeros(env.observation_space.nvec[-1])]
        mk_tile_state(tile_states[0], state)
        actions = [(tile_states[0] @ W).argmax() if np.random.random() > epsilon else np.random.randint(2)]
        rewards = [0.0]  # rewards[0] unused; rewards[i] = reward received after step i-1

        T = float('inf')
        t = 0
        while True:
            if t < T:
                next_state, reward, terminated, truncated, _ = env.step(actions[t] if actions[t] == 0 else actions[t] + 1)
                rewards.append(reward)
                if terminated or truncated:
                    T = t + 1
                    tile_states.append(None)
                    actions.append(None)
                else:
                    s_next = np.zeros(env.observation_space.nvec[-1])
                    mk_tile_state(s_next, next_state)
                    tile_states.append(s_next)
                    actions.append((s_next @ W).argmax() if np.random.random() > epsilon else np.random.randint(2))

            tau = t - n + 1
            if tau >= 0:
                G = sum(args.gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T) + 1))
                if tau + n < T:
                    G += args.gamma ** n * (tile_states[tau + n] @ W)[actions[tau + n]]
                s_tau = tile_states[tau]
                a_tau = actions[tau]
                W[:, a_tau] += alpha * (G - (s_tau @ W)[a_tau]) * s_tau

            if tau == T - 1:
                break
            t += 1

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])
            training = (env.episode != args.epsilon_final_at)

    # Final evaluation
    tile_state = np.zeros(env.observation_space.nvec[-1])
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            mk_tile_state(tile_state, state)
            action = (tile_state @ W).argmax()
            state, reward, terminated, truncated, _ = env.step(action if action == 0 else action + 1)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteMountainCarWrapper(gym.make("npfl139/MountainCar1000-v0"), tiles=main_args.tiles),
        main_args.seed, main_args.render_each,
    )

    main(main_env, main_args)
