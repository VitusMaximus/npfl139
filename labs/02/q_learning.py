#!/usr/bin/env python3
# f5419161-0138-4909-8252-ba9794a63e53
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
import argparse

import gymnasium as gym
import numpy as np

import npfl139
npfl139.require_version("2526.2")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.2, type=float, help="Learning rate.")
parser.add_argument("--eps_init", default=0.5, type=float, help="Exploration factor.")
parser.add_argument("--eps_min", default=0.01, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--episodes",default=12000,type=int)


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)

    # TODO: Variable creation and initialization
    Q = np.zeros((env.observation_space.n, 2))
    actions = [0,1]
    eps = args.eps_init
    eps_end = args.eps_min
    eps_step = (eps-eps_end)/args.episodes
    alp = args.alpha
    gam = args.gamma

    training = True
    for _ in range(args.episodes):
        # Perform episode
        state, done = env.reset()[0], False
        while not done:
            # TODO: Perform an action.
            action = np.argmax(Q[state,:]) if np.random.random() > eps else np.random.choice(actions)
            if action == 1:
                next_state, reward, terminated, truncated, _ = env.step(action+1)
            else:
                next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # TODO: Update the action-value estimates
            Q[state,action] += alp *(reward+gam*Q[next_state,:].max()-Q[state,action]) 
            state = next_state
            
        eps = max(eps_end, eps - eps_step)

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose a greedy action
            action = np.argmax(Q[state])
            if action == 1:
                action +=1 
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteMountainCarWrapper(gym.make("npfl139/MountainCar1000-v0")),
        main_args.seed, main_args.render_each,
    )

    main(main_env, main_args)
