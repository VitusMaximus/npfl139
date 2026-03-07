#!/usr/bin/env python3
# f5419161-0138-4909-8252-ba9794a63e53
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
import argparse

import gymnasium as gym
import numpy as np

import npfl139
npfl139.require_version("2526.3")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=True, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_decay", default=0.9999, type=float, help="Exploration decay after each episode.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")


state_bounds = [
    (-1.0, 1.0, 10),  # x
    (-0.2, 1.2, 10),  # y
    (-1.5, 1.5, 20),  # vx
    (-1.5, 0.5, 20),  # vy
    (-0.5, 0.5, 20),  # angle
    (-1.0, 1.0, 20),  # v-angle
    (0, 1, 2),        # touch L
    (0, 1, 2)         # touch R
]

bins =  [np.linspace(low, high, num=num_bins) for low, high, num_bins in state_bounds]


def discretize(state):
    return tuple(np.digitize(s, b) for s, b in zip(state, bins))




def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # TODO: Load the agent
        Q: np.ndarray = np.load("lunar_lander_q.npy")
        # Final evaluation
        while True:
            state, done = env.reset(start_evaluation=True)[0], False
            while not done:
                # TODO: Choose a greedy action
                action = Q[state].argmax()
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    # TODO: Implement a suitable RL algorithm and train the agent.

    epsilon = args.epsilon
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    rand = np.random.RandomState(args.seed)

    cnt = 0
    training = True
    while training:
        # To generate an expert episode, you can use the following:
        #   episode = env.expert_episode()

        # Otherwise, you can generate a training episode the usual way:

        epsilon = epsilon * args.epsilon_decay
        if epsilon < 0.0001:
            training = False
        if cnt % 100 == 0:
            print("Epsilon: ", epsilon)

        state, done = env.reset()[0], False



        while not done:
            if rand.rand() < epsilon:
                action = env.action_space.sample()
            else:            
                action = Q[state].argmax()

            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            Q[state, action] += args.alpha * (reward + args.gamma * Q[new_state].max() - Q[state, action])

            state = new_state

        cnt += 1

    np.save("lunar_lander_q.npy", Q)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteLunarLanderWrapper(gym.make("LunarLander-v3")), main_args.seed, main_args.render_each)

    main(main_env, main_args)
