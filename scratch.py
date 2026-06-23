import ale_py
import gymnasium as gym
import numpy as np
gym.register_envs(ale_py)

env = ale_py.AtariVectorEnv(
    game="pong",
    num_envs=8,
    frameskip=4, stack_num=4, grayscale=True,
    img_height=84, img_width=84,
    use_fire_reset=False, reward_clipping=False, repeat_action_probability=0.25,
    autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
)
states1, _ = env.reset()
actions = np.zeros(8, dtype=np.int64)
states2, _, _, _, _ = env.step(actions)

print("Same object?", states1 is states2)
print("Share memory?", np.shares_memory(states1, states2))
