#!/usr/bin/env python3
# f5419161-0138-4909-8252-ba9794a63e53
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
import argparse

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.v2  as v2
import npfl139
npfl139.require_version("2526.6")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=16, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--episodes", default=5410, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=2**(-9), type=float, help="Learning rate.")
parser.add_argument("--value_learning_rate", default=2**(-6), type=float, help="Learning rate.")
parser.add_argument("--grayscale", default=True, type=bool, help="Grayscale the input.")
parser.add_argument("--filters_first", default=9, type=int, help="Number of filters in the first convolutional layer.")
parser.add_argument("--num_blocks", default=2, type=int, help="Number of convolutional blocks.")
parser.add_argument("--block_size", default=1, type=int, help="Number of convolutional layers in each block.")
parser.add_argument("--dropout_last", default=0.4, type=float, help="Dropout rate for the last fully connected layer.")

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = v2.Compose([
    v2.ToTensor(),
    v2.Resize((64,64)),
])
C = 3
H = 64
OUT_CLASSES = 2

def preprocess(np_image):
    return transform(np_image)


class CNN(torch.nn.Module):

    def _create_block(self, in_channels, channels, block_size):
        modules = torch.nn.ModuleList()
        modules.append(torch.nn.Conv2d(in_channels, channels, kernel_size=1, padding="same"))
        for i in range(block_size):
            modules.append(torch.nn.Conv2d(channels, channels, kernel_size=3, padding="same"))
            modules.append(torch.nn.ReLU())
            modules.append(torch.nn.BatchNorm2d(channels))
        modules.append(torch.nn.MaxPool2d(kernel_size=2))
        return modules
    
    def _create_conv_network(self,args):
        modules = torch.nn.ModuleList()
        for i in range(args.num_blocks):
            channels = args.filters_first * (2**i)
            in_channels = int(channels / 2) if i != 0 else C
            modules.append(self._create_block(in_channels, channels, args.block_size))
        return modules

    def __init__(self, args, device):
        super().__init__()
        self.conv_blocks = self._create_conv_network(args)
        self.to(device)
    
    def forward(self, x):
        for block in self.conv_blocks:
            x = block[0](x)
            f = x
            for lay in block[1:-1]:
                f = lay(f)
            h = f + x
            x = block[-1](h)
        x = torch.flatten(x, 1)
        return x

class Agent:
    device = torch.device("cpu")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        def cnn_layers():
            return [
                torch.nn.Conv2d(3, 12, 3, 1, "same"),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(12, 24, 3, 1, "same"),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(24, 48, 3, 1, "same"),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Flatten(),
            ]

        self._policy = torch.nn.Sequential(
            *cnn_layers(),
            torch.nn.LazyLinear(args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n),
        ).to(self.device)

        self._value = torch.nn.Sequential(
            *cnn_layers(),
            torch.nn.LazyLinear(args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1),
        ).to(self.device)

        self._optimizer_policy = torch.optim.Adam(self._policy.parameters(), args.learning_rate)
        self._optimizer_value = torch.optim.Adam(self._value.parameters(), args.value_learning_rate)
        self.gamma = args.gamma

        self._loss = torch.nn.CrossEntropyLoss(reduction="none")
        self._value_loss = torch.nn.MSELoss(reduction="none")

    def save_weights(self, path):
        torch.save(self._policy.state_dict(), path + "_policy.pt")
        torch.save(self._value.state_dict(), path + "_value.pt")

    def load_weights(self, path):
        self._policy.load_state_dict(torch.load(path + "_policy.pt"))
        self._value.load_state_dict(torch.load(path + "_value.pt"))

    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Define the training method.
        #
        # You should:
        # - compute the predicted baseline using the baseline model
        # - train the policy model, using `returns - predicted_baseline` as
        #   advantage estimate
        # - train the baseline model to predict `returns`

        self._optimizer_value.zero_grad()
        value_pred = self._value(states).squeeze(-1)
        loss_value = self._value_loss(value_pred, returns).mean()
        loss_value.backward()
        self._optimizer_value.step()

        self._optimizer_policy.zero_grad()
        delta = (returns - value_pred.detach())
        
        # Normalize advantages to prevent gradient explosion as episode lengths increase
        if delta.shape[0] > 1:
            delta = (delta - delta.mean()) / (delta.std() + 1e-8)
            
        loss_policy = (delta * self._loss(self._policy(states), actions)).mean()
        loss_policy.backward()
        self._optimizer_policy.step()
        

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return torch.nn.functional.softmax(self._policy(states[None]), dim=-1).cpu().numpy()


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # TODO: Load the agent
        agent = Agent(env,args)
        agent.load_weights("./agent")

        # Final evaluation
        while True:
            state, done = env.reset(options={"start_evaluation": True})[0], False
            while not done:
                # TODO: Choose a greedy action.
                action = np.argmax(agent.predict(preprocess(state)))
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        return

    # TODO: Perform training
    agent = Agent(env,args)
    gamma = args.gamma

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset()[0], False
            while not done:
                # TODO(reinforce): Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `agent.predict` and current `state`.
                state = preprocess(state)
                action = np.random.choice(env.action_space.n, size=1, p=agent.predict(state).flatten())
                action = action[0]

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            returns = np.zeros(len(rewards), dtype=np.float32)
            returns[-1] = rewards[-1]
            for t in range(len(rewards)-2,-1,-1):
                returns[t] = rewards[t] + gamma * returns[t+1]

            # TODO: Add states, actions and returns to the training batch
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)

        # TODO: Train using the generated batch.
        agent.train(torch.stack(batch_states), batch_actions, batch_returns)
        agent.save_weights("./agent")

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO(reinforce): Choose a greedy action.
            state = preprocess(state)
            action = np.argmax(agent.predict(state))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make("npfl139/CartPolePixels-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)
