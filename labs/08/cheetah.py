#!/usr/bin/env python3
# f5419161-0138-4909-8252-ba9794a63e53
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
import argparse
import collections

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import npfl139
npfl139.require_version("2526.7")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="HalfCheetah-v5", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--hidden_size", default=[86,128], type=list, help="Hidden layers size")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("--sigma", default=0.2, type=float, help="Std for exploration noise")
parser.add_argument("--c", default = 0.4, type =float, help = "Clipping noise")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--max_iters", default=20000, type=int, help="Number of iterations.")
parser.add_argument("--eval_every", default=20, type=int, help="Eval frequency")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
parser.add_argument("--tau", default=0.005, type=float, help="Soft update parameter")
parser.add_argument("--eval_num", default=5, type=int, help="Number of episodes per evaluation")
parser.add_argument("--save_path", default="models/cheetah", type=str, help="Path to save the model")
parser.add_argument("--train_every",default=2, type=int, help="Number of episodes between policy updates")

# For these and any other arguments you add, ReCodEx will keep your default value.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def create_networks(obs:int, act:int, hidden:list[int], tanh:bool = False):
    modules = [nn.Linear(obs, hidden[0])]
    for i in range(len(hidden)-1):
        modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden[i], hidden[i+1]))
    modules.append(nn.ReLU())
    modules.append(nn.Linear(hidden[-1], act))
    if tanh:
        modules.append(nn.Tanh())
    return nn.Sequential(*modules).to(device)
    

class TD3Agent:
    def __init__(self, obs_dim:int, act_dim:int, args: argparse.Namespace) -> None:
        self.sigma = args.sigma
        self.c = args.c
        self.gamma = args.gamma
        self.tau = args.tau
        self.action_dim = act_dim

        self.a = create_networks(obs=obs_dim,act= act_dim,hidden= args.hidden_size, tanh = True)
        self.at = create_networks(obs=obs_dim,act= act_dim,hidden= args.hidden_size, tanh = True)
        self.at.load_state_dict(self.a.state_dict())
        self.at.requires_grad_(False)

        self.q1 = create_networks(obs=obs_dim + act_dim,act=1,hidden= args.hidden_size)
        self.q1t = create_networks(obs=obs_dim + act_dim,act=1,hidden= args.hidden_size)
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q1t.requires_grad_(False)
        
        self.q2 = create_networks(obs=obs_dim + act_dim,act=1,hidden= args.hidden_size)
        self.q2t = create_networks(obs=obs_dim + act_dim,act=1,hidden= args.hidden_size)
        self.q2t.load_state_dict(self.q2.state_dict())
        self.q2t.requires_grad_(False)

        self.critic_optim = torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()), lr = args.learning_rate)
        self.actor_optim = torch.optim.Adam(self.a.parameters(), lr = args.learning_rate)
    
    def load_actor(self, path:str):
        with open(path, "rb") as f:
            self.a.load_state_dict(torch.load(f,map_location=torch.device('cpu')))

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, state, add_noise: bool = True):
        with torch.no_grad():
            dist = self.a(state).cpu().numpy().flatten()
            if add_noise:
                dist = dist + np.clip(np.random.normal(0, self.sigma,dist.shape[0]),-self.c,self.c)
            return dist
    
    @npfl139.typed_torch_function(device, torch.float32, torch.float32, torch.float32, torch.bool, torch.float32, torch.bool)
    def train(self, s:torch.Tensor, a:torch.Tensor, r:torch.Tensor, d:torch.Tensor, next_s:torch.Tensor, fit_actor:bool=False):
        B = s.shape[0]
        self.critic_optim.zero_grad()
        anext = self.at(next_s).detach() + torch.clamp(torch.normal(0,self.sigma,size=(B,self.action_dim)),-self.c,self.c).to(device)
        t_inputs = torch.cat([next_s, anext], dim=1)
        c1 = self.q1t(t_inputs)
        c2 = self.q2t(t_inputs)
        c = torch.min(c1, c2)

        r = r.reshape(c.shape)
        y = r + self.gamma * c * (~d).float().reshape(c.shape)

        c_inputs = torch.cat([s, a], dim=1)
        c1_loss = torch.mean((self.q1(c_inputs)-y)**2)
        c2_loss = torch.mean((self.q2(c_inputs)-y)**2)
        critic_loss = c1_loss+c2_loss
        critic_loss.backward()
        self.critic_optim.step()

        if fit_actor:
            self.actor_optim.zero_grad()
            actions = self.a(s)
            actor_loss = - torch.mean(self.q1(torch.cat([s, actions], dim=1)))
            actor_loss.backward()
            self.actor_optim.step()

            # targets
            npfl139.update_params_by_ema(self.at, self.a, self.tau)
            npfl139.update_params_by_ema(self.q1t, self.q1, self.tau)
            npfl139.update_params_by_ema(self.q2t, self.q2, self.tau)
    def save(self, base_path:str):

        torch.save(self.a.state_dict(),f"{base_path}_actor.pth")
        torch.save(self.q1.state_dict(),f"{base_path}_critic1.pth")
        torch.save(self.q2.state_dict(),f"{base_path}_critic2.pth")
        torch.save(self.at.state_dict(),f"{base_path}_actor_target.pth")
        torch.save(self.q1t.state_dict(),f"{base_path}_critic1_target.pth")
        torch.save(self.q2t.state_dict(),f"{base_path}_critic2_target.pth")
        

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = TD3Agent(obs_dim,act_dim,args)
    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            # TODO: Predict an action using the greedy policy.
            action = agent.predict(state[np.newaxis])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward

        return rewards
    
    

    # Evaluation in ReCodEx
    if args.recodex:
        # TODO: Load a pretrain model and perform evaluation.
        agent = TD3Agent(obs_dim,act_dim,args)
        agent.load_actor("cheetah_actor.pth")
        while True:
            evaluate_episode(start_evaluation=True)
        return

    # TODO: Perform training
    buffer = npfl139.ReplayBuffer(60000)
    c = 0
    max_reward = -np.inf
    for i in range(1,args.max_iters+1):
        state = env.reset()[0]
        done = False
        while not done:
            action = agent.predict(state[np.newaxis])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.append(Transition(state, action, reward, terminated, next_state))
            state = next_state
            c+=1
            batch = buffer.sample(args.batch_size)
            agent.train(batch.state, batch.action, batch.reward, batch.done, batch.next_state,
                        c%args.train_every==0)
        
        if i % args.eval_every == 0:
            reward = np.mean([evaluate_episode() for _ in range(args.eval_num)])
            print(f"Episode {i}, Rewards: {reward}")
            if reward > max_reward:
                max_reward = reward
                agent.save(args.save_path)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)

    main(main_env, main_args)
