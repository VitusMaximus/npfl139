#!/usr/bin/env python3
# f5419161-0138-4909-8252-ba9794a63e53
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
import argparse
import collections

import numpy as np
import torch
import torch.nn as nn

import npfl139
npfl139.require_version("2526.11.2")
from npfl139.board_games import AZQuiz

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.3, type=float, help="MCTS root Dirichlet alpha")
parser.add_argument("--batch_size", default=512, type=int, help="Number of game positions to train on.")
parser.add_argument("--epsilon", default=0.25, type=float, help="MCTS exploration epsilon in root")
parser.add_argument("--evaluate_each", default=10, type=int, help="Evaluate each number of iterations.")
parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="az_quiz_og2.pt", type=str, help="Model path")
parser.add_argument("--num_simulations", default=100, type=int, help="Number of simulations in one MCTS.")
parser.add_argument("--replay_buffer_length", default=100_000, type=int, help="Replay buffer max length.")
parser.add_argument("--sampling_moves", default=8, type=int, help="Sampling moves.")
parser.add_argument("--show_sim_games", default=False, action="store_true", help="Show simulated games.")
parser.add_argument("--sim_games", default=1, type=int, help="Simulated games to generate in every iteration.")
parser.add_argument("--train_for", default=1, type=int, help="Update steps in every iteration.")
parser.add_argument("--c",default=1.2,type=float,help="MCTS exploration constant")
parser.add_argument("--load", default="az_quiz_og2.pt", type=str, help="Path to load model weights from before training.")
parser.add_argument("--filters", default=32, type=int, help="Number of conv filters in the residual trunk.")
parser.add_argument("--blocks", default=6, type=int, help="Number of residual blocks in the trunk.")
parser.add_argument("--max_iterations", default=2000, type=int, help="Maximum training iterations.")
parser.add_argument("--target_score", default=0.97, type=float, help="Stop when eval score reaches this (consecutively).")
parser.add_argument("--target_streak", default=3, type=int, help="Consecutive evals above target to stop.")

#########
# Agent #
#########

class Model(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._preprocessor = nn.Sequential(
            nn.Conv2d(AZQuiz.C, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self._policy_head = nn.Sequential(
            nn.Conv2d(20, 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * AZQuiz.N * AZQuiz.N, AZQuiz.ACTIONS),
        )
        self._value_head = nn.Sequential(
            nn.Conv2d(20, 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * AZQuiz.N * AZQuiz.N, 1),
            nn.Tanh(),
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._preprocessor(x)
        return self._policy_head(x), self._value_head(x)

class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, args: argparse.Namespace):
        # TODO: Define an agent network in `self._model`.
        #
        # A possible architecture known to work consists of
        # - 5 convolutional layers with 3x3 kernel and 15-20 filters,
        # - a policy head, which first uses 3x3 convolution to reduce the number of channels
        #   to 2, flattens the representation, and finally uses a dense layer to produce
        #   the policy logits,
        # - a value head, which again uses 3x3 convolution to reduce the number of channels
        #   to 2, flattens, and produces expected return using an output dense layer with
        #   `tanh` activation.
        self._model = Model(args).to(self.device)
        self._optim = torch.optim.AdamW(self._model.parameters(), lr=args.learning_rate,weight_decay=0.001)
        self._loss = nn.CrossEntropyLoss(reduction="none")

    @classmethod
    def load(cls, path: str, args: argparse.Namespace) -> "Agent":
        # A static method returning a new Agent loaded from the given path.
        agent = Agent(args)
        agent._model.load_state_dict(torch.load(path, map_location=agent.device))
        return agent

    def save(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)

    @npfl139.typed_torch_function(device, torch.float32, torch.float32, torch.float32)
    def train(self, boards: torch.Tensor, target_policies: torch.Tensor, target_values: torch.Tensor) -> None:
        # TODO: Train the model based on given boards, target policies and target values.
        # Note that the model returns logits.
        self._model.train()
        self._optim.zero_grad()
        logits, values = self._model(boards)
        policy_loss = self._loss(logits, target_policies)
        val_loss = (target_values - values.squeeze(-1)) ** 2
        loss = (policy_loss + val_loss).mean()
        loss.backward()
        self._optim.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, boards: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Return the predicted policy and the value function. Because the model
        # returns logits, you should apply softmax to return policy probabilities.
        self._model.eval()
        with torch.no_grad():
            logits, values = self._model(boards)
            return nn.functional.softmax(logits,-1).cpu().numpy(), values.cpu().numpy()

    def board_features(self, game: AZQuiz) -> np.ndarray:
        # TODO: Generate the boards from the current game.
        #
        # The `game.board_features` returns a board representation, but you also
        # need to somehow indicate who is the current player. You can either
        # - change the game so that the current player is always the same one
        #   (i.e., always 0 or always 1; `swap_players` option of `AZQuiz.clone`
        #   method might come handy);
        # - indicate the current player by adding channels to the representation.
        if game.to_play != 0:
            game = game.clone(swap_players=True)
        return np.transpose(game.board_features, (2, 0, 1))[np.newaxis]


########
# MCTS #
########
class MCTNode:
    INF = 1e8

    def __init__(self, prior: float | None):
        self.prior = prior  # Prior probability from the agent.
        self.game = None    # If the node is evaluated, the corresponding game instance.
        self.children = {}  # If the node is evaluated, mapping of valid actions to the child `MCTNode`s.
        self.visit_count = 0
        self.total_value = 0

    def value(self) -> float:
        # TODO: Return the value of the current node, handling the
        # case when `self.visit_count` is 0.
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


    def is_evaluated(self) -> bool:
        # A node is evaluated if it has non-zero `self.visit_count`.
        # In such case `self.game` is not None.
        return self.visit_count > 0

    def evaluate(self, game: AZQuiz, agent: Agent) -> None:
        # Each node can be evaluated at most once
        assert self.game is None
        self.game = game

        # TODO: Compute the value of the current game.
        # - If the game has ended, compute the value directly
        # - Otherwise, use the given `agent` to evaluate the current
        #   game. Then, for all valid actions, populate `self.children` with
        #   new `MCTNodes` with the priors from the policy predicted
        #   by the network.
        outcome = game.outcome()
        if outcome is not None:
            value = float(outcome.value - 2)
        else:
            policy, values = agent.predict(agent.board_features(game))
            value = float(values[0, 0])
            for action in game.valid_actions():
                child = MCTNode(float(policy[0, action]))
                self.children[action] = child

        self.total_value = value
        self.visit_count = 1

    def add_exploration_noise(self, epsilon: float, alpha: float) -> None:
        # TODO: Update the children priors by exploration noise
        # Dirichlet(alpha), so that the resulting priors are
        #   epsilon * Dirichlet(alpha) + (1 - epsilon) * original_prior
        noise = np.random.dirichlet([alpha] * len(self.children))
        for (_, child), n in zip(self.children.items(), noise):
            child.prior = epsilon * float(n) + (1 - epsilon) * child.prior

    def select_child(self) -> tuple[int, "MCTNode"]:
        # Select a child according to the PUCT formula.
        def ucb_score(child: "MCTNode"):
            # TODO: For a given child, compute the UCB score as
            #   Q(s, a) + C(s) * P(s, a) * (sqrt(N(s)) / (N(s, a) + 1)),
            # where:
            # - Q(s, a) is the estimated value of the action stored in the
            #   `child` node. However, the value in the `child` node is estimated
            #   from the view of the player playing in the `child` node, which
            #   is usually the other player than the one playing in `self`,
            #   and in that case the estimated value must be "inverted";
            # - C(s) in AlphaZero is defined as
            #     log((1 + N(s) + 19652) / 19652) + 1.25
            #   Personally I used 1965.2 to account for shorter games, but I do not
            #   think it makes any difference;
            # - P(s, a) is the prior computed by the agent;
            # - N(s) is the number of visits of state `s`;
            # - N(s, a) is the number of visits of action `a` in state `s`.
            C = np.log((1 + self.visit_count + 19652) / 19652) + 1.25
            U = C * child.prior * (np.sqrt(self.visit_count) / (child.visit_count + 1))
            Q = -child.value() if child.visit_count > 0 else 0.0
            return Q + U

        # TODO: Return the (action, child) pair with the highest `ucb_score`.
        return max(self.children.items(), key=lambda x: ucb_score(x[1]))


def mcts(game: AZQuiz, agent: Agent, args: argparse.Namespace, explore: bool) -> np.ndarray:
    # Run the MCTS search and return the policy proportional to the visit counts,
    # optionally including exploration noise to the root children.
    root = MCTNode(None)
    root.evaluate(game, agent)
    if explore:
        root.add_exploration_noise(args.epsilon, args.alpha)

    # Perform the `args.num_simulations` number of MCTS simulations.
    for _ in range(args.num_simulations):
        # TODO: Starting in the root node, traverse the tree using `select_child()`,
        # until a `node` without `children` is found.
        node = root
        game_clone = game.clone()
        path = [root]
        while len(node.children) > 0:
            action, node = node.select_child()
            game_clone.move(action)
            path.append(node)

        # If the node has not been evaluated, evaluate it.
        if not node.is_evaluated():
            # TODO: Evaluate the `node` using the `evaluate` method. To that
            # end, create a suitable `AZQuiz` instance for this node by cloning
            # the `game` from its parent and performing a suitable action.
            node.evaluate(game_clone, agent)
        else:
            # TODO: If the node has been evaluated but has no children, the
            # game ends in this node. Update it appropriately.
            node.total_value += node.value()
            node.visit_count += 1

        # Get the value of the node.
        value = node.value()

        # TODO: For all parents of the `node`, update their value estimate,
        # i.e., the `visit_count` and `total_value`.
        v = value
        for parent in reversed(path[:-1]):
            v = -v
            parent.visit_count += 1
            parent.total_value += v

    # TODO: Compute a policy proportional to visit counts of the root children.
    # Note that invalid actions are not the children of the root, but the
    # policy should still return 0 for them.
    policy = np.zeros(AZQuiz.ACTIONS, dtype=np.float32)
    for action, child in root.children.items():
        policy[action] = child.visit_count
    policy /= policy.sum()
    return policy


############
# Training #
############
ReplayBufferEntry = collections.namedtuple("ReplayBufferEntry", ["board", "policy", "outcome"])


def sim_game(agent: Agent, args: argparse.Namespace) -> list[ReplayBufferEntry]:
    # Simulate a game, return a list of `ReplayBufferEntry`s.
    game = AZQuiz()
    states = []
    while game.outcome() is None:
        # TODO: Run the `mcts` with exploration.
        policy = mcts(game, agent, args, explore=True)

        # TODO: Select an action, either by sampling from the policy or greedily,
        # according to the `args.sampling_moves`.
        if len(states) < args.sampling_moves:
            action = int(np.random.choice(len(policy), p=policy))
        else:
            action = int(np.argmax(policy))

        states.append((agent.board_features(game)[0].astype(np.float32),
                       policy.astype(np.float32),
                       game.to_play))
        game.move(action)

    # TODO: Return all encountered game states, each consisting of
    # - the board (probably via `agent.board_features`),
    # - the policy obtained by MCTS,
    # - the outcome based on the outcome of the whole game.
    return [
        ReplayBufferEntry(board, policy, np.float32(game.outcome(to_play).value - 2))
        for board, policy, to_play in states
    ]


def train(args: argparse.Namespace, agent: Agent | None = None) -> Agent:
    # Perform training, optionally continuing from a pre-built `agent`.
    if agent is None:
        agent = Agent(args)
    replay_buffer = npfl139.ReplayBuffer(max_length=args.replay_buffer_length)

    iteration = 0
    training = True
    best_score = -1.0
    streak = 0
    while training:
        iteration += 1

        # Generate simulated games
        for _ in range(args.sim_games):
            game = sim_game(agent, args)
            replay_buffer.extend(game)

            # If required, show the generated game, as 8 very long lines showing
            # all encountered boards, each field showing as
            # - `XX` for the fields belonging to player 0,
            # - `..` for the fields belonging to player 1,
            # - percentage of visit counts for valid actions.
            if args.show_sim_games:
                log = [[] for _ in range(8)]
                for i, (board, policy, outcome) in enumerate(game):
                    log[0].append(f"Move {i}, result {outcome}".center(28))
                    action = 0
                    for row in range(7):
                        log[1 + row].append("  " * (6 - row))
                        for col in range(row + 1):
                            log[1 + row].append(
                                " XX " if board[row, col, 0] else
                                " .. " if board[row, col, 1] else
                                f"{policy[action] * 100:>3.0f} ")
                            action += 1
                        log[1 + row].append("  " * (6 - row))
                print(*["".join(line) for line in log], sep="\n")

        # Train
        for _ in range(args.train_for):
            # TODO: Perform training by sampling an `args.batch_size` of positions
            # from the `replay_buffer` and running `agent.train` on them.
            if len(replay_buffer) < args.batch_size:
                break
            batch = replay_buffer.sample(args.batch_size)
            agent.train(batch.board, batch.policy, batch.outcome)

        # Evaluate
        if iteration % args.evaluate_each == 0:
            # Pure-policy evaluation against both the simple and the stronger
            # fork heuristics. We save on the combined score so checkpoint
            # selection isn't dominated by 56-game noise against one opponent.
            eval_player = Player(agent, argparse.Namespace(num_simulations=0))
            score_simple = npfl139.board_games.evaluate(
                AZQuiz, [eval_player,
                         AZQuiz.player_from_name("simple_heuristic")(seed=main_args.seed)],
                games=56, first_chosen=False, render=False, verbose=False,
            )
            score_fork = npfl139.board_games.evaluate(
                AZQuiz, [eval_player,
                         AZQuiz.player_from_name("fork_heuristic")(seed=main_args.seed)],
                games=56, first_chosen=False, render=False, verbose=False,
            )
            score = 0.5 * (score_simple + score_fork)
            print(f"Evaluation after iteration {iteration}: "
                  f"simple {100 * score_simple:.1f}%, fork {100 * score_fork:.1f}%, "
                  f"combined {100 * score:.1f}%", flush=True)

            if score > best_score:
                best_score = score
                agent.save(args.model_path)

            if score_simple >= args.target_score and score_fork >= args.target_score:
                streak += 1
                if streak >= args.target_streak:
                    training = False
            else:
                streak = 0

        if iteration >= args.max_iterations:
            training = False

    agent.save(args.model_path)
    return agent


#############################
# BoardGamePlayer interface #
#############################
class Player(npfl139.board_games.BoardGamePlayer[AZQuiz]):
    def __init__(self, agent: Agent, args: argparse.Namespace):
        self.agent = agent
        self.args = args

    def play(self, game: AZQuiz) -> int:
        # Predict a best possible action.
        if self.args.num_simulations == 0:
            # TODO: If no simulations should be performed, use directly
            # the policy predicted by the agent on the current game board.
            policy = self.agent.predict(self.agent.board_features(game))[0][0]
        else:
            # TODO: Otherwise run the `mcts` without exploration and
            # utilize the policy returned by it.
            policy = mcts(game, self.agent, self.args, explore=False)

        # Now select a valid action with the largest probability.
        return max(game.valid_actions(), key=lambda action: policy[action])


########
# Main #
########
def main(args: argparse.Namespace) -> Player:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    if args.recodex:
        # Load the trained agent
        agent = Agent.load(args.model_path, args)
    else:
        # Optionally warm-start training from an existing checkpoint
        if args.load is not None:
            print(f"Loading weights from {args.load} to continue training.", flush=True)
            agent = Agent.load(args.load, args)
        else:
            agent = Agent(args)
        agent = train(args, agent)

    return Player(agent, args)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    player = main(main_args)

    # Run an evaluation versus the simple heuristic with the same parameters as in ReCodEx.
    npfl139.board_games.evaluate(
        AZQuiz, [player, AZQuiz.player_from_name("simple_heuristic")(seed=main_args.seed)],
        games=56, first_chosen=False, render=False, verbose=True,
    )
