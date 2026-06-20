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
from npfl139.board_games import BoardGame
import board_game_cpp

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=32, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.3, type=float, help="MCTS root Dirichlet alpha")
parser.add_argument("--batch_size", default=512, type=int, help="Number of game positions to train on.")
parser.add_argument("--epsilon", default=0.25, type=float, help="MCTS exploration epsilon in root")
parser.add_argument("--evaluate_against", default="simple_heuristic", type=str, help="Player to evaluate against")
parser.add_argument("--evaluate_each", default=10, type=int, help="Evaluate each number of iterations.")
parser.add_argument("--game", default="pisqorky", type=str, help="Board game to play")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="pisqorky.pt", type=str, help="Model path")
parser.add_argument("--load_path", default="pisqorky.pt", type=str, help="If set (and not --recodex), warm-start training from this checkpoint.")
parser.add_argument("--num_simulations", default=800, type=int, help="Number of simulations in one MCTS.")
parser.add_argument("--replay_buffer_length", default=100_000, type=int, help="Replay buffer max length.")
parser.add_argument("--sampling_moves", default=8, type=int, help="Sampling moves.")
parser.add_argument("--show_sim_games", default=False, action="store_true", help="Show simulated games.")
parser.add_argument("--sim_games", default=16, type=int, help="Simulated games to generate in every iteration.")
parser.add_argument("--train_for", default=8, type=int, help="Update steps in every iteration.")
parser.add_argument("--augment", default=True, action=argparse.BooleanOptionalAction,help="Use symmetries")
parser.add_argument("--filters", default=48, type=int, help="Conv filters.")
parser.add_argument("--blocks", default=6, type=int, help="Residual blocks.")
parser.add_argument("--max_iterations", default=4500, type=int, help="Maximum training iterations.")


#########
# Agent #
#########
class ResidualBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = nn.functional.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return nn.functional.relu(x + y)


class Model(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        G = args.Game
        f = args.filters
        self._stem = nn.Sequential(
            nn.Conv2d(G.C, f, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(),
        )
        self._blocks = nn.Sequential(*[ResidualBlock(f) for _ in range(args.blocks)])
        self._policy_head = nn.Sequential(
            nn.Conv2d(f, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * G.N * G.N, G.ACTIONS),
        )
        self._value_head = nn.Sequential(
            nn.Conv2d(f, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(G.N * G.N, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._blocks(self._stem(x))
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
        self._optim = torch.optim.AdamW(self._model.parameters(), lr=args.learning_rate, weight_decay=0.001)
        self._loss = nn.CrossEntropyLoss(reduction="none")
        self._game = args.game

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
            return nn.functional.softmax(logits, -1).cpu().numpy(), values.cpu().numpy()

    def board_features(self, game: BoardGame) -> np.ndarray:
        # Canonical "side-to-move" view: when player 1 is on move, swap the two
        # player-stone channels so the network always sees its own stones in the
        # same channel. Matches the C++ board_features in az_quiz.h / pisqorky.h.
        bf = np.asarray(game.board_features)
        if game.to_play == 1:
            if self._game == "pisqorky":
                bf = bf[..., [0, 2, 1]]
            else:  # az_quiz / az_quiz_randomized
                bf = bf[..., [1, 0, 2, 3]]
        return np.transpose(bf, (2, 0, 1))[np.newaxis]

    def cpp_evaluator(self):
        def evaluate(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            boards = np.transpose(boards, (0, 3, 1, 2)).astype(np.float32, copy=False)
            policy, values = self.predict(boards)
            return policy.astype(np.float32, copy=False), values.squeeze(-1).astype(np.float32, copy=False)
        return evaluate


########
# MCTS #
########
class MCTNode:
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

    def evaluate(self, game: BoardGame, agent: Agent) -> None:
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
                self.children[action] = MCTNode(float(policy[0, action]))

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


def mcts(game: BoardGame, agent: Agent, args: argparse.Namespace, explore: bool) -> np.ndarray:
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
            # end, create a suitable `BoardGame` instance for this node by cloning
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
    policy = np.zeros(args.Game.ACTIONS, dtype=np.float32)
    for action, child in root.children.items():
        policy[action] = child.visit_count
    policy /= policy.sum()
    return policy


############
# Training #
############
ReplayBufferEntry = collections.namedtuple("ReplayBufferEntry", ["board", "policy", "outcome"])


def _augment(board: np.ndarray, policy: np.ndarray, n: int):
    pol2d = policy.reshape(n, n)
    for k in range(4):
        b = np.rot90(board, k, axes=(1, 2))
        p = np.rot90(pol2d, k)
        yield np.ascontiguousarray(b, dtype=np.float32), np.ascontiguousarray(p.reshape(-1), dtype=np.float32)
        yield (np.ascontiguousarray(np.flip(b, axis=2), dtype=np.float32),
               np.ascontiguousarray(np.flip(p, axis=1).reshape(-1), dtype=np.float32))


def sim_game(agent: Agent, args: argparse.Namespace) -> list[ReplayBufferEntry]:
    history = board_game_cpp.simulated_game(agent.cpp_evaluator())

    augment = args.augment and args.game.startswith("pisqorky")
    n = args.Game.N

    entries = []
    for board, policy, value in history:
        board = np.transpose(board, (2, 0, 1)).astype(np.float32, copy=False)
        policy = policy.astype(np.float32, copy=False)
        value = np.float32(value)
        if augment:
            for sym_board, sym_policy in _augment(board, policy, n):
                entries.append(ReplayBufferEntry(sym_board, sym_policy, value))
        else:
            entries.append(ReplayBufferEntry(board, policy, value))
    return entries


def train(args: argparse.Namespace) -> Agent:

    if args.load_path:
        agent = Agent.load(args.load_path, args)
    else:
        agent = Agent(args)
    replay_buffer = npfl139.ReplayBuffer(max_length=args.replay_buffer_length)

    board_game_cpp.simulated_games_start(
        threads=args.threads, num_simulations=args.num_simulations,
        sampling_moves=args.sampling_moves, epsilon=args.epsilon, alpha=args.alpha,
    )

    iteration = 0
    training = True
    best_score = -1.0
    while training:
        iteration += 1

        for _ in range(args.sim_games):
            game = sim_game(agent, args)
            replay_buffer.extend(game)
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
            # Run an evaluation on 2*56 games versus the simple heuristics,
            # using the `Player` instance defined below.
            # For speed, the implementation does not use MCTS during evaluation,
            # but you can of course change it so that it does.
            score = npfl139.board_games.evaluate(
                args.Game, [Player(agent, argparse.Namespace(
                                num_simulations=args.num_simulations // 4, alpha=args.alpha)),
                            args.Game.player_from_name(args.evaluate_against)(seed=main_args.seed)],
                games=56, first_chosen=False, render=False, verbose=False,
            )
            print(f"Evaluation after iteration {iteration}: {100 * score:.1f}%", flush=True)
            agent.save(args.model_path + ".latest")
            if score > best_score:
                best_score = score
                agent.save(args.model_path)

        if iteration >= args.max_iterations:
            training = False

    board_game_cpp.simulated_games_stop()
    agent.save(args.model_path)
    return agent


#############################
# BoardGamePlayer interface #
#############################
class Player(npfl139.board_games.BoardGamePlayer[BoardGame]):
    def __init__(self, agent: Agent, args: argparse.Namespace):
        self.agent = agent
        self.args = args

    def play(self, game: BoardGame) -> int:
        # Predict a best possible action.
        if self.args.num_simulations == 0:
            # TODO: If no simulations should be performed, use directly
            # the policy predicted by the agent on the current game board.
            policy = self.agent.predict(self.agent.board_features(game))[0][0]
        else:
            # TODO: Otherwise run the `mcts` without exploration and
            # utilize the policy returned by it.
            policy = board_game_cpp.mcts(
                game.board, game.to_play, self.agent.cpp_evaluator(),
                num_simulations=self.args.num_simulations, epsilon=0.0, alpha=self.args.alpha,
            )

        # Now select a valid action with the largest probability.
        return max(game.valid_actions(), key=lambda action: policy[action])


########
# Main #
########
def main(args: argparse.Namespace) -> Player:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Create the game class based on the given name.
    args.Game = BoardGame.from_name(args.game)
    board_game_cpp.select_game(args.game)

    if args.game.startswith("pisqorky") and args.evaluate_against == "simple_heuristic":
        args.evaluate_against = "heuristic"

    if args.recodex:
        # Load the trained agent
        agent = Agent.load(args.model_path, args)
    else:
        # Perform training
        agent = train(args)

    return Player(agent, args)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    player = main(main_args)

    # Run an evaluation versus the simple heuristic with the same parameters as in ReCodEx.
    npfl139.board_games.evaluate(
        main_args.Game, [player, main_args.Game.player_from_name(main_args.evaluate_against)(seed=main_args.seed)],
        games=56, first_chosen=False, render=False, verbose=True,
    )
