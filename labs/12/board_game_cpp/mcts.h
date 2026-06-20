// This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "board_game.h"

template <BoardGame G> using Policy = std::array<float, G::ACTIONS>;

template <BoardGame G>
using Evaluator = std::function<void(const G &, Policy<G> &, float &)>;

template <BoardGame G> class MCTNode {
public:
  MCTNode(float p) : prior(p) {};
  float prior, total_value = 0.0f;
  int visit_count = 0;
  std::optional<G> game;
  std::unordered_map<int, std::unique_ptr<MCTNode<G>>> children;

  float value() {
    if (visit_count == 0)
      return 0;
    return total_value / visit_count;
  }

  bool is_evaluated() { return visit_count > 0; }
  void evaluate(const G &g, const Evaluator<G> &eval) {
    Outcome outcome = g.outcome();
    float val;
    if (outcome != 0) {
      val = (float)outcome - 2.0f;
    } else {
      Policy<G> p;
      eval(g, p, val);
      for (int i = 0; i < G::ACTIONS; ++i) {
        if (g.valid(i)) {
          children.emplace(i, std::make_unique<MCTNode>(p[i]));
        }
      }
    }
    total_value = val;
    visit_count = 1;
  }
  void add_exploration_noise(float eps, float alpha, std::mt19937 &rng) {
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    std::vector<float> noise(children.size());
    float sum = 0;
    for (auto &x : noise) {
      x = gamma(rng);
      sum += x;
    }
    for (auto &x : noise)
      x /= sum;

    std::size_t i = 0;
    for (auto &[_, child] : children) {
      child->prior = eps * noise[i++] + (1.0f - eps) * child->prior;
    }
  }
  std::pair<int, MCTNode *> select_child() {
    int best_a = -1;
    MCTNode *node = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();
    double sqrtN = std::sqrt(visit_count);
    double C = std::log((1.0f + visit_count + 19652.0f) / 19652.f) + 1.25f;
    double U, Q;
    for (const auto &[action, child] : children) {
      U = C * child->prior * sqrtN / (child->visit_count + 1.0f);
      Q = -child->value();
      if (Q + U > best_score) {
        best_score = Q + U;
        best_a = action;
        node = child.get();
      }
    }
    return std::make_pair(best_a, node);
  }
};

template <BoardGame G>
void mcts(const G &game, const Evaluator<G> &evaluator, int num_simulations,
          float epsilon, float alpha, Policy<G> &policy) {
  // TODO: Implement MCTS, returning the generated `policy`.
  //
  // To run the neural network, use the given `evaluator`, which returns a
  // policy and a value function for the given game.
  auto root = std::make_unique<MCTNode<G>>(0.0f);
  MCTNode<G> *node;
  float value;
  std::vector<MCTNode<G> *> path;
  path.reserve(G::N * G::N);

  root->evaluate(game, evaluator);
  root->add_exploration_noise(epsilon, alpha, *board_game_generator);

  for (int i = 0; i < num_simulations; ++i) {
    node = root.get();
    G game_copy = game;
    path.push_back(node);

    while (node->children.size() > 0) {
      auto [action, child] = node->select_child();
      game_copy.move(action);
      node = child;
      path.push_back(node);
    }
    if (!node->is_evaluated())
      node->evaluate(game_copy, evaluator);
    else {
      node->total_value += node->value();
      node->visit_count++;
    }
    value = node->value();
    for (auto it = path.rbegin() + 1; it != path.rend(); ++it) {
      value = -value;
      (*it)->total_value += value;
      (*it)->visit_count++;
    }
    path.clear();
  }
  float sum = 0.0f;
  for (auto &[action, child] : root->children) {
    policy[action] = child->visit_count;
    sum += child->visit_count;
  }
  for (auto &[action, child] : root->children) {
    policy[action] /= sum;
  }
}
