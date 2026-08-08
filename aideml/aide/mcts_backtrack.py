"""
MCTS-based node selection for intelligent backtracking in AIDE.

This module implements Thompson Sampling, a Bayesian bandit algorithm that is:
- Sample efficient (learns from fewer trials)
- Computationally fast (minimal overhead)
- Zero hyperparameters (no tuning required)
- Robust to non-stationary rewards

Thompson Sampling models uncertainty about each node's quality using Beta distributions
and selects nodes by sampling from their posterior distributions.
"""

import logging
import random
from typing import Optional

from .journal import Node

logger = logging.getLogger("aide")


class ThompsonSelector:
    """Fast Thompson Sampling for node selection in backtracking.

    Uses Beta distributions to model uncertainty about node quality.
    Each node maintains Beta(alpha, beta) parameters:
    - alpha: accumulated success (increases with high rewards)
    - beta: accumulated failure (increases with low rewards)

    Selection works by:
    1. Sample a value from each node's Beta(alpha, beta) distribution
    2. Pick the node with the highest sampled value

    This naturally balances exploration (high uncertainty = wide distribution)
    and exploitation (high mean reward = distribution shifted right).

    Attributes:
        prior_alpha: Prior belief about success (default 1.0 = uniform)
        prior_beta: Prior belief about failure (default 1.0 = uniform)
        alpha: Dict mapping node IDs to alpha parameters
        beta: Dict mapping node IDs to beta parameters
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """Initialize Thompson Sampling selector.

        Args:
            prior_alpha: Prior alpha (success count). Higher = more optimistic.
            prior_beta: Prior beta (failure count). Higher = more pessimistic.

        Note:
            Beta(1, 1) is a uniform prior (no bias), which is recommended.
            Beta(2, 1) would be optimistic (assume nodes are good until proven otherwise).
            Beta(1, 2) would be pessimistic (assume nodes are bad until proven otherwise).
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        # Sparse storage: only track nodes we've actually tried
        # This saves memory in large search trees
        self.alpha: dict[str, float] = {}
        self.beta: dict[str, float] = {}

        logger.info(
            f"[Thompson Sampling] Initialized with prior Beta({prior_alpha}, {prior_beta})"
        )

    def select_best_node(self, candidates: list[Node]) -> Node:
        """Select the most promising node using Thompson Sampling.

        For each candidate:
        1. Get its Beta(alpha, beta) parameters (or use prior if unseen)
        2. Sample a value from Beta(alpha, beta)
        3. Return the candidate with the highest sampled value

        Args:
            candidates: List of nodes to choose from

        Returns:
            The selected node

        Raises:
            ValueError: If candidates list is empty
        """
        if not candidates:
            raise ValueError("No candidates provided for selection")

        best_sample = -1.0
        best_node = candidates[0]

        # Sample from each candidate's posterior distribution
        samples = []
        for node in candidates:
            alpha = self.alpha.get(node.id, self.prior_alpha)
            beta = self.beta.get(node.id, self.prior_beta)

            # Sample from Beta distribution
            # Unseen nodes have uniform Beta(1,1) → equal chance across [0,1]
            # Seen nodes have updated distributions → favor high performers
            sample = random.betavariate(alpha, beta)
            samples.append((sample, node.id[:8]))  # For logging

            if sample > best_sample:
                best_sample = sample
                best_node = node

        # Log selection details for debugging
        mean, std, n_samples = self.get_node_stats(best_node)
        logger.info(
            f"[Thompson Sampling] Selected node {best_node.id[:8]} "
            f"(sample={best_sample:.3f}, posterior: μ={mean:.3f}, σ={std:.3f}, n={n_samples:.0f})"
        )
        logger.debug(f"[Thompson Sampling] All samples: {samples}")

        return best_node

    def update_reward(self, node: Node, reward: float):
        """Update Beta distribution parameters with observed reward.

        Performs Bayesian update:
        - alpha increases by the reward (success)
        - beta increases by (1 - reward) (failure)

        Args:
            node: The node that was executed
            reward: Normalized reward in [0, 1]
                   - 0.0 = complete failure (buggy code)
                   - 0.5 = neutral performance
                   - 1.0 = perfect performance (best metric seen)
        """
        # Initialize if first time seeing this node
        if node.id not in self.alpha:
            self.alpha[node.id] = self.prior_alpha
            self.beta[node.id] = self.prior_beta

        # Bayesian update: add observed reward to parameters
        # Higher reward → more success (alpha increases more)
        # Lower reward → more failure (beta increases more)
        self.alpha[node.id] += reward
        self.beta[node.id] += (1.0 - reward)

        # Log the update
        mean, std, n_samples = self.get_node_stats(node)
        logger.info(
            f"[Thompson Sampling] Updated node {node.id[:8]} with reward={reward:.3f} "
            f"(posterior: μ={mean:.3f}, σ={std:.3f}, n={n_samples:.0f})"
        )

    def get_node_stats(self, node: Node) -> tuple[float, float, float]:
        """Get posterior statistics for a node.

        Computes mean, standard deviation, and effective sample count
        from the node's Beta distribution.

        Args:
            node: The node to get statistics for

        Returns:
            Tuple of (mean, std, num_samples):
            - mean: Expected reward (alpha / (alpha + beta))
            - std: Uncertainty in reward estimate
            - num_samples: Effective number of observations (alpha + beta - 2)
        """
        alpha = self.alpha.get(node.id, self.prior_alpha)
        beta = self.beta.get(node.id, self.prior_beta)

        # Beta distribution statistics
        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        std = variance ** 0.5
        num_samples = alpha + beta - 2.0  # Subtract prior counts

        return mean, std, num_samples

    def get_all_stats(self) -> list[dict]:
        """Get statistics for all tracked nodes (for debugging/monitoring).

        Returns:
            List of dicts containing node_id and statistics
        """
        stats = []
        for node_id in self.alpha.keys():
            alpha = self.alpha[node_id]
            beta = self.beta[node_id]
            mean = alpha / (alpha + beta)
            variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
            std = variance ** 0.5
            n_samples = alpha + beta - 2.0

            stats.append({
                'node_id': node_id[:8],
                'mean': mean,
                'std': std,
                'n_samples': n_samples,
                'alpha': alpha,
                'beta': beta,
            })

        # Sort by mean reward (descending)
        stats.sort(key=lambda x: x['mean'], reverse=True)
        return stats


class UCBSelector:
    """UCB1-based node selection (alternative to Thompson Sampling).

    Uses Upper Confidence Bound (UCB1) algorithm:
    UCB = average_reward + c × sqrt(ln(total_visits) / node_visits)

    This is more traditional MCTS but requires tuning the exploration constant c.
    Generally less sample-efficient than Thompson Sampling in practice.
    """

    def __init__(self, exploration_weight: float = 1.414):
        """Initialize UCB selector.

        Args:
            exploration_weight: Exploration constant (c in UCB formula).
                              Default √2 ≈ 1.414 balances exploration/exploitation.
                              Higher = more exploration, lower = more exploitation.
        """
        self.exploration_weight = exploration_weight
        self.node_visits: dict[str, int] = {}
        self.node_rewards: dict[str, float] = {}

        logger.info(
            f"[UCB1] Initialized with exploration_weight={exploration_weight}"
        )

    def select_best_node(self, candidates: list[Node]) -> Node:
        """Select node with highest UCB1 score."""
        if not candidates:
            raise ValueError("No candidates provided for selection")

        # Calculate total visits for UCB formula
        total_visits = sum(self.node_visits.get(n.id, 0) for n in candidates)
        if total_visits == 0:
            total_visits = 1  # Avoid log(0)

        best_score = -float('inf')
        best_node = candidates[0]

        import math

        for node in candidates:
            visits = self.node_visits.get(node.id, 0)

            if visits == 0:
                # Unexplored nodes get infinite score (try all once first)
                score = float('inf')
            else:
                total_reward = self.node_rewards.get(node.id, 0.0)
                avg_reward = total_reward / visits
                exploration_bonus = self.exploration_weight * math.sqrt(
                    math.log(total_visits) / visits
                )
                score = avg_reward + exploration_bonus

            if score > best_score:
                best_score = score
                best_node = node

        visits = self.node_visits.get(best_node.id, 0)
        avg = self.node_rewards.get(best_node.id, 0.0) / max(visits, 1)
        logger.info(
            f"[UCB1] Selected node {best_node.id[:8]} "
            f"(score={best_score:.3f}, avg={avg:.3f}, visits={visits})"
        )

        return best_node

    def update_reward(self, node: Node, reward: float):
        """Update node statistics with observed reward."""
        if node.id not in self.node_visits:
            self.node_visits[node.id] = 0
            self.node_rewards[node.id] = 0.0

        self.node_visits[node.id] += 1
        self.node_rewards[node.id] += reward

        avg = self.node_rewards[node.id] / self.node_visits[node.id]
        logger.info(
            f"[UCB1] Updated node {node.id[:8]} with reward={reward:.3f} "
            f"(avg={avg:.3f}, visits={self.node_visits[node.id]})"
        )

    def get_node_stats(self, node: Node) -> tuple[float, float, float]:
        """Get statistics for a node (mean, 0, visits)."""
        visits = self.node_visits.get(node.id, 0)
        if visits == 0:
            return 0.0, 0.0, 0.0

        avg = self.node_rewards[node.id] / visits
        return avg, 0.0, float(visits)


def create_selector(method: str, **kwargs):
    """Factory function to create node selectors.

    Args:
        method: Selection method ('thompson', 'ucb1', or 'random')
        **kwargs: Additional arguments passed to selector constructor

    Returns:
        Appropriate selector instance

    Raises:
        ValueError: If method is not recognized
    """
    if method == "thompson":
        return ThompsonSelector(**kwargs)
    elif method == "ucb1":
        return UCBSelector(**kwargs)
    elif method == "random":
        return RandomSelector()
    else:
        raise ValueError(
            f"Unknown MCTS method: {method}. "
            f"Choose from: 'thompson', 'ucb1', 'random'"
        )


class RandomSelector:
    """Random baseline selector (no learning)."""

    def __init__(self):
        logger.info("[Random] Using random selection (no MCTS)")

    def select_best_node(self, candidates: list[Node]) -> Node:
        """Select a random node."""
        if not candidates:
            raise ValueError("No candidates provided for selection")

        selected = random.choice(candidates)
        logger.info(f"[Random] Selected node {selected.id[:8]} (random)")
        return selected

    def update_reward(self, node: Node, reward: float):
        """No-op for random selection."""
        pass

    def get_node_stats(self, node: Node) -> tuple[float, float, float]:
        """Return zeros for random selection."""
        return 0.0, 0.0, 0.0
