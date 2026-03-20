"""
OpenAI-style Evolutionary Strategy engine — gradient-free optimization.

Uses only forward passes (fitness evaluations) to estimate the direction
of steepest improvement. No backpropagation, no gradient computation.

Key features:
- Antithetic sampling (pairs of +ε, -ε perturbations) for variance reduction
- Rank-based reward normalization for stability
- Weight decay for regularization
- Momentum for smoother convergence

Reference: Salimans et al., "Evolution Strategies as a Scalable
Alternative to Reinforcement Learning", 2017.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("npu-es.engine")


@dataclass
class ESConfig:
    """Hyperparameters for the Evolutionary Strategy."""

    population_size: int = 50       # Number of perturbations per generation
    sigma: float = 0.02             # Noise standard deviation
    learning_rate: float = 0.03     # Step size
    weight_decay: float = 0.005     # L2 regularization
    momentum: float = 0.9           # Momentum coefficient (0 = disabled)
    seed: int = 42


class EvolutionaryStrategy:
    """Gradient-free optimizer using Evolutionary Strategies.

    Usage::

        es = EvolutionaryStrategy(weights, biases, config)

        for gen in range(num_generations):
            candidates = es.ask()        # get perturbation pairs
            rewards = evaluate(candidates)  # forward-pass on NPU
            es.tell(rewards)             # update base weights
            best_w, best_b = es.get_weights()

    Parameters
    ----------
    weights : list[np.ndarray]
        Initial weight matrices for the model.
    biases : list[np.ndarray]
        Initial bias vectors for the model.
    config : ESConfig
        Hyperparameter configuration.
    """

    def __init__(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        config: ESConfig | None = None,
    ):
        self.config = config or ESConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Current "best" weights (mutable — updated each generation)
        self.weights = [w.copy() for w in weights]
        self.biases = [b.copy() for b in biases]

        # Momentum buffers
        self._velocity_w = [np.zeros_like(w) for w in weights]
        self._velocity_b = [np.zeros_like(b) for b in biases]

        # Stored perturbations from the latest ask()
        self._noise_w: list[list[np.ndarray]] = []
        self._noise_b: list[list[np.ndarray]] = []

        self.generation: int = 0
        self.best_reward: float = -float("inf")

    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        total = sum(w.size for w in self.weights)
        total += sum(b.size for b in self.biases)
        return total

    def ask(self) -> list[tuple[list[np.ndarray], list[np.ndarray]]]:
        """Generate a population of perturbed weight sets.

        Uses **antithetic sampling**: for each noise vector ε, we evaluate
        both (θ + σε) and (θ − σε). This halves the variance of the
        gradient estimate at no extra cost.

        Returns
        -------
        candidates : list[tuple[weights, biases]]
            List of ``population_size`` candidate weight sets.
        """
        pop = self.config.population_size
        half = pop // 2
        sigma = self.config.sigma

        self._noise_w = []
        self._noise_b = []
        candidates = []

        for _ in range(half):
            # Random noise for each layer
            nw = [self.rng.standard_normal(w.shape).astype(np.float32) for w in self.weights]
            nb = [self.rng.standard_normal(b.shape).astype(np.float32) for b in self.biases]
            self._noise_w.append(nw)
            self._noise_b.append(nb)

            # Positive perturbation: θ + σε
            pos_w = [w + sigma * n for w, n in zip(self.weights, nw)]
            pos_b = [b + sigma * n for b, n in zip(self.biases, nb)]
            candidates.append((pos_w, pos_b))

            # Negative perturbation: θ − σε
            neg_w = [w - sigma * n for w, n in zip(self.weights, nw)]
            neg_b = [b - sigma * n for b, n in zip(self.biases, nb)]
            candidates.append((neg_w, neg_b))

        return candidates

    def tell(self, rewards: list[float]) -> dict:
        """Update base weights using the reward-weighted noise.

        Parameters
        ----------
        rewards : list[float]
            Rewards for each candidate (same order as ``ask()`` output).

        Returns
        -------
        dict
            Statistics for this generation: mean_reward, best_reward, etc.
        """
        pop = len(rewards)
        half = pop // 2

        # Rank-based normalization — more robust than raw rewards
        ranked = _rank_normalize(rewards)

        # Estimate "gradient" from antithetic pairs
        lr = self.config.learning_rate
        sigma = self.config.sigma
        wd = self.config.weight_decay
        mom = self.config.momentum

        for layer_idx in range(len(self.weights)):
            grad_w = np.zeros_like(self.weights[layer_idx])
            grad_b = np.zeros_like(self.biases[layer_idx])

            for i in range(half):
                pos_reward = ranked[2 * i]       # reward for +ε
                neg_reward = ranked[2 * i + 1]   # reward for -ε

                # The gradient estimate: Σ (R+ - R-) * ε / (2 * half * σ)
                grad_w += (pos_reward - neg_reward) * self._noise_w[i][layer_idx]
                grad_b += (pos_reward - neg_reward) * self._noise_b[i][layer_idx]

            grad_w /= 2 * half * sigma
            grad_b /= 2 * half * sigma

            # Apply weight decay
            grad_w -= wd * self.weights[layer_idx]
            grad_b -= wd * self.biases[layer_idx]

            # Momentum update
            self._velocity_w[layer_idx] = mom * self._velocity_w[layer_idx] + (1 - mom) * grad_w
            self._velocity_b[layer_idx] = mom * self._velocity_b[layer_idx] + (1 - mom) * grad_b

            # Step
            self.weights[layer_idx] += lr * self._velocity_w[layer_idx]
            self.biases[layer_idx] += lr * self._velocity_b[layer_idx]

        # Track stats
        self.generation += 1
        mean_r = float(np.mean(rewards))
        best_r = float(np.max(rewards))
        self.best_reward = max(self.best_reward, best_r)

        stats = {
            "generation": self.generation,
            "mean_reward": round(mean_r, 4),
            "best_reward": round(best_r, 4),
            "best_ever": round(self.best_reward, 4),
        }
        logger.info(
            "Gen %3d | mean=%.4f | best=%.4f | best_ever=%.4f",
            self.generation, mean_r, best_r, self.best_reward,
        )
        return stats

    def get_weights(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return current base weights and biases."""
        return self.weights, self.biases


def _rank_normalize(rewards: list[float]) -> np.ndarray:
    """Rank-based reward normalization (fitness shaping).

    Maps raw rewards to a zero-mean distribution based on rank.
    This makes ES invariant to monotonic transformations of the reward
    and improves robustness.
    """
    n = len(rewards)
    ranks = np.zeros(n)
    sorted_indices = np.argsort(rewards)
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank

    # Normalize to [-0.5, 0.5] range
    ranks = ranks / (n - 1) - 0.5
    return ranks
