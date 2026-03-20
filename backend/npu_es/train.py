"""
NPU-Only Evolutionary Strategy Trainer — main entry point.

Trains a small MLP on MNIST using only forward passes on the NPU.
No backpropagation, no gradient computation — just evolutionary search.

Usage::

    python -m npu_es.train --generations 50 --pop-size 50 --sigma 0.02 --lr 0.03

The trained model is saved as an ONNX file compatible with the existing
SwarmNet inference server (backend/server.py).
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

# Ensure parent dir is on path for package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from npu_es.dataset import load_mnist
from npu_es.es_engine import ESConfig, EvolutionaryStrategy
from npu_es.evaluator import NPUEvaluator
from npu_es.onnx_model import build_mlp_onnx, init_weights, save_onnx, update_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("npu-es.train")

# ── Banner ──────────────────────────────────────────────────────────────────
BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║   _   _ ____  _   _       _____ ____    _____          _       ║
║  | \ | |  _ \| | | |     | ____/ ___|  |_   _| __ __ _(_)_ __  ║
║  |  \| | |_) | | | |_____|  _| \___ \    | || '__/ _` | | '_ \ ║
║  | |\  |  __/| |_| |_____| |___ ___) |   | || | | (_| | | | | |║
║  |_| \_|_|    \___/      |_____|____/    |_||_|  \__,_|_|_| |_|║
║                                                                  ║
║  Gradient-Free Model Training Using NPU Forward Passes Only     ║
║  Evolutionary Strategies — No Backpropagation Required          ║
╚══════════════════════════════════════════════════════════════════╝
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a small MLP on MNIST using Evolutionary Strategies (NPU-only forward passes).",
    )
    p.add_argument("--generations", type=int, default=100, help="Number of ES generations (default: 100)")
    p.add_argument("--pop-size", type=int, default=50, help="Population size — must be even (default: 50)")
    p.add_argument("--sigma", type=float, default=0.02, help="Noise standard deviation (default: 0.02)")
    p.add_argument("--lr", type=float, default=0.03, help="Learning rate (default: 0.03)")
    p.add_argument("--momentum", type=float, default=0.9, help="Momentum coefficient (default: 0.9)")
    p.add_argument("--weight-decay", type=float, default=0.005, help="L2 weight decay (default: 0.005)")
    p.add_argument("--batch-size", type=int, default=500, help="Evaluation batch size (default: 500)")
    p.add_argument("--max-train", type=int, default=10000, help="Max training samples (default: 10000)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--output", type=str, default=None, help="Output ONNX path (default: backend/model/es_trained.onnx)")
    return p.parse_args()


def main():
    args = parse_args()
    print(BANNER)

    # Ensure even population size (required for antithetic sampling)
    if args.pop_size % 2 != 0:
        args.pop_size += 1
        logger.info("Adjusted pop_size to %d (must be even for antithetic sampling)", args.pop_size)

    # ── 1. Load dataset ─────────────────────────────────────────────────
    logger.info("Loading MNIST dataset ...")
    X_train, y_train, X_test, y_test = load_mnist(max_train=args.max_train, max_test=2000)
    logger.info("Train: %s, Test: %s", X_train.shape, X_test.shape)

    # ── 2. Initialize model weights ──────────────────────────────────────
    layers = [784, 128, 64, 10]
    weights, biases = init_weights(layers, seed=args.seed)
    total_params = sum(w.size for w in weights) + sum(b.size for b in biases)
    logger.info("Model: %s — %d parameters", " → ".join(map(str, layers)), total_params)

    # ── 3. Set up ES engine + NPU evaluator ──────────────────────────────
    config = ESConfig(
        population_size=args.pop_size,
        sigma=args.sigma,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        seed=args.seed,
    )
    es = EvolutionaryStrategy(weights, biases, config)
    evaluator = NPUEvaluator()
    logger.info("ES config: pop=%d, σ=%.4f, lr=%.4f, momentum=%.2f", config.population_size, config.sigma, config.learning_rate, config.momentum)

    # Build initial ONNX model template
    onnx_model = build_mlp_onnx(weights, biases)

    # ── 4. Training loop ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STARTING ES TRAINING — %d generations", args.generations)
    logger.info("=" * 60)

    rng = np.random.default_rng(args.seed + 1000)
    train_start = time.perf_counter()

    for gen in range(args.generations):
        gen_start = time.perf_counter()

        # Sample a random batch from training data
        batch_idx = rng.choice(len(X_train), size=args.batch_size, replace=False)
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        # Ask ES for candidate weight sets
        candidates = es.ask()

        # Evaluate each candidate on NPU
        rewards = []
        for cand_w, cand_b in candidates:
            model = update_weights(onnx_model, cand_w, cand_b)
            model_bytes = model.SerializeToString()
            reward = evaluator.evaluate(model_bytes, X_batch, y_batch)
            rewards.append(reward)

        # Tell ES the results → updates base weights
        stats = es.tell(rewards)

        gen_time = time.perf_counter() - gen_start

        # Periodic test evaluation
        if gen % 5 == 0 or gen == args.generations - 1:
            current_w, current_b = es.get_weights()
            test_model = update_weights(onnx_model, current_w, current_b)
            test_bytes = test_model.SerializeToString()
            test_acc = evaluator.evaluate(test_bytes, X_test, y_test)
            logger.info(
                "  ├─ Test accuracy: %.2f%% | Gen time: %.1fs | Provider: %s",
                test_acc * 100, gen_time, evaluator.active_provider,
            )

    total_time = time.perf_counter() - train_start

    # ── 5. Final evaluation ──────────────────────────────────────────────
    final_w, final_b = es.get_weights()
    final_model = update_weights(onnx_model, final_w, final_b)
    final_bytes = final_model.SerializeToString()
    final_acc = evaluator.evaluate(final_bytes, X_test, y_test)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("  Total time:     %.1f seconds", total_time)
    logger.info("  Final test acc: %.2f%%", final_acc * 100)
    logger.info("  Best gen reward:%.4f", es.best_reward)
    logger.info("  NPU stats:      %s", evaluator.stats)
    logger.info("=" * 60)

    # ── 6. Save trained model ────────────────────────────────────────────
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "..", "model", "es_trained.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_onnx(final_model, output_path)
    logger.info("Saved trained model → %s", os.path.abspath(output_path))

    print(f"\n✅ Model trained with {total_params:,} parameters using ZERO gradients!")
    print(f"   Final accuracy: {final_acc*100:.2f}% on MNIST test set")
    print(f"   Execution provider: {evaluator.active_provider}")
    print(f"   Output: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
