"""
NPU-Only Evolutionary Strategy Training Engine.

Trains neural networks using ONLY forward passes (no backpropagation).
Uses OpenAI-style Evolutionary Strategies with NPU-accelerated fitness
evaluation via ONNX Runtime + QNN Execution Provider.
"""

from .es_engine import EvolutionaryStrategy
from .evaluator import NPUEvaluator
from .onnx_model import build_mlp_onnx, update_weights

__all__ = [
    "EvolutionaryStrategy",
    "NPUEvaluator",
    "build_mlp_onnx",
    "update_weights",
]
