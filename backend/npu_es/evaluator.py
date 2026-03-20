"""
NPU-based fitness evaluator for Evolutionary Strategy training.

Runs ONNX model forward passes through the NPU (via QNN Execution Provider)
to compute fitness scores for ES candidate weight sets.

OPTIMIZED: Uses a SINGLE cached ONNX Runtime session with a dynamic-weights
model. Weights are passed as runtime inputs to session.run() instead of
being baked as initializers. This eliminates the massive CPU overhead of
creating a new session for every candidate (~15ms CPU each × hundreds of
evaluations).
"""

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger("npu-es.evaluator")


def _get_providers() -> list:
    """Return ordered ONNX execution providers — NPU first, CPU fallback."""
    providers = []
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        logger.info("Available providers: %s", available)

        # Qualcomm Hexagon NPU
        if "QNNExecutionProvider" in available:
            providers.append(
                (
                    "QNNExecutionProvider",
                    {
                        "backend_path": "QnnHtp.dll",
                        "htp_performance_mode": "burst",
                        "htp_graph_finalization_optimization_mode": "3",
                        "enable_htp_fp16_precision": "1",
                    },
                )
            )

        # AMD Ryzen AI NPU
        if "VitisAIExecutionProvider" in available:
            providers.append("VitisAIExecutionProvider")

        # Intel NPU via OpenVINO
        if "OpenVINOExecutionProvider" in available:
            providers.append("OpenVINOExecutionProvider")

        # GPU fallbacks
        for gpu in ("CUDAExecutionProvider", "DmlExecutionProvider"):
            if gpu in available:
                providers.append(gpu)
    except Exception:
        pass

    # Always include CPU as final fallback
    providers.append("CPUExecutionProvider")
    return providers


class NPUEvaluator:
    """Evaluate ES candidates by running forward passes on the NPU.

    OPTIMIZATION: Uses a single cached ONNX Runtime session with dynamic-weight
    inputs. Instead of recreating a session for each candidate (CPU-heavy), we
    create ONE session at startup and pass different weight tensors each call.

    This shifts the workload from CPU (session compilation) to NPU (inference),
    dramatically reducing CPU usage and increasing NPU utilization.

    Parameters
    ----------
    providers : list, optional
        ONNX execution providers override. If None, auto-detects NPU/CPU.
    layers : list[int], optional
        MLP layer sizes. Default: [784, 128, 64, 10].
    """

    def __init__(self, providers: Optional[list] = None, layers: Optional[list] = None):
        self.providers = providers or _get_providers()
        self.layers = layers or [784, 128, 64, 10]
        self._active_provider: str = "unknown"
        self._eval_count: int = 0
        self._total_inference_ms: float = 0.0
        self._cached_session = None
        self._input_names: list[str] = []
        self._output_name: str = ""
        logger.info("NPUEvaluator initialized with providers: %s", self.providers)

    def _ensure_session(self):
        """Create and cache the ONNX Runtime session on first use."""
        if self._cached_session is not None:
            return

        import onnxruntime as ort
        from npu_es.onnx_model import build_mlp_dynamic_weights

        # Build the dynamic-weights model (weights are inputs, not initializers)
        model = build_mlp_dynamic_weights(self.layers)
        model_bytes = model.SerializeToString()

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        # Limit CPU threads to reduce contention
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1

        try:
            session = ort.InferenceSession(
                model_bytes, sess_options=opts, providers=self.providers
            )
        except Exception:
            logger.warning("NPU session creation failed, falling back to CPU")
            session = ort.InferenceSession(
                model_bytes,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

        self._cached_session = session
        self._active_provider = session.get_providers()[0]
        self._input_names = [inp.name for inp in session.get_inputs()]
        self._output_name = session.get_outputs()[0].name

        logger.info(
            "Cached NPU session created (provider: %s, inputs: %s)",
            self._active_provider,
            self._input_names,
        )

    @property
    def active_provider(self) -> str:
        return self._active_provider

    @property
    def stats(self) -> dict:
        return {
            "sessions_created": 1 if self._cached_session else 0,
            "evaluations": self._eval_count,
            "total_inference_ms": round(self._total_inference_ms, 2),
            "avg_inference_ms": round(
                self._total_inference_ms / max(1, self._eval_count), 2
            ),
            "active_provider": self._active_provider,
        }

    def evaluate_weights(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        X_batch: np.ndarray,
        y_batch: np.ndarray,
    ) -> float:
        """Evaluate a candidate's weights on a data batch.

        Uses the CACHED session — no session recreation overhead.
        Weights are passed as runtime inputs alongside the data.

        Parameters
        ----------
        weights : list[np.ndarray]
            Weight matrices for each layer.
        biases : list[np.ndarray]
            Bias vectors for each layer.
        X_batch : np.ndarray
            Input data, shape ``(batch_size, input_dim)``.
        y_batch : np.ndarray
            True labels, shape ``(batch_size,)`` — integer class indices.

        Returns
        -------
        float
            Accuracy on this batch (0.0 to 1.0).
        """
        self._ensure_session()

        # Build feed dict: data + all weight/bias tensors
        feed = {"input": X_batch.astype(np.float32)}
        for i, (w, b) in enumerate(zip(weights, biases)):
            feed[f"W{i}"] = w.astype(np.float32)
            feed[f"B{i}"] = b.astype(np.float32)

        # Run inference on NPU — single session, no recreation
        t0 = time.perf_counter()
        logits = self._cached_session.run([self._output_name], feed)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._eval_count += 1
        self._total_inference_ms += elapsed_ms

        # Compute accuracy as reward
        predictions = np.argmax(logits, axis=1)
        accuracy = float(np.mean(predictions == y_batch))
        return accuracy

    def evaluate(
        self,
        onnx_model_bytes: bytes,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
    ) -> float:
        """Legacy evaluate method — creates a new session per call.

        Kept for backward compatibility with non-training code paths
        (e.g., test accuracy evaluation after training completes).
        """
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1

        try:
            session = ort.InferenceSession(
                onnx_model_bytes, sess_options=opts, providers=self.providers
            )
        except Exception:
            session = ort.InferenceSession(
                onnx_model_bytes,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

        self._active_provider = session.get_providers()[0]

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_data = X_batch.astype(np.float32)

        t0 = time.perf_counter()
        logits = session.run([output_name], {input_name: input_data})[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._total_inference_ms += elapsed_ms

        predictions = np.argmax(logits, axis=1)
        accuracy = float(np.mean(predictions == y_batch))
        return accuracy

    def evaluate_batch(
        self,
        candidates: list[tuple[list[np.ndarray], list[np.ndarray]]],
        X_batch: np.ndarray,
        y_batch: np.ndarray,
    ) -> list[float]:
        """Evaluate multiple candidates using the cached session.

        No session recreation between candidates — all use the same session
        with different weight inputs.
        """
        rewards = []
        for cand_w, cand_b in candidates:
            reward = self.evaluate_weights(cand_w, cand_b, X_batch, y_batch)
            rewards.append(reward)
        return rewards
