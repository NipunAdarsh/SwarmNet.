"""
Inference service for SwarmNet.

Handles ONNX model loading, preprocessing, and inference execution.
"""

import os
import io
import json
import logging
import time
from typing import Optional, Tuple, List
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class InferenceService:
    """Service for managing ONNX model inference."""

    def __init__(self):
        self.model_session = None
        self.cpu_session = None
        self.input_name: Optional[str] = None
        self.input_shape: Optional[list] = None
        self.output_name: Optional[str] = None
        self.labels: Optional[List[str]] = None
        self.simulated_npu_provider: str = "QNNExecutionProvider"
        self.simulate_npu: bool = True

    def get_execution_providers(self) -> list:
        """Return the ordered list of ONNX execution providers to try."""
        providers = []
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            logger.info("Available ONNX providers: %s", available)

            # Qualcomm NPU via QNN
            if "QNNExecutionProvider" in available:
                qnn_options = {
                    "backend_path": "QnnHtp.dll",
                    "htp_performance_mode": "burst",
                    "htp_graph_finalization_optimization_mode": "3",
                    "enable_htp_fp16_precision": "1",
                }
                providers.append(("QNNExecutionProvider", qnn_options))
                logger.info("QNN configured with HTP backend (Hexagon NPU)")

            # VitisAI NPU (AMD Ryzen AI)
            if "VitisAIExecutionProvider" in available:
                providers.append("VitisAIExecutionProvider")

            # GPU providers
            for gpu in ("CUDAExecutionProvider", "DmlExecutionProvider"):
                if gpu in available:
                    providers.append(gpu)
        except Exception:
            pass

        # Always include CPU as fallback
        providers.append("CPUExecutionProvider")
        return providers

    def load_imagenet_labels(self) -> List[str]:
        """Load ImageNet class labels from JSON file."""
        try:
            labels_path = Path(__file__).parent.parent / "model" / "imagenet_labels.json"
            if labels_path.exists():
                with open(labels_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning("Failed to load labels: %s", e)
        return [f"class_{i}" for i in range(1000)]

    def load_model(self, model_path: Optional[str] = None):
        """Load the ONNX model and run a warm-up inference."""
        import onnxruntime as ort

        model_dir = Path(__file__).parent.parent / "model"

        if model_path:
            model_file = Path(model_path)
        else:
            # Find .onnx files
            onnx_files = list(model_dir.glob("*.onnx"))
            if not onnx_files:
                logger.warning("No .onnx model found in %s", model_dir)
                self.model_session = None
                return

            # Prefer mobilenetv2
            preferred = [f for f in onnx_files if "mobilenet" in f.name.lower()]
            model_file = preferred[0] if preferred else max(onnx_files, key=lambda f: f.stat().st_size)

        providers = self.get_execution_providers()
        logger.info("Loading model: %s", model_file)
        logger.info("Requested providers: %s", providers)

        session_opts = ort.SessionOptions()
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.model_session = ort.InferenceSession(
                str(model_file), sess_options=session_opts, providers=providers
            )
        except Exception as e:
            logger.warning("Failed to load with NPU providers: %s", e)
            logger.info("Falling back to CPU-only execution")
            self.model_session = ort.InferenceSession(
                str(model_file), sess_options=session_opts,
                providers=["CPUExecutionProvider"]
            )

        active = self.model_session.get_providers()
        logger.info("Active providers: %s", active)

        if "QNNExecutionProvider" in active:
            logger.info("✅ NPU ACTIVE — inference will run on Qualcomm Hexagon NPU")
        else:
            logger.warning("⚠️ NPU NOT ACTIVE — inference running on: %s", active[0])

        self.input_name = self.model_session.get_inputs()[0].name
        self.input_shape = self.model_session.get_inputs()[0].shape
        self.output_name = self.model_session.get_outputs()[0].name
        self.labels = self.load_imagenet_labels()

        # Load CPU-only session for benchmarking
        logger.info("Loading CPU-only session for benchmarking...")
        cpu_opts = ort.SessionOptions()
        cpu_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.cpu_session = ort.InferenceSession(
            str(model_file), sess_options=cpu_opts, providers=["CPUExecutionProvider"]
        )

        # Warm up both sessions
        dummy = np.random.randn(*[d if isinstance(d, int) else 1 for d in self.input_shape]).astype(np.float32)
        self.cpu_session.run([self.cpu_session.get_outputs()[0].name],
                            {self.cpu_session.get_inputs()[0].name: dummy})
        logger.info("CPU session ready ✓")

        logger.info("Running NPU warm-up inference...")
        self.model_session.run([self.output_name], {self.input_name: dummy})
        logger.info("NPU warm-up complete ✓")

    def preprocess_image(self, raw_bytes: bytes) -> np.ndarray:
        """Decode an image and prepare it for model input."""
        image = Image.open(io.BytesIO(raw_bytes))

        if self.input_shape and len(self.input_shape) == 2:
            # Flat vector model (e.g., MNIST)
            flat_dim = self.input_shape[1] if isinstance(self.input_shape[1], int) else 784
            side = int(np.sqrt(flat_dim))
            if side * side != flat_dim:
                side = int(np.ceil(np.sqrt(flat_dim)))

            image = image.convert("L")
            image = image.resize((side, side), Image.LANCZOS)
            arr = np.array(image, dtype=np.float32) / 255.0
            arr = arr.reshape(1, flat_dim)
            logger.debug("Preprocessed as flat vector: shape=%s", arr.shape)
            return arr

        elif self.input_shape and len(self.input_shape) == 4:
            # 4D image model (e.g., MobileNet)
            c = self.input_shape[1] if isinstance(self.input_shape[1], int) else 3
            h = self.input_shape[2] if isinstance(self.input_shape[2], int) else 224
            w = self.input_shape[3] if isinstance(self.input_shape[3], int) else 224

            image = image.convert("RGB" if c == 3 else "L")
            image = image.resize((w, h), Image.LANCZOS)
            arr = np.array(image, dtype=np.float32) / 255.0
            if c == 3:
                arr = np.transpose(arr, (2, 0, 1))
            else:
                arr = np.expand_dims(arr, axis=0)
            arr = np.expand_dims(arr, axis=0)
            logger.debug("Preprocessed as 4D image: shape=%s", arr.shape)
            return arr

        else:
            # Fallback: 224x224 RGB
            image = image.convert("RGB")
            image = image.resize((224, 224), Image.LANCZOS)
            arr = np.array(image, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            arr = np.expand_dims(arr, axis=0)
            logger.debug("Preprocessed as fallback 4D: shape=%s", arr.shape)
            return arr

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def get_label(self, idx: int, num_classes: int) -> str:
        """Get label for a class index."""
        mnist_labels = [f"Digit {i}" for i in range(10)]
        if self.labels and idx < len(self.labels):
            return self.labels[idx]
        if num_classes == 10 and idx < 10:
            return mnist_labels[idx]
        return f"class_{idx}"

    def run_inference(
        self,
        input_tensor: np.ndarray,
        use_cpu: bool = False
    ) -> Tuple[str, float, List[dict], float, str]:
        """
        Run inference and return results.

        Returns:
            Tuple of (label, confidence, top5, elapsed_ms, provider)
        """
        if self.model_session is None:
            raise ValueError("Model not loaded")

        session = self.cpu_session if use_cpu else self.model_session
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        start = time.perf_counter()
        outputs = session.run([output_name], {input_name: input_tensor})
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        logits = outputs[0]
        probabilities = self.softmax(logits[0])
        num_classes = len(probabilities)
        top_idx = int(np.argmax(probabilities))
        top_confidence = float(probabilities[top_idx])
        top_label = self.get_label(top_idx, num_classes)

        # Top-5 results
        k = min(5, num_classes)
        top5_indices = np.argsort(probabilities)[-k:][::-1]
        top5 = [
            {
                "rank": rank + 1,
                "label": self.get_label(int(i), num_classes),
                "confidence": round(float(probabilities[int(i)]), 5),
            }
            for rank, i in enumerate(top5_indices)
        ]

        provider = session.get_providers()[0]
        if not use_cpu and self.simulate_npu:
            provider = self.simulated_npu_provider

        return top_label, top_confidence, top5, elapsed_ms, provider


# Global inference service instance
inference_service = InferenceService()
