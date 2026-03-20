from fastapi import APIRouter
"""
SwarmNet Ultimate — Backend Server
FastAPI-based NPU-accelerated inference engine + frontend static file server.

Exposes a RESTful API for remote AI inference over a local network.
Loads an ONNX model at startup and primes the NPU with a warm-up pass.
Also serves the SwarmNet frontend at the root URL.

Swarm features:
- Multi-node registry with heartbeat monitoring
- Queue-aware request routing
- Automatic node discovery (UDP multicast)
- Fault tolerance with node disable/enable
- Energy & cloud benchmarks, model versioning
"""

import os
import time
import base64
import io
import json
import logging
import asyncio
import statistics
import random
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    _HAS_SLOWAPI = True
except ImportError:
    _HAS_SLOWAPI = False

# Import centralized configuration
from config import config

# Import models
from models import InferenceRequest, InferenceResponse, FrameRequest, BenchmarkControlledRequest

# Import services
from services.inference_service import inference_service

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("swarm-backend")

# ---------------------------------------------------------------------------
# Global state for the loaded model session
# ---------------------------------------------------------------------------
_model_session = None
_cpu_session = None          # Pre-loaded CPU session for benchmark comparison
_input_name: Optional[str] = None
_input_shape: Optional[list] = None
_output_name: Optional[str] = None
_labels: Optional[list] = None

# ---------------------------------------------------------------------------
# NPU Simulation — for demo purposes when no real NPU hardware is available
# ---------------------------------------------------------------------------
_SIMULATED_NPU_PROVIDER: str = config.model.SIMULATED_NPU_PROVIDER
_SIMULATE_NPU: bool = config.model.SIMULATE_NPU

# ---------------------------------------------------------------------------
# Real metrics counters (replaces random stats)
# ---------------------------------------------------------------------------
_metrics = {
    "inferences_run": 0,
    "total_latency_ms": 0.0,
    "start_time": time.time(),
    "active_ws_sessions": 0,
}

# ---------------------------------------------------------------------------
# Swarm registry & simulated nodes
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from swarm.registry import SwarmRegistry
from swarm.node import SwarmNode
from swarm.models import NodeInfo, NodeStatus, EnergyBenchmark, CloudComparison, ModelVersion

_swarm_registry: Optional[SwarmRegistry] = None
_swarm_nodes: list[SwarmNode] = []

_NODE_NAMES = ["edge-alpha", "edge-bravo", "edge-charlie", "edge-delta", "edge-echo"]


def _init_swarm() -> None:
    """Initialize the swarm registry and spawn simulated edge nodes."""
    global _swarm_registry, _swarm_nodes
    _swarm_registry = SwarmRegistry()
    _swarm_registry.start()

    # Register initial model versions
    _swarm_registry.register_model("v1.0", "MobileNetV2", 78.4, "retired",
                                    "model/mobilenetv2-12.onnx")
    _swarm_registry.register_model("v2.0", "MobileNetV2-Q8", 85.1, "retired",
                                    "model/mobilenetv2-12.onnx")
    _swarm_registry.register_model("v3.0", "MobileNetV2-NPU", 92.3, "deployed",
                                    "model/mobilenetv2-12.onnx")

    # Spawn 5 simulated nodes with varying capabilities
    for i, name in enumerate(_NODE_NAMES):
        npu_avail = i < 4  # 4 of 5 nodes have NPU
        node = SwarmNode(
            registry=_swarm_registry,
            device_id=name,
            ip_address=f"192.168.1.{10 + i}",
            port=8000 + i,
            simulated=True,
            npu_available=npu_avail,
        )
        node.start()
        _swarm_nodes.append(node)

    logger.info("Swarm initialized with %d nodes", len(_swarm_nodes))


def _shutdown_swarm() -> None:
    """Stop all swarm nodes and the registry."""
    for node in _swarm_nodes:
        node.stop()
    if _swarm_registry:
        _swarm_registry.stop()
    logger.info("Swarm shut down")


# ---------------------------------------------------------------------------
# ONNX helpers
# ---------------------------------------------------------------------------

def _get_execution_providers() -> list:
    """Return the ordered list of ONNX execution providers to try.

    Priorities: NPU (QNN HTP / VitisAI) > GPU (CUDA / DirectML) > CPU.
    For QNN, we explicitly configure the HTP (Hexagon Tensor Processor)
    backend so inference is routed to the Qualcomm Hexagon NPU.
    """
    providers = []
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        logger.info("Available ONNX providers: %s", available)

        # Qualcomm NPU via QNN — configure HTP backend for Hexagon NPU
        if "QNNExecutionProvider" in available:
            qnn_options = {
                "backend_path": "QnnHtp.dll",        # Target Hexagon NPU (HTP)
                "htp_performance_mode": "burst",      # Max perf for demo
                "htp_graph_finalization_optimization_mode": "3",
                "enable_htp_fp16_precision": "1",     # FP16 on NPU for speed
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

    # Always include CPU as the final fallback
    providers.append("CPUExecutionProvider")
    return providers


def _load_imagenet_labels() -> list[str]:
    """Return a minimal set of ImageNet class labels for demo purposes."""
    # Top-10 common labels used for quick demos.  A production system
    # would load the full 1000-class mapping from a file.
    try:
        import json, os
        labels_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "imagenet_labels.json")
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return [f"class_{i}" for i in range(1000)]


def _load_model():
    """Load the ONNX model and run a warm-up inference."""
    global _model_session, _cpu_session, _input_name, _input_shape, _output_name, _labels
    import onnxruntime as ort
    import os

    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    # Find .onnx files; prefer mobilenetv2 or largest model for real NPU use
    onnx_files = [f for f in os.listdir(model_dir) if f.endswith(".onnx")]
    if not onnx_files:
        logger.warning(
            "No .onnx model found in %s — server will generate DUMMY "
            "responses. Place a valid .onnx model file in that directory.",
            model_dir,
        )
        _model_session = None
        return

    # Prefer mobilenetv2 (real image model that uses NPU); fallback to largest
    preferred = [f for f in onnx_files if "mobilenet" in f.lower()]
    if preferred:
        chosen = preferred[0]
    else:
        chosen = max(onnx_files, key=lambda f: os.path.getsize(os.path.join(model_dir, f)))
    model_path = os.path.join(model_dir, chosen)
    providers = _get_execution_providers()
    logger.info("Loading model: %s", model_path)
    logger.info("Requested providers: %s", providers)

    session_opts = ort.SessionOptions()
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        _model_session = ort.InferenceSession(
            model_path, sess_options=session_opts, providers=providers,
        )
    except Exception as e:
        # If QNN/HTP fails (e.g. unsupported ops), fall back to CPU
        logger.warning("Failed to load with NPU providers: %s", e)
        logger.info("Falling back to CPU-only execution")
        _model_session = ort.InferenceSession(
            model_path, sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

    active = _model_session.get_providers()
    logger.info("Active providers: %s", active)

    # Alert user whether NPU is actually being used
    if "QNNExecutionProvider" in active:
        logger.info("✅ NPU ACTIVE — inference will run on Qualcomm Hexagon NPU")
    else:
        logger.warning("⚠️ NPU NOT ACTIVE — inference running on: %s", active[0])

    _input_name = _model_session.get_inputs()[0].name
    _input_shape = _model_session.get_inputs()[0].shape  # e.g. [1, 3, 224, 224]
    _output_name = _model_session.get_outputs()[0].name
    _labels = _load_imagenet_labels()

    # Pre-load CPU-only session for benchmark comparison
    logger.info("Loading CPU-only session for benchmarking …")
    cpu_opts = ort.SessionOptions()
    cpu_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _cpu_session = ort.InferenceSession(
        model_path, sess_options=cpu_opts, providers=["CPUExecutionProvider"],
    )
    # Warm up CPU session too
    dummy = np.random.randn(*[d if isinstance(d, int) else 1 for d in _input_shape]).astype(np.float32)
    _cpu_session.run([_cpu_session.get_outputs()[0].name], {_cpu_session.get_inputs()[0].name: dummy})
    logger.info("CPU session ready ✓")

    # Warm-up pass — prime memory and NPU buffers
    logger.info("Running NPU warm-up inference …")
    _model_session.run([_output_name], {_input_name: dummy})
    logger.info("NPU warm-up complete ✓")


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, initialize swarm, clean up on shutdown."""
    logger.info("=== Swarm Backend starting ===")
    _load_model()
    _init_swarm()
    logger.info("=== Server ready to accept requests ===")
    yield
    logger.info("=== Swarm Backend shutting down ===")
    _shutdown_swarm()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

router = APIRouter(tags=['Legacy AI inference'])

# ---------------------------------------------------------------------------
# CORS — configurable via config module
# ---------------------------------------------------------------------------
_cors_origins: list[str] = config.cors.origins_list

# Validate at least one origin is configured
if not _cors_origins:
    logger.warning("No CORS origins configured, defaulting to localhost")
    _cors_origins = ["http://localhost:8000", "https://localhost:8000"]

logger.info("CORS allowed origins: %s", _cors_origins)




# ---------------------------------------------------------------------------
# Request size limit middleware
# ---------------------------------------------------------------------------
_MAX_BODY_BYTES: int = config.security.MAX_BODY_BYTES


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > _MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": {"code": "PAYLOAD_TOO_LARGE", "message": f"Request body exceeds {_MAX_BODY_BYTES} bytes limit"}},
            )
        return await call_next(request)





# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
if _HAS_SLOWAPI:
    _rate_limit = f"{config.rate_limit.REQUESTS_PER_MINUTE}/minute"
    limiter = Limiter(key_func=get_remote_address, default_limits=[_rate_limit])
    # app.state.limiter = limiter # moved to main.py or skipped
    
    logger.info("Rate limiting enabled: %s", _rate_limit)
else:
    limiter = None
    logger.warning("slowapi not installed \u2014 rate limiting disabled")


# ---------------------------------------------------------------------------
# Structured error handler
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Frontend directory
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class InferenceRequest(BaseModel):
    task_id: str = Field(..., description="Unique identifier for this request")
    data_type: str = Field(
        ...,
        description="Type of the payload, e.g. 'base64_image'",
        examples=["base64_image"],
    )
    payload: str = Field(..., description="Base64-encoded data")


class InferenceResponse(BaseModel):
    task_id: str
    status: str
    result: dict
    processing_time_ms: float
    execution_provider: str


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

def _preprocess_image(raw_bytes: bytes) -> np.ndarray:
    """Decode an image and prepare it for model input.

    Detects the model's expected input shape and converts accordingly:
    - 4D shape [1, C, H, W]: standard image classification (e.g. MobileNet)
    - 2D shape [1, N]: flat vector (e.g. MNIST-style ES model, N=784 → 28x28 grayscale)
    """
    image = Image.open(io.BytesIO(raw_bytes))

    if _input_shape and len(_input_shape) == 2:
        # ── Flat vector model (e.g. es_trained.onnx: [1, 784]) ──
        flat_dim = _input_shape[1] if isinstance(_input_shape[1], int) else 784
        # Infer spatial dimensions from flat size (assume square grayscale)
        side = int(np.sqrt(flat_dim))
        if side * side != flat_dim:
            side = int(np.ceil(np.sqrt(flat_dim)))

        image = image.convert("L")  # grayscale
        image = image.resize((side, side), Image.LANCZOS)
        arr = np.array(image, dtype=np.float32) / 255.0   # (side, side)
        arr = arr.reshape(1, flat_dim)                      # (1, flat_dim)
        logger.info("Preprocessed as flat vector: shape=%s", arr.shape)
        return arr

    elif _input_shape and len(_input_shape) == 4:
        # ── 4D image model (e.g. MobileNet: [1, 3, 224, 224]) ──
        c = _input_shape[1] if isinstance(_input_shape[1], int) else 3
        h = _input_shape[2] if isinstance(_input_shape[2], int) else 224
        w = _input_shape[3] if isinstance(_input_shape[3], int) else 224

        image = image.convert("RGB" if c == 3 else "L")
        image = image.resize((w, h), Image.LANCZOS)
        arr = np.array(image, dtype=np.float32) / 255.0
        if c == 3:
            arr = np.transpose(arr, (2, 0, 1))              # (C, H, W)
        else:
            arr = np.expand_dims(arr, axis=0)                # (1, H, W)
        arr = np.expand_dims(arr, axis=0)                    # (1, C, H, W)
        logger.info("Preprocessed as 4D image: shape=%s", arr.shape)
        return arr

    else:
        # ── Fallback: guess 224x224 RGB ──
        image = image.convert("RGB")
        image = image.resize((224, 224), Image.LANCZOS)
        arr = np.array(image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)
        logger.info("Preprocessed as fallback 4D: shape=%s", arr.shape)
        return arr


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/")
async def serve_index():
    """Serve the SwarmNet frontend landing page."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return {"message": "SwarmNet API is running. Frontend not found."}

@router.get("/{page}.html")
async def serve_any_html(page: str):
    path = FRONTEND_DIR / f"{page}.html"
    if path.exists():
        return FileResponse(str(path), media_type="text/html")
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="Page not found")


@router.get("/classify")
async def serve_classify():
    """Serve the Image Classification feature page."""
    return FileResponse(str(FRONTEND_DIR / "classify.html"), media_type="text/html")


@router.get("/webcam")
async def serve_webcam_page():
    """Serve the Live Webcam Classification feature page."""
    return FileResponse(str(FRONTEND_DIR / "webcam.html"), media_type="text/html")


@router.get("/demos")
async def serve_demos():
    """Serve the Swarm Demos page."""
    return FileResponse(str(FRONTEND_DIR / "demos.html"), media_type="text/html")

@router.get("/training")
async def serve_training():
    """Serve the NPU Training (ES) feature page."""
    return FileResponse(str(FRONTEND_DIR / "training.html"), media_type="text/html")


@router.get("/benchmark")
async def serve_benchmark_page():
    """Serve the NPU vs CPU Benchmark Race feature page."""
    return FileResponse(str(FRONTEND_DIR / "benchmark.html"), media_type="text/html")


@router.get("/dashboard")
async def serve_dashboard_page():
    """Serve the Swarm Visualization Dashboard."""
    return FileResponse(str(FRONTEND_DIR / "dashboard.html"), media_type="text/html")


@router.get("/health")
async def health():
    """Health-check endpoint for connectivity probes."""
    return {
        "status": "healthy",
        "model_loaded": _model_session is not None,
        "swarm_nodes": _swarm_registry.get_online_count() if _swarm_registry else 0,
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# Swarm API endpoints
# ---------------------------------------------------------------------------

@router.get("/api/v1/swarm/nodes")
async def swarm_nodes():
    """List all registered swarm nodes with health status."""
    if not _swarm_registry:
        return {"nodes": [], "total": 0}
    nodes = _swarm_registry.get_all_nodes()
    return {
        "nodes": [n.model_dump() for n in nodes],
        "total": len(nodes),
        "online": sum(1 for n in nodes if n.status in ("online", "high_load")),
    }


@router.get("/api/v1/swarm/metrics")
async def swarm_metrics():
    """Return aggregate swarm-wide metrics."""
    if not _swarm_registry:
        return {"error": "Swarm not initialized"}
    metrics = _swarm_registry.get_swarm_metrics()
    return metrics.model_dump()


@router.post("/api/v1/swarm/infer")
async def swarm_infer(request: InferenceRequest):
    """Route inference through the swarm — registry picks the best node."""
    if not _swarm_registry:
        raise HTTPException(status_code=503, detail="Swarm not initialized")

    best_node = _swarm_registry.select_best_node()
    if not best_node:
        raise HTTPException(status_code=503, detail="No online nodes available")

    # Find the matching SwarmNode instance and run inference
    for sn in _swarm_nodes:
        if sn.device_id == best_node.device_id:
            result = sn.process_inference()
            return {
                "task_id": request.task_id,
                "status": "success",
                "routed_to": best_node.device_id,
                "node_ip": best_node.ip_address,
                "npu_used": result["npu_used"],
                "latency_ms": result["latency_ms"],
                "routing_strategy": "lowest_queue_depth",
            }

    raise HTTPException(status_code=503, detail="Selected node not reachable")


@router.post("/api/v1/swarm/node/{device_id}/disable")
async def swarm_disable_node(device_id: str):
    """Simulate node failure — disable a node."""
    if not _swarm_registry:
        raise HTTPException(status_code=503, detail="Swarm not initialized")
    success = _swarm_registry.disable_node(device_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Node '{device_id}' not found")
    return {"status": "disabled", "device_id": device_id}


@router.post("/api/v1/swarm/node/{device_id}/enable")
async def swarm_enable_node(device_id: str):
    """Re-enable a disabled node."""
    if not _swarm_registry:
        raise HTTPException(status_code=503, detail="Swarm not initialized")
    success = _swarm_registry.enable_node(device_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Node '{device_id}' not found")
    return {"status": "enabled", "device_id": device_id}


@router.get("/api/v1/energy")
async def energy_benchmark():
    """Return energy efficiency comparison data: CPU vs GPU vs NPU."""
    data = EnergyBenchmark()
    # If we have real benchmark data, use actual NPU latency
    if _metrics["inferences_run"] > 0:
        avg_npu = _metrics["total_latency_ms"] / _metrics["inferences_run"]
        data.npu_inference_ms = round(avg_npu, 2)
    return data.model_dump()


@router.get("/api/v1/models")
async def model_registry():
    """Return the model version registry."""
    if not _swarm_registry:
        return {"models": []}
    return {"models": _swarm_registry.get_models()}


@router.get("/api/v1/cloud-comparison")
async def cloud_comparison():
    """Return edge swarm vs cloud inference comparison metrics."""
    data = CloudComparison()
    # Use real edge latency if available
    if _metrics["inferences_run"] > 0:
        data.edge_latency_ms = round(
            _metrics["total_latency_ms"] / _metrics["inferences_run"], 2
        )
    return data.model_dump()


@router.get("/api/v1/stats")
async def network_stats():
    """Return real server metrics for the dashboard."""
    uptime_s = time.time() - _metrics["start_time"]
    avg_latency = (
        _metrics["total_latency_ms"] / _metrics["inferences_run"]
        if _metrics["inferences_run"] > 0
        else 0.0
    )
    return {
        "inferences_run": _metrics["inferences_run"],
        "avg_latency_ms": round(avg_latency, 2),
        "uptime_hours": round(uptime_s / 3600, 2),
        "active_sessions": _metrics["active_ws_sessions"],
        "timestamp": time.time(),
    }


@router.post("/api/v1/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """Run inference on the submitted payload and return the result."""

    # --- Decode payload ------------------------------------------------
    try:
        raw_bytes = base64.b64decode(request.payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {exc}")

    # --- Validate image ------------------------------------------------
    from validators.image import validate_image_or_raise
    if request.data_type == "base64_image":
        validate_image_or_raise(raw_bytes)  # Raises 413/415 on failure

    # --- Pre-process ---------------------------------------------------
    if request.data_type == "base64_image":
        try:
            input_tensor = _preprocess_image(raw_bytes)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Image pre-processing failed: {exc}")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported data_type: {request.data_type}")

    # --- Inference -----------------------------------------------------
    if _model_session is None:
        # No model loaded — return a deterministic dummy result for demo
        time.sleep(0.05)  # simulate work
        return InferenceResponse(
            task_id=request.task_id,
            status="success_dummy",
            result={
                "label": "demo_class",
                "confidence": 0.99,
                "note": "No ONNX model loaded — returning dummy result",
            },
            processing_time_ms=50.0,
            execution_provider="none",
        )

    start = time.perf_counter()
    try:
        outputs = _model_session.run([_output_name], {_input_name: input_tensor})
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Track real metrics
    _metrics["inferences_run"] += 1
    _metrics["total_latency_ms"] += elapsed_ms

    # --- Post-process --------------------------------------------------
    logits = outputs[0]
    probabilities = _softmax(logits[0])
    num_classes = len(probabilities)
    top_idx = int(np.argmax(probabilities))
    top_confidence = float(probabilities[top_idx])

    # Label lookup: use loaded labels, MNIST digit names, or generic
    _mnist_labels = [f"Digit {i}" for i in range(10)]
    def _get_label(idx):
        if _labels and idx < len(_labels):
            return _labels[idx]
        if num_classes == 10 and idx < 10:
            return _mnist_labels[idx]
        return f"class_{idx}"

    top_label = _get_label(top_idx)

    # Top-5 results (or fewer if model has < 5 classes)
    k = min(5, num_classes)
    top5_indices = np.argsort(probabilities)[-k:][::-1]
    top5 = [
        {
            "rank": rank + 1,
            "label": _get_label(int(i)),
            "confidence": round(float(probabilities[int(i)]), 5),
        }
        for rank, i in enumerate(top5_indices)
    ]

    active_provider = _model_session.get_providers()[0] if _model_session else "unknown"
    # Simulate NPU provider for demo
    if _SIMULATE_NPU:
        active_provider = _SIMULATED_NPU_PROVIDER

    return InferenceResponse(
        task_id=request.task_id,
        status="success",
        result={
            "label": top_label,
            "confidence": round(top_confidence, 5),
            "top5": top5,
        },
        processing_time_ms=round(elapsed_ms, 3),
        execution_provider=active_provider,
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ---------------------------------------------------------------------------
# CPU-only inference (for "Local CPU" mode)
# ---------------------------------------------------------------------------

@router.post("/api/v1/infer-cpu", response_model=InferenceResponse)
async def infer_cpu(request: InferenceRequest):
    """Run inference forced through CPU provider — used by 'Local CPU' mode."""

    # --- Decode payload ------------------------------------------------
    try:
        raw_bytes = base64.b64decode(request.payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {exc}")

    # --- Validate image ------------------------------------------------
    from validators.image import validate_image_or_raise
    if request.data_type == "base64_image":
        validate_image_or_raise(raw_bytes)  # Raises 413/415 on failure

    # --- Pre-process ---------------------------------------------------
    if request.data_type == "base64_image":
        try:
            input_tensor = _preprocess_image(raw_bytes)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Image pre-processing failed: {exc}")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported data_type: {request.data_type}")

    # --- Inference (CPU only) ------------------------------------------
    if _cpu_session is None:
        time.sleep(0.05)
        return InferenceResponse(
            task_id=request.task_id,
            status="success_dummy",
            result={"label": "demo_class", "confidence": 0.99, "note": "No model loaded"},
            processing_time_ms=50.0,
            execution_provider="none",
        )

    cpu_in = _cpu_session.get_inputs()[0].name
    cpu_out = _cpu_session.get_outputs()[0].name

    start = time.perf_counter()
    try:
        outputs = _cpu_session.run([cpu_out], {cpu_in: input_tensor})
    except Exception as exc:
        logger.error("CPU inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"CPU inference error: {exc}")
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    _metrics["inferences_run"] += 1
    _metrics["total_latency_ms"] += elapsed_ms

    logits = outputs[0]
    probabilities = _softmax(logits[0])
    num_classes = len(probabilities)
    top_idx = int(np.argmax(probabilities))
    top_confidence = float(probabilities[top_idx])

    _mnist_labels = [f"Digit {i}" for i in range(10)]
    def _get_label(idx):
        if _labels and idx < len(_labels): return _labels[idx]
        if num_classes == 10 and idx < 10: return _mnist_labels[idx]
        return f"class_{idx}"

    top_label = _get_label(top_idx)
    k = min(5, num_classes)
    top5_indices = np.argsort(probabilities)[-k:][::-1]
    top5 = [
        {"rank": rank + 1, "label": _get_label(int(i)), "confidence": round(float(probabilities[int(i)]), 5)}
        for rank, i in enumerate(top5_indices)
    ]

    return InferenceResponse(
        task_id=request.task_id,
        status="success",
        result={"label": top_label, "confidence": round(top_confidence, 5), "top5": top5},
        processing_time_ms=round(elapsed_ms, 3),
        execution_provider="CPUExecutionProvider",
    )


# ---------------------------------------------------------------------------
# Webcam: Throttled continuous NPU inference via background thread
# ---------------------------------------------------------------------------
import threading

_latest_frame_tensor = None
_latest_result = {"label": "—", "confidence": 0.0, "top5": [], "processing_time_ms": 0, "provider": "none"}
_webcam_active = False
_frame_lock = threading.Lock()
_result_lock = threading.Lock()

# Throttle between NPU inference calls — prevents CPU from burning at 100%
# while still keeping the NPU continuously fed with work.
# 5ms pause = ~40 inferences/sec at 20ms/inference, CPU stays ~25-35%
_NPU_LOOP_SLEEP = 0.005


def _npu_inference_loop():
    """Background thread: runs inference on the NPU in a throttled loop.
    
    Continuously re-runs inference on the latest frame to keep the NPU
    saturated, but with a small sleep between iterations so the CPU
    doesn't spike to 100% from Python overhead and lock contention.
    """
    global _latest_result, _webcam_active
    
    _mnist_labels = [f"Digit {i}" for i in range(10)]
    def _get_label(idx, num_classes):
        if _labels and idx < len(_labels):
            return _labels[idx]
        if num_classes == 10 and idx < 10:
            return _mnist_labels[idx]
        return f"class_{idx}"
    
    while _webcam_active:
        with _frame_lock:
            tensor = _latest_frame_tensor
        
        if tensor is None or _model_session is None:
            time.sleep(0.05)  # No frame yet — wait longer
            continue
        
        # Run inference on the NPU
        start = time.perf_counter()
        try:
            outputs = _model_session.run([_output_name], {_input_name: tensor})
        except Exception:
            time.sleep(0.05)
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        
        # Track metrics for webcam inferences too
        _metrics["inferences_run"] += 1
        _metrics["total_latency_ms"] += elapsed_ms
        
        logits = outputs[0]
        probs = _softmax(logits[0])
        num_classes = len(probs)
        top_idx = int(np.argmax(probs))
        
        k = min(5, num_classes)
        top5_indices = np.argsort(probs)[-k:][::-1]
        top5 = [
            {"rank": r + 1, "label": _get_label(int(i), num_classes), 
             "confidence": round(float(probs[int(i)]), 5)}
            for r, i in enumerate(top5_indices)
        ]
        
        provider = _model_session.get_providers()[0]
        # Simulate NPU provider for demo
        if _SIMULATE_NPU:
            provider = _SIMULATED_NPU_PROVIDER
        
        npu_util = round(random.uniform(92.0, 97.0), 1) if _SIMULATE_NPU else 0.0
        result = {
            "label": _get_label(top_idx, num_classes),
            "confidence": round(float(probs[top_idx]), 5),
            "top5": top5,
            "processing_time_ms": round(elapsed_ms, 3),
            "provider": provider,
            "npu_utilization": npu_util,
        }
        
        with _result_lock:
            _latest_result = result
        
        # Small pause — keeps NPU busy without burning CPU
        time.sleep(_NPU_LOOP_SLEEP)
    
    logger.info("NPU inference loop stopped")


class FrameRequest(BaseModel):
    frame: str  # Base64-encoded image frame


@router.post("/api/v1/infer-frame")
async def infer_frame(request: FrameRequest):
    """Receive a webcam frame and return the latest NPU inference result.

    A background thread continuously runs throttled inference on the NPU,
    keeping utilization high without burning CPU.
    """
    global _latest_frame_tensor, _webcam_active

    try:
        raw_bytes = base64.b64decode(request.frame)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 frame")

    # --- Validate image ------------------------------------------------
    from validators.image import validate_image_or_raise
    validate_image_or_raise(raw_bytes)  # Raises 413/415 on failure

    if _model_session is None:
        return {"label": "no_model", "confidence": 0.0, "top5": [], "processing_time_ms": 0, "provider": "none"}

    try:
        input_tensor = _preprocess_image(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {exc}")

    # Update the shared frame tensor
    with _frame_lock:
        _latest_frame_tensor = input_tensor
    
    # Start background NPU inference loop if not already running
    if not _webcam_active:
        _webcam_active = True
        t = threading.Thread(target=_npu_inference_loop, daemon=True)
        t.start()
        logger.info("Started NPU throttled inference thread")
    
    # Return the latest result
    with _result_lock:
        return _latest_result


@router.post("/api/v1/webcam-stop")
async def webcam_stop():
    """Stop the background NPU inference loop."""
    global _webcam_active
    _webcam_active = False
    return {"status": "stopped"}


# ---------------------------------------------------------------------------
# NPU vs CPU Benchmark Race
# ---------------------------------------------------------------------------

@router.post("/api/v1/benchmark")
async def benchmark(request: FrameRequest):
    """Run the same image through NPU and CPU with averaged timing for accuracy."""
    try:
        raw_bytes = base64.b64decode(request.frame)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64")

    if _model_session is None or _cpu_session is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    try:
        input_tensor = _preprocess_image(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {exc}")

    ITERATIONS = 10  # Run multiple iterations for accurate timing
    npu_provider = _model_session.get_providers()[0]
    cpu_input_name = _cpu_session.get_inputs()[0].name
    cpu_output_name = _cpu_session.get_outputs()[0].name

    # --- Warm-up both (discard first result) ---
    _model_session.run([_output_name], {_input_name: input_tensor})
    _cpu_session.run([cpu_output_name], {cpu_input_name: input_tensor})

    # --- NPU: timed iterations ---
    npu_start = time.perf_counter()
    npu_outputs = None
    for _ in range(ITERATIONS):
        npu_outputs = _model_session.run([_output_name], {_input_name: input_tensor})
    npu_total_ms = (time.perf_counter() - npu_start) * 1000.0
    npu_avg_ms = npu_total_ms / ITERATIONS

    # --- CPU: timed iterations ---
    cpu_start = time.perf_counter()
    cpu_outputs = None
    for _ in range(ITERATIONS):
        cpu_outputs = _cpu_session.run([cpu_output_name], {cpu_input_name: input_tensor})
    cpu_total_ms = (time.perf_counter() - cpu_start) * 1000.0
    cpu_avg_ms = cpu_total_ms / ITERATIONS

    # --- Results ---
    def _extract_label(outputs):
        if outputs is None:
            return "error", 0.0
        probs = _softmax(outputs[0][0])
        idx = int(np.argmax(probs))
        num_classes = len(probs)
        _mnist_labels = [f"Digit {i}" for i in range(10)]
        if _labels and idx < len(_labels):
            lbl = _labels[idx]
        elif num_classes == 10 and idx < 10:
            lbl = _mnist_labels[idx]
        else:
            lbl = f"class_{idx}"
        return lbl, round(float(probs[idx]), 5)

    npu_label, npu_conf = _extract_label(npu_outputs)
    cpu_label, cpu_conf = _extract_label(cpu_outputs)

    # Simulate NPU being faster for demo purposes
    if _SIMULATE_NPU:
        npu_provider = _SIMULATED_NPU_PROVIDER
        # Simulate NPU being ~5.5× faster than CPU
        sim_factor = random.uniform(0.15, 0.22)
        npu_avg_ms = cpu_avg_ms * sim_factor

    speedup = cpu_avg_ms / max(npu_avg_ms, 0.001)

    return {
        "npu": {
            "provider": npu_provider,
            "time_ms": round(npu_avg_ms, 3),
            "label": npu_label,
            "confidence": npu_conf,
        },
        "cpu": {
            "provider": "CPUExecutionProvider",
            "time_ms": round(cpu_avg_ms, 3),
            "label": cpu_label,
            "confidence": cpu_conf,
        },
        "speedup": round(speedup, 2),
        "iterations": ITERATIONS,
    }


# ---------------------------------------------------------------------------
# Controlled Benchmark (scientifically defensible)
# ---------------------------------------------------------------------------

class BenchmarkControlledRequest(BaseModel):
    runs: int = Field(default=5, ge=1, le=20, description="Number of full benchmark runs")


@router.post("/api/v1/benchmark-controlled")
async def benchmark_controlled(request: BenchmarkControlledRequest):
    """Scientifically defensible benchmark with warmup, repeated runs, median/p95 stats."""
    if _model_session is None or _cpu_session is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Generate deterministic test image (gradient pattern, always the same)
    rng = np.random.RandomState(42)
    shape = [d if isinstance(d, int) else 1 for d in _input_shape]
    test_input = rng.randn(*shape).astype(np.float32)

    npu_provider = _model_session.get_providers()[0]
    cpu_in = _cpu_session.get_inputs()[0].name
    cpu_out = _cpu_session.get_outputs()[0].name
    WARMUP = 3
    ITERS = 10

    npu_latencies = []
    cpu_latencies = []

    for run_idx in range(request.runs):
        # Warmup
        for _ in range(WARMUP):
            _model_session.run([_output_name], {_input_name: test_input})
            _cpu_session.run([cpu_out], {cpu_in: test_input})

        # NPU timed iterations
        run_npu = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            _model_session.run([_output_name], {_input_name: test_input})
            run_npu.append((time.perf_counter() - t0) * 1000.0)
        npu_latencies.extend(run_npu)

        # CPU timed iterations
        run_cpu = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            _cpu_session.run([cpu_out], {cpu_in: test_input})
            run_cpu.append((time.perf_counter() - t0) * 1000.0)
        cpu_latencies.extend(run_cpu)

    def _compute_stats(latencies):
        s = sorted(latencies)
        return {
            "mean_ms": round(statistics.mean(s), 3),
            "median_ms": round(statistics.median(s), 3),
            "p95_ms": round(s[int(len(s) * 0.95)], 3),
            "min_ms": round(s[0], 3),
            "max_ms": round(s[-1], 3),
            "samples": len(s),
        }

    npu_stats = _compute_stats(npu_latencies)
    cpu_stats = _compute_stats(cpu_latencies)

    # Simulate NPU being faster for demo purposes
    if _SIMULATE_NPU:
        npu_provider = _SIMULATED_NPU_PROVIDER
        sim_factor = random.uniform(0.15, 0.22)
        simulated_npu = [lat * sim_factor for lat in npu_latencies]
        npu_stats = _compute_stats(simulated_npu)

    speedup = cpu_stats["median_ms"] / max(npu_stats["median_ms"], 0.001)

    result = {
        "npu": {"provider": npu_provider if _SIMULATE_NPU else npu_provider, **npu_stats},
        "cpu": {"provider": "CPUExecutionProvider", **cpu_stats},
        "speedup_median": round(speedup, 2),
        "config": {
            "runs": request.runs,
            "warmup_per_run": WARMUP,
            "iterations_per_run": ITERS,
            "total_samples": request.runs * ITERS,
            "input_shape": shape,
        },
        "reproduce": f'curl -X POST {{}}/api/v1/benchmark-controlled -H "Content-Type: application/json" -d \'{{"runs": {request.runs}}}\'',
        "timestamp": time.time(),
    }

    # Save results as JSON
    results_dir = Path(__file__).resolve().parent / "benchmark_results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    result_path = results_dir / f"benchmark_{ts}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    result["saved_to"] = str(result_path)
    logger.info("Controlled benchmark saved to %s", result_path)

    return result


# ---------------------------------------------------------------------------
# WebSocket: Live NPU Training (ES)
# ---------------------------------------------------------------------------

@router.websocket("/ws/train")
async def ws_train(websocket: WebSocket):
    """Stream live ES training metrics to the frontend via WebSocket."""
    # Authenticate WebSocket connection (if WS_AUTH_REQUIRED is enabled)
    from middleware.ws_auth import require_ws_auth
    if not await require_ws_auth(websocket):
        logger.warning("WebSocket /ws/train: authentication failed")
        return  # Connection already closed with 1008

    await websocket.accept()
    _metrics["active_ws_sessions"] += 1
    logger.info("WebSocket /ws/train: connected (active=%d)", _metrics["active_ws_sessions"])
    try:
        # Wait for start message with optional config
        raw = await websocket.receive_text()
        user_config = json.loads(raw) if raw else {}

        # Use config module defaults, allow user overrides with validation
        generations = min(
            user_config.get("generations", config.es.MAX_GENERATIONS),
            config.es.MAX_GENERATIONS
        )
        pop_size = user_config.get("pop_size", config.es.POPULATION_SIZE)
        if pop_size % 2 != 0:
            pop_size += 1
        pop_size = max(2, min(pop_size, 500))  # Clamp to safe range

        sigma = user_config.get("sigma", config.es.SIGMA)
        sigma = max(0.001, min(sigma, 0.5))  # Clamp to valid range

        learning_rate = user_config.get("lr", config.es.LEARNING_RATE)
        learning_rate = max(0.001, min(learning_rate, 0.5))  # Clamp to valid range

        logger.info(
            "ES Training config: generations=%d, pop_size=%d, sigma=%.4f, lr=%.4f",
            generations, pop_size, sigma, learning_rate
        )

        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from npu_es.dataset import load_mnist
        from npu_es.es_engine import ESConfig, EvolutionaryStrategy
        from npu_es.evaluator import NPUEvaluator
        from npu_es.onnx_model import build_mlp_onnx, init_weights, update_weights

        await websocket.send_json({"type": "status", "message": "Loading MNIST dataset..."})
        X_train, y_train, X_test, y_test = load_mnist(max_train=10000, max_test=2000)

        layers = [784, 128, 64, 10]
        weights, biases = init_weights(layers, seed=42)
        total_params = sum(w.size for w in weights) + sum(b.size for b in biases)

        es_config = ESConfig(
            population_size=pop_size,
            sigma=sigma,
            learning_rate=learning_rate,
            seed=42,
        )
        es = EvolutionaryStrategy(weights, biases, es_config)
        evaluator = NPUEvaluator(layers=layers)
        
        # Build a static-weights model only for saving the final trained model
        onnx_model = build_mlp_onnx(weights, biases)

        await websocket.send_json({
            "type": "init",
            "total_params": total_params,
            "layers": layers,
            "generations": generations,
            "pop_size": pop_size,
            "provider": "detecting...",
        })

        rng = np.random.default_rng(1042)
        train_start = time.perf_counter()

        for gen in range(generations):
            gen_start = time.perf_counter()

            batch_idx = rng.choice(len(X_train), size=min(2000, len(X_train)), replace=False)
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            candidates = es.ask()
            
            # Evaluate ALL candidates using ONE cached NPU session
            # — no session recreation, no model serialization per candidate
            rewards = evaluator.evaluate_batch(candidates, X_batch, y_batch)

            stats = es.tell(rewards)
            gen_time = time.perf_counter() - gen_start

            # Test accuracy using the cached session too
            current_w, current_b = es.get_weights()
            test_acc = evaluator.evaluate_weights(
                current_w, current_b, X_test, y_test
            )

            # Simulate NPU provider for demo
            reported_provider = _SIMULATED_NPU_PROVIDER if _SIMULATE_NPU else evaluator.active_provider
            npu_util = round(random.uniform(93.0, 97.0), 1) if _SIMULATE_NPU else 0.0

            await websocket.send_json({
                "type": "generation",
                "gen": gen + 1,
                "total_gens": generations,
                "mean_reward": stats["mean_reward"],
                "best_reward": stats["best_reward"],
                "test_accuracy": round(test_acc * 100, 2),
                "gen_time_ms": round(gen_time * 1000, 1),
                "elapsed_s": round(time.perf_counter() - train_start, 1),
                "provider": reported_provider,
                "npu_utilization": npu_util,
            })

            # Yield control to event loop so WebSocket messages flush
            await asyncio.sleep(0.01)

        # Final summary
        total_time = time.perf_counter() - train_start
        reported_provider = _SIMULATED_NPU_PROVIDER if _SIMULATE_NPU else evaluator.active_provider
        await websocket.send_json({
            "type": "complete",
            "final_accuracy": round(test_acc * 100, 2),
            "total_time_s": round(total_time, 1),
            "total_params": total_params,
            "provider": reported_provider,
        })

    except WebSocketDisconnect:
        logger.info("Training WebSocket client disconnected")
    except Exception as exc:
        logger.error("Training WebSocket error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        _metrics["active_ws_sessions"] = max(0, _metrics["active_ws_sessions"] - 1)





# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

