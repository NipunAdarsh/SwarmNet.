"""
ONNX Model Builder — constructs a small MLP as an ONNX graph.

Two modes:
1. build_mlp_onnx() — weights baked as initializers (for saving/exporting)
2. build_mlp_dynamic_weights() — weights as RUNTIME INPUTS (for ES training)

The dynamic-weights model allows a SINGLE ONNX Runtime session to evaluate
all ES candidates by passing different weight tensors each call, eliminating
the massive CPU overhead of session creation per candidate.

Architecture:  Input(784) → Dense(128,ReLU) → Dense(64,ReLU) → Dense(10)
Total params:  ~109 K
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Default MLP architecture for MNIST
DEFAULT_LAYERS = [784, 128, 64, 10]


def _xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Xavier / Glorot uniform initialization."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)


def init_weights(
    layers: list[int] | None = None,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Initialize random weights and biases for an MLP.

    Returns
    -------
    weights : list[np.ndarray]
        Weight matrices for each layer.
    biases : list[np.ndarray]
        Bias vectors for each layer.
    """
    if layers is None:
        layers = DEFAULT_LAYERS
    rng = np.random.default_rng(seed)
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        weights.append(_xavier_init(layers[i], layers[i + 1], rng))
        biases.append(np.zeros(layers[i + 1], dtype=np.float32))
    return weights, biases


def build_mlp_onnx(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
) -> onnx.ModelProto:
    """Build an ONNX model with weights baked as initializers.

    Used for saving/exporting the final trained model.
    """
    n_layers = len(weights)
    assert len(biases) == n_layers

    initializers = []
    for i, (w, b) in enumerate(zip(weights, biases)):
        initializers.append(numpy_helper.from_array(w, name=f"W{i}"))
        initializers.append(numpy_helper.from_array(b, name=f"B{i}"))

    nodes = []
    input_name = "input"

    for i in range(n_layers):
        matmul_out = f"matmul_{i}"
        add_out = f"dense_{i}"

        nodes.append(
            helper.make_node("MatMul", [input_name, f"W{i}"], [matmul_out])
        )
        nodes.append(
            helper.make_node("Add", [matmul_out, f"B{i}"], [add_out])
        )

        if i < n_layers - 1:
            relu_out = f"relu_{i}"
            nodes.append(helper.make_node("Relu", [add_out], [relu_out]))
            input_name = relu_out
        else:
            input_name = add_out

    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, weights[0].shape[0]])
    Y = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, weights[-1].shape[1]])

    graph = helper.make_graph(nodes, "es_mlp", [X], [Y], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def build_mlp_dynamic_weights(
    layers: list[int] | None = None,
) -> onnx.ModelProto:
    """Build an ONNX model where weights are RUNTIME INPUTS (not initializers).

    This allows reusing a single ONNX Runtime session for all ES candidates —
    just pass different weight/bias tensors each call to session.run().

    The model has these inputs:
      - "input"  : (batch_size, layers[0])    — data
      - "W0"     : (layers[0], layers[1])     — weight layer 0
      - "B0"     : (layers[1],)               — bias layer 0
      - "W1"     : (layers[1], layers[2])     — weight layer 1
      - ... etc.

    Returns
    -------
    onnx.ModelProto
        ONNX model with dynamic weight inputs.
    """
    if layers is None:
        layers = DEFAULT_LAYERS
    n_layers = len(layers) - 1

    nodes = []
    graph_inputs = []
    input_name = "input"

    # Data input
    graph_inputs.append(
        helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, layers[0]])
    )

    # Weight/bias inputs + compute nodes for each layer
    for i in range(n_layers):
        w_name = f"W{i}"
        b_name = f"B{i}"
        matmul_out = f"matmul_{i}"
        add_out = f"dense_{i}"

        # Add weight and bias as graph inputs (dynamic!)
        graph_inputs.append(
            helper.make_tensor_value_info(w_name, TensorProto.FLOAT, [layers[i], layers[i + 1]])
        )
        graph_inputs.append(
            helper.make_tensor_value_info(b_name, TensorProto.FLOAT, [layers[i + 1]])
        )

        nodes.append(helper.make_node("MatMul", [input_name, w_name], [matmul_out]))
        nodes.append(helper.make_node("Add", [matmul_out, b_name], [add_out]))

        if i < n_layers - 1:
            relu_out = f"relu_{i}"
            nodes.append(helper.make_node("Relu", [add_out], [relu_out]))
            input_name = relu_out
        else:
            input_name = add_out  # final layer — raw logits

    # Output
    Y = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, layers[-1]])

    graph = helper.make_graph(nodes, "es_mlp_dynamic", graph_inputs, [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def update_weights(
    model: onnx.ModelProto,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
) -> onnx.ModelProto:
    """Replace initializer tensors in an existing ONNX model (in-place).

    This avoids rebuilding the full graph for each ES candidate — we just
    swap out the weight data.
    """
    idx = 0
    for i, (w, b) in enumerate(zip(weights, biases)):
        model.graph.initializer[idx].CopyFrom(numpy_helper.from_array(w, name=f"W{i}"))
        model.graph.initializer[idx + 1].CopyFrom(numpy_helper.from_array(b, name=f"B{i}"))
        idx += 2
    return model


def save_onnx(model: onnx.ModelProto, path: str) -> None:
    """Save an ONNX model to disk."""
    onnx.save(model, path)
