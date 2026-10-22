"""
swiglu functions for cs336 assignment1 test and for further reuse
junyu, 2026.10.22
"""

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

def sigmoid(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """
    Compute the sigmoid of the input tensor.

    Args:
        x (Float[Tensor, "..."]): Input tensor.

    Returns:
        Float[Tensor, "..."]: Tensor with the same shape as `x`, where each element is the sigmoid of the corresponding element in `x`.
    """
    return 1 / (1 + torch.exp(-x))


def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """
    Compute the SiLU (Sigmoid Linear Unit) activation function of the input tensor.

    Args:
        x (Float[Tensor, "..."]): Input tensor.

    Returns:
        Float[Tensor, "..."]: Tensor with the same shape as `x`, where each element is the SiLU of the corresponding element in `x`.
    """
    return x * sigmoid(x)


def swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight

    weighted_x = torch.matmul(in_features, w1_weight.T)
    silu_x = silu(weighted_x)
    output = torch.matmul(silu_x * torch.matmul(in_features, w3_weight.T), w2_weight.T)

    return output

if __name__ == "__main__":
    d_model = 512
    d_ff = 2048
    w1_weight = torch.randn(d_ff, d_model)
    w2_weight = torch.randn(d_model, d_ff)
    w3_weight = torch.randn(d_ff, d_model)
    in_features = torch.randn(2, 3, d_model)

    out = swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
        in_features=in_features,
    )
    print(out.shape)  # should be (2, 3, 512)