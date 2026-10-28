"""
RMSNorm for cs336 assignment1 test and for further reuse
junyu, 2026.10.24
"""

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rms = torch.sqrt(torch.mean(torch.pow(in_features, 2), dim=-1, keepdim=True) + eps)
    return torch.mul(weights, in_features) / rms


if __name__ == "__main__":
    # Example usage
    d_model = 4
    eps = 1e-5
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
    in_features = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    output = rmsnorm(d_model, eps, weights, in_features)
    print(output)
