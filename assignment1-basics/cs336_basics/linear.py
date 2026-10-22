"""
basic functions for cs336 assignment1 test and for further reuse
junyu, 2026.10.22
"""

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

def linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    out = torch.matmul(in_features, weights.T)
    return out


if __name__ == '__main__':
    out = linear(10, 20, torch.randn(20, 10), torch.randn(5, 10))
    print(out.shape)  # should be (5, 20)