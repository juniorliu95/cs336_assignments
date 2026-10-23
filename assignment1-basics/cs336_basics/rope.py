"""
rope implementation for cs336 assignment1 test and for further reuse
junyu, 2026.10.22
"""
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    sequence_length = token_positions.shape[-1]
    assert sequence_length <= max_seq_len, f"sequence_length {sequence_length} exceeds max_seq_len {max_seq_len}"

    # prepare the basic values
    theta_k_power = torch.div(torch.arange(0, d_k, 2, dtype=torch.float32), d_k) # d_k // 2
    theta_k = torch.div(1, torch.pow(theta, theta_k_power))
    theta_matrix = torch.einsum("i,j->ij", token_positions, theta_k) # [sequence_length, d_k // 2]
    cos_theta = torch.cos(theta_matrix)
    sin_theta = torch.sin(theta_matrix)

    # make the rope matrix
    rope_matrices = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=-1) # [sequence_length, d_k//2, 4]

    # apply the rope matrix to the input tensor
    output = torch.empty_like(in_query_or_key)
    output[..., 0::2] = in_query_or_key[..., 0::2] * rope_matrices[..., 0] + in_query_or_key[..., 1::2] * rope_matrices[..., 1]
    output[..., 1::2] = in_query_or_key[..., 0::2] * rope_matrices[..., 2] + in_query_or_key[..., 1::2] * rope_matrices[..., 3]
    
    return output



if __name__ == "__main__":
    d_k = 4
    theta = 10000.0
    max_seq_len = 10
    in_query_or_key = torch.rand(2, 5, d_k)
    token_positions = torch.tensor([0, 1, 2, 3, 4])
    output = rope(d_k, theta, max_seq_len, in_query_or_key, token_positions)
    print(output.shape)  # should be (2, 5, d_k)