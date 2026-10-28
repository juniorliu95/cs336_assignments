"""
attention functions for cs336 assignment1 test and for further reuse
junyu, 2026.10.22
"""

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from .linear import linear
from .rope import rope

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d = torch.tensor(Q.shape[-1], dtype=torch.float32)
    qk = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(d)
    masked_qk = qk.masked_fill(mask == False, float("-inf")) if mask is not None else qk
    attention_weights = torch.softmax(masked_qk, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output

def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    d_out, d_in = q_proj_weight.shape
    q_proj = linear(d_in, d_out, q_proj_weight, in_features)
    k_proj = linear(d_in, d_out, k_proj_weight, in_features)
    v_proj = linear(d_in, d_out, v_proj_weight, in_features)

    # split into heads
    head_dim = d_model // num_heads
    q_projs = q_proj.view(*q_proj.shape[:-1], num_heads, head_dim).permute(0, 2, 1, 3)  # (..., num_heads, sequence_length, d_k)
    k_projs = k_proj.view(*k_proj.shape[:-1], num_heads, head_dim).permute(0, 2, 1, 3)
    v_projs = v_proj.view(*v_proj.shape[:-1], num_heads, head_dim).permute(0, 2, 1, 3)

    mask = torch.ones(*q_projs.shape[:-1], k_projs.shape[-2], dtype=torch.bool)  # (..., sequence_length, sequence_length)
    mask = torch.tril(mask)
    att = scaled_dot_product_attention(q_projs, k_projs, v_projs, mask)
    att = att.permute(0, 2, 1, 3).contiguous()  # (..., sequence_length, num_heads, d_k)
    # import pdb;pdb.set_trace()
    att = att.view(*att.shape[:-2], d_model)
    output = linear(d_in, d_out, o_proj_weight, att)

    return output


def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    d_out, d_in = q_proj_weight.shape
    q_proj = linear(d_in, d_out, q_proj_weight, in_features)
    k_proj = linear(d_in, d_out, k_proj_weight, in_features)
    v_proj = linear(d_in, d_out, v_proj_weight, in_features)

    # split into heads
    head_dim = d_model // num_heads
    q_projs = q_proj.view(*q_proj.shape[:-1], num_heads, head_dim).permute(0, 2, 1, 3)  # (..., num_heads, sequence_length, d_k)
    k_projs = k_proj.view(*k_proj.shape[:-1], num_heads, head_dim).permute(0, 2, 1, 3)
    v_projs = v_proj.view(*v_proj.shape[:-1], num_heads, head_dim).permute(0, 2, 1, 3)

    if token_positions is not None:
        token_positions = token_positions.squeeze()
    else: 
        token_positions = torch.arange(q_projs.shape[-2], device=q_projs.device)

    q_projs = rope(head_dim, theta, max_seq_len, q_projs, token_positions)
    k_projs = rope(head_dim, theta, max_seq_len, k_projs, token_positions)

    mask = torch.ones(*q_projs.shape[:-1], k_projs.shape[-2], dtype=torch.bool, device=q_projs.device)  # (..., sequence_length, sequence_length)
    mask = torch.tril(mask)
    att = scaled_dot_product_attention(q_projs, k_projs, v_projs, mask)
    att = att.permute(0, 2, 1, 3).contiguous()  # (..., sequence_length, num_heads, d_k)
    att = att.view(*att.shape[:-2], d_model)

    output = linear(d_in, d_out, o_proj_weight, att)

    return output


if __name__ == "__main__":
    # # scaled_dot_product_attention
    # Q = torch.rand(2, 3, 4)
    # K = torch.rand(2, 5, 4)
    # V = torch.rand(2, 5, 6)
    # true_mask = torch.ones(2,3,5, dtype=torch.bool)
    # mask = torch.tril(true_mask) 
    # output = scaled_dot_product_attention(Q, K, V, mask)
    # print(output.shape)

    # # multihead_self_attention
    # d_model = 8
    # num_heads = 2
    # q_proj_weight = torch.rand(d_model, d_model)
    # k_proj_weight = torch.rand(d_model, d_model)
    # v_proj_weight = torch.rand(d_model, d_model)
    # o_proj_weight = torch.rand(d_model, d_model)
    # in_features = torch.rand(2, 3, d_model)
    # output = multihead_self_attention(d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features)
    # print(output.shape)

    # multihead_self_attention_with_rope
    d_model = 8
    num_heads = 2
    max_seq_len = 10
    theta = 10000.0
    q_proj_weight = torch.rand(d_model, d_model)
    k_proj_weight = torch.rand(d_model, d_model)
    v_proj_weight = torch.rand(d_model, d_model)
    o_proj_weight = torch.rand(d_model, d_model)
    in_features = torch.rand(2, 5, d_model)
    token_positions = torch.tensor([0, 1, 2, 3, 4])
    output = multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, theta, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features, token_positions)
    print(output.shape)  # should be (2, 5, d_model)