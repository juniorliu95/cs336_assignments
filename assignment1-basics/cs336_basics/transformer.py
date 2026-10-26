"""
transformer block for cs336 assignment1 test and for further reuse
junyu, 2026.10.24
"""

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from .embedding import embedding
from .activations import swiglu
from.attention import multihead_self_attention_with_rope
from .rmsnorm import rmsnorm


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = (
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.output_proj.weight"],
    )

    prenorm_x = rmsnorm(d_model, 1e-5, weights["ln1.weight"], in_features)
    y = multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, theta, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, prenorm_x) + in_features
    prenorm_y = rmsnorm(d_model, 1e-5, weights["ln2.weight"], y)
    z = swiglu(d_model, d_ff, weights["ffn.w1.weight"], weights["ffn.w2.weight"], weights["ffn.w3.weight"], prenorm_y) + y
    return z


def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """

    # embedding layer
    token_embeddings = weights["token_embeddings.weight"]
    x = embedding(vocab_size, d_model, token_embeddings, in_indices)

    # transformer blocks
    for layer_idx in range(num_layers):
        layer_weights = {
            "attn.q_proj.weight": weights[f"layers.{layer_idx}.attn.q_proj.weight"],
            "attn.k_proj.weight": weights[f"layers.{layer_idx}.attn.k_proj.weight"],
            "attn.v_proj.weight": weights[f"layers.{layer_idx}.attn.v_proj.weight"],
            "attn.output_proj.weight": weights[f"layers.{layer_idx}.attn.output_proj.weight"],
            "ln1.weight": weights[f"layers.{layer_idx}.ln1.weight"],
            "ffn.w1.weight": weights[f"layers.{layer_idx}.ffn.w1.weight"],
            "ffn.w2.weight": weights[f"layers.{layer_idx}.ffn.w2.weight"],
            "ffn.w3.weight": weights[f"layers.{layer_idx}.ffn.w3.weight"],
            "ln2.weight": weights[f"layers.{layer_idx}.ln2.weight"],
        }
        x = transformer_block(d_model, num_heads, d_ff, context_length, rope_theta, layer_weights, x)

    # output layer
    out = rmsnorm(d_model, 1e-5, weights["ln_final.weight"], x)

    # linear trans to map output (batch, seq, d_model) to (batch, seq, vocab_size).
    lm_head_weight = weights["lm_head.weight"]
    out = torch.einsum("bld,vd->blv", out, lm_head_weight)  # (batch_size, sequence_length, vocab_size)

    return out



if __name__ == "__main__":
    # test transformer block
    d_model = 8
    num_heads = 2
    d_ff = 16
    max_seq_len = 4
    theta = 10000.0

    weights = {
        "attn.q_proj.weight": torch.randn(d_model, d_model),
        "attn.k_proj.weight": torch.randn(d_model, d_model),
        "attn.v_proj.weight": torch.randn(d_model, d_model),
        "attn.output_proj.weight": torch.randn(d_model, d_model),
        "ln1.weight": torch.randn(d_model),
        "ffn.w1.weight": torch.randn(d_ff, d_model),
        "ffn.w2.weight": torch.randn(d_model, d_ff),
        "ffn.w3.weight": torch.randn(d_ff, d_model),
        "ln2.weight": torch.randn(d_model),
    }

    in_features = torch.randn(2, 4, d_model)

    output = transformer_block(
        d_model,
        num_heads,
        d_ff,
        max_seq_len,
        theta,
        weights,
        in_features,
    )
    print(output.shape)