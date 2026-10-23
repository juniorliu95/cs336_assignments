"""
attention functions for cs336 assignment1 test and for further reuse
junyu, 2026.10.22
"""
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

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

if __name__ == "__main__":
    Q = torch.rand(2, 3, 4)
    K = torch.rand(2, 5, 4)
    V = torch.rand(2, 5, 6)
    true_mask = torch.ones(2,3,5, dtype=torch.bool)
    mask = torch.tril(true_mask) 
    output = scaled_dot_product_attention(Q, K, V, mask)
    print(output.shape)