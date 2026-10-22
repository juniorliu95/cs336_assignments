"""
embedding functions for cs336 assignment1 test and for further reuse
junyu, 2026.10.22
"""
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

def embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    assert weights.shape[0] == vocab_size, f"Expected weights shape[0] to be {vocab_size}, but got {weights.shape[0]}"

    out = weights[token_ids]

    assert out.shape[-1] == d_model, f"Expected embedding dimension {d_model}, but got {out.shape[-1]}"

    return out

if __name__ == "__main__":
    out = embedding(
        vocab_size=10_000,
        d_model=512,
        weights=torch.randn(10_000, 512),
        token_ids=torch.randint(0, 10_000, (2, 3)),
    )
    print(out.shape)  # should be (2, 3, 512)
