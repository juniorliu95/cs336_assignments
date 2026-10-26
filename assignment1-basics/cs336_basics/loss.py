"""
loss functions for cs336 assignment1 test and for further reuse
junyu, 2026.10.24
"""

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from .activations import softmax


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """

    # NOTE: we cannot use softmax here for calculation.
    # the incorrect category value after softmax can be too small and will underflow to 0 in float32.
    # so we directly calculate the cross entropy loss of log-softmax and the target index.
    #  −log softmax(x)_t = −(x_t-max) + log Σⱼ exp((xⱼ-max))
    #                    = −x_t + max + log Σⱼ exp(xⱼ-max)

    max_logits = torch.max(inputs, dim=1, keepdim=True).values
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs - max_logits), dim=1, keepdim=True))
    log_softmax = inputs - max_logits - log_sum_exp # broadcast so all rows have the same max and log_sum_exp
    loss = -log_softmax[torch.arange(inputs.shape[0]), targets] # select the log-softmax values corresponding to the target indices
    return torch.mean(loss)