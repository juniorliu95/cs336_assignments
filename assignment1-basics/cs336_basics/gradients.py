"""
gradient related functions for cs336 assignment1 test and for further reuse
junyu, 2026.10.24
"""

from collections.abc import Iterable
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor



def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    l2 = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in parameters if p.grad is not None))
    if l2 > max_l2_norm:
        scale = torch.div(max_l2_norm, l2)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)

    return

