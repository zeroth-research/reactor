"""The tensors that the operations hold, as the evaluators compute with them.

An operation holds each matrix as given: a ``torch.Tensor``, or a list of rows, which
is also how a tensor made in Rust comes back. The frontends make torch tensors;
``as_torch`` and ``as_numpy`` give any of these as a torch tensor or a NumPy array,
for code that computes with them.
"""

import numpy as np
import torch


def as_torch(t) -> torch.Tensor:
    """``t`` as a torch tensor: itself, or a list of rows copied."""
    if isinstance(t, torch.Tensor):
        return t
    return torch.tensor(t)


def as_numpy(t) -> np.ndarray:
    """``t`` as a NumPy array."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)
