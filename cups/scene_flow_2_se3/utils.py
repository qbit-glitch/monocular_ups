import torch
from torch import Tensor


def remap_ids(input: Tensor) -> Tensor:
    """Remaps IDs of cluster [0, N].

    Args:
        input (Tensor): ID tensor of the shape [M].

    Returns:
        output (Tensor): Remapped tensor of the shape [M].
    """
    dtype: torch.dtype = input.dtype
    device: torch.device = input.device
    ids = input.unique()
    weight = torch.ones(ids.amax() + 1, dtype=dtype, device=device)
    weight[ids] = torch.arange(start=0, end=ids.shape[0], dtype=dtype, device=device)
    output: Tensor = torch.embedding(indices=input, weight=weight.view(-1, 1))[..., 0]
    return output
