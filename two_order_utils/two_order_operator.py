import torch

from functools import reduce
from torch import Tensor
from torch.optim import Optimizer
from typing import Any, Dict, Iterable, Union


_params_t = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]


def _gather_flat_grad(optimizer: Optimizer):
    views = []
    for p in optimizer._params:
        if p.grad is None:
            view = p.new(p.numel()).zero_()
        elif p.grad.is_sparse:
            view = p.grad.to_dense().view(-1)
        else:
            view = p.grad.view(-1)
        views.append(view)
    return torch.cat(views, 0)


def _gather_flat_param(optimizer: Optimizer) -> Tensor:
    """
    Get flatten parameters of the inputs.

    Args:
        optimizer (Optimizer): self, optimizer of selected.

    Returns:
        Tensor: Flatten inputs.
    """
    views = [p.data.view(-1) for p in optimizer._params]
    return torch.cat(views, 0)


def _numel(optimizer: Optimizer):
    if optimizer._numel_cache is None:
        optimizer._numel_cache = reduce(lambda total,
                                        p: total + p.numel(),
                                        optimizer._params, 0)
    return optimizer._numel_cache


def _clone_param(optimizer: Optimizer):
    return [p.clone(memory_format=torch.contiguous_format) for p in optimizer._params]


@torch.no_grad()
def _set_param(optimizer: Optimizer, params_data):
    for p, pdata in zip(optimizer._params, params_data):
        p.copy_(pdata)


def _directional_evaluate(optimizer: Optimizer, closure, x, t: float, d: Tensor):
    """
    Evaluate the value of line search direction.

    Args:
        optimizer (Optimizer): self, class of the optimizer.
        closure (_type_): closure
        x (_type_): Cloned parameters of the model.
        t (float): Alpha value of the updates.
        d (Tensor): Search direction of the updates.

    Returns:
        _type_: _description_
    """
    _add_grad(optimizer, t, d)
    loss = float(closure())
    flat_grad = _gather_flat_grad(optimizer)
    _set_param(optimizer, x)
    return loss, flat_grad


@torch.no_grad()
def _add_grad(optimizer: Optimizer,
              step_size: float | Tensor,
              update: Tensor
              ) -> None:
    """
    Parameters update for flatten gradient.

    Args:
        optimizer (Optimizer): self, class of the optimizer.
        step_size (float | Tensor): Alpha value of the updates.
        update (Tensor): The descent direction used for updates.
    """
    offset = 0
    for p in optimizer._params:
        numel = p.numel()
        # Assert if step size is a number
        # if not isinstance(step_size, (int, float, complex)):
        if isinstance(step_size, (Tensor,)) and step_size.numel() == 1:
            step_size = step_size.data
        # view as to avoid deprecated pointwise semantics
        p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
        offset += numel
    assert offset == optimizer._numel()
