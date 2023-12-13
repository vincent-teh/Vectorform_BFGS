import torch

from functools import reduce
from torch.optim import Optimizer
from torch import Tensor
from typing import Any, Callable, Dict, Optional
from .line_search import _strong_wolfe


class BfgsBaseOptimizer(Optimizer):
    def __init__(self, params, defaults: Dict[str, Any]) -> None:
        super().__init__(params, defaults)
        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

    def gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def gather_flat_param(self) -> Tensor:
        """
        Get flatten parameters of the inputs.

        Returns:
            Tensor: Flatten inputs.
        """
        views = [p.data.view(-1) for p in self._params]
        return torch.cat(views, 0)

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0
            )
        return self._numel_cache

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    @torch.no_grad()
    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t: float, d: Tensor):
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
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self.gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def _add_grad(self, step_size: float | Tensor, update: Tensor) -> None:
        """
        Parameters update for flatten gradient.

        Args:
            optimizer (Optimizer): self, class of the optimizer.
            step_size (float | Tensor): Alpha value of the updates.
            update (Tensor): The descent direction used for updates.
        """
        offset = 0
        for p in self._params:
            numel = p.numel()
            # Assert if step size is a number
            # if not isinstance(step_size, (int, float, complex)):
            if isinstance(step_size, (Tensor,)) and step_size.numel() == 1:
                step_size = step_size.data
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def update(
        self,
        closure: Callable[[], float],
        d: Tensor,
        g: Tensor,
        loss: float,
        cond: Optional[str] = "StrongWolfe",
        max_iter: int = 100,
    ) -> tuple[float, Tensor, float]:
        """Line search algorithm
        ------
        Parameters
        optimizer (Optimizer) : self, instance of the optimizer in used.
        closure (function)  : closure fx of standard PyTorch optimizer.
        d (tensor)  : Single row vector of the descent direction.
        g (tensor)  : Single row vector of the current gradient.
        loss (Any|float)  : The output of the fx.
        cond (str | None)  : Line search algorithm supported - BackTrack, StrongWolfe, None to perform update only.
        max_iter (int)  : Maximum iteration allowed by the algorithm.
        ------
        Return
        Tuple(float, Tensor, float): Loss, Gradient, StepSize(alpha).
        ------
        Description
        """
        if cond is None:
            self._add_grad(1, d)
            loss = closure()
            return loss, self.gather_flat_grad(), 1.0

        if cond == "BackTrack":
            alpha = 1
            gamma = 0.0001
            beta = 0.8
            # Right hand side of the condition
            R = loss + gamma * alpha * g.dot(d)
            self._add_grad(1, d)
            i = 0
            loss = closure()
            while loss > R and i < max_iter:
                alpha_new = alpha * beta
                for _ in self._params:
                    # p - ((alpha_new - alpha_old)*d)
                    self._add_grad(self, (alpha_new - alpha), d)
                alpha = alpha_new
                i = i + 1
                loss = closure()
            g = self.gather_flat_grad()
            return loss, g, alpha

        if cond == "StrongWolfe":
            alpha = 1
            x_init = self._clone_param()
            # directional derivative
            gtd = g.dot(d)  # g * d

            def obj_func(x, t, d):
                return self._directional_evaluate(closure, x, t, d)

            loss, g, alpha, ls_func_evals = _strong_wolfe(
                obj_func, x_init, alpha, d, loss, g, gtd
            )
            self._add_grad(alpha, d)
            return loss, g, alpha

        raise ValueError(f"The line search {cond} was not supported")
