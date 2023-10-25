import torch
import two_order_utils as TOU

from torch import linalg as LA
from torch.optim import Optimizer
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional
from two_order_utils import _params_t


class VMBFGS(Optimizer):
    def __init__(
        self, params: _params_t, window_size: int = 3, weight_decay: float = 0
    ) -> None:
        if weight_decay < 0:
            raise ValueError(f"Weight decay {weight_decay} should greater than 0.")
        if window_size < 1:
            raise ValueError(f"Memory size {window_size} must be greater than 0")
        defaults = {"weight_decay": weight_decay, "window_size": window_size}
        super().__init__(params, defaults)
        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

    def step(self, closure: Callable[[], float] | None = ...) -> float | None:
        epsilon_stop = 1e-6
        group = self.param_groups[0]
        max_window_size = group["window_size"]

        loss = closure()
        if LA.norm(TOU._gather_flat_grad(self)) < epsilon_stop:
            return loss

        if not self.state:
            self.state["step"] = 0

            x_prev = TOU._gather_flat_param(self)
            g_prev = TOU._gather_flat_grad(self)

            s_prev: List[Tensor] = []
            u_prev: List[Tensor] = []
            n_prev: List[float] = []
            v_prev: List[float] = []

            d = g_prev.clone().neg()
            window_size = 0

        else:
            self.state["step"] += 1

            x_prev: Tensor = self.state.get("x_prev")
            g_prev: Tensor = self.state.get("g_prev")

            s_prev: List[Tensor] = self.state.get("s_prev")
            u_prev: List[Tensor] = self.state.get("u_prev")
            n_prev: List[float] = self.state.get("n_prev")
            v_prev: List[float] = self.state.get("v_prev")

            d: Tensor = self.state.get("d")
            window_size: int = self.state.get("window_size")

        loss, g, _ = TOU._LineSearch_n_Update(self, closure, d, g_prev, loss)

        # Calculation for k+1 iteration
        x = TOU._gather_flat_param(self)
        if group["weight_decay"] > 0:
            g.add_(x, alpha=group["weight_decay"])
        y = g - g_prev
        window_size += 1

        if window_size == 1:
            u_k_1 = y.clone()
        else:
            u_k_1 = self._calc_VM_eqn(
                y, s_prev, n_prev, v_prev, u_prev, window_size - 1
            )
        u_prev.append(u_k_1)
        u = u_prev

        s_prev.append(x - x_prev)  # x_k - x_k-1
        s = s_prev

        n_prev.append(s_prev[-1].dot(y))
        n = n_prev

        v_prev.append(y.dot(u_k_1))
        v = v_prev

        for name, var in (("s", s), ("n", n), ("v", v), ("u", u)):
            if len(var) != window_size:
                raise ValueError(f"{name} does not match the size {window_size}")

        self.state["d"] = self._calc_VM_eqn(g.neg(), s, n, v, u, window_size)
        self.state["x_prev"] = x
        self.state["g_prev"] = g

        if window_size < max_window_size:
            self.state["s_prev"] = s
            self.state["n_prev"] = n
            self.state["u_prev"] = u
            self.state["v_prev"] = v
            self.state["window_size"] = window_size
        if window_size == 3:
            self.state["s_prev"] = []
            self.state["n_prev"] = []
            self.state["u_prev"] = []
            self.state["v_prev"] = []
            self.state["window_size"] = 0

        return loss

    def _calc_b(
        self, var: List[Tensor], s: List[Tensor], n: List[float], window_size: int
    ) -> List[float]:
        """Calculate the value of b for ONE iteration."""
        b = []
        for j in range(window_size):
            b.append(s[j].dot(var) / (n[j] + 1e-8))
        return b

    def _calc_a(
        self,
        u: List[Tensor],
        var: List[Tensor],
        n: List[float],
        v: List[Tensor],
        b: List[Tensor],
        window_size: int,
    ) -> List[float]:
        """
        Calculate the value of a for ONE iteration.

        a(r) = uT*r/n - (1-v/n)*b(r), r = var
        """
        a = []
        for j in range(window_size):
            n[j] += 1e-8
            a.append(u[j].dot(var).div(n[j]) - (1 + v[j] / n[j]) * b[j])
        return a

    def _calc_VM_eqn(
        self,
        var: Tensor,
        s: List[Tensor],
        n: List[float],
        v: List[float],
        u: List[Tensor],
        window_size,
    ) -> Tensor:
        """
        Calculates the VMBFGS equation. var(Tensor) subject to g.neg() or y.
        """
        b = self._calc_b(var, s, n, window_size)
        a = self._calc_a(u, var, n, v, b, window_size)

        summation = torch.zeros_like(var)
        for j in range(window_size):
            summation += a[j] * s[j] + b[j] * u[j]
        return var - summation
