import two_order_utils as TOU

from torch import linalg as LA
from torch import Tensor
from typing import Callable
from torch.optim import Optimizer
from two_order_utils import _params_t


class OMBFGS(Optimizer):
    def __init__(self, params: _params_t, window_size: int = 3, weight_decay: float = 0) -> None:
        if window_size < 1:
            raise ValueError("Window size must be greater than 1.")
        if weight_decay < 0:
            raise ValueError(f"Weight decay {weight_decay} should greater than 0.")
        defaults = {"window_size": window_size,
                    "weight_decay": weight_decay}
        super().__init__(params, defaults)
        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

    def step(self, closure: Callable[[], float] | None = ...) -> float | None:
        epsilon_stop = 1e-6
        if not closure:
            raise ValueError("Closure must be provided and cannot be None")
        loss = closure()
        if LA.norm(TOU.gather_flat_grad(self)) < epsilon_stop:
            return loss

        if not self.state:
            self.state["step"] = 0

            x_prev = TOU.gather_flat_param(self)
            g_prev = TOU.gather_flat_grad(self)

            s_prev: list[Tensor] = []
            y_prev: list[Tensor] = []
            n_prev: list[Tensor] = []
            d = g_prev.clone().neg()
            win_size = 1
            self.state['win_size'] = 1
        else:
            self.state["step"] += 1
            win_size: int = self.state.get("win_size")

            x_prev: Tensor = self.state.get("x_prev")
            g_prev: Tensor = self.state.get("g_prev")

            s_prev: list[Tensor] = self.state.get("s_prev")
            y_prev: list[Tensor] = self.state.get("y_prev")
            n_prev: list[Tensor] = self.state.get("n_prev")
            d: Tensor = self.state.get("d")

        loss, g, alpha = TOU.LineSearch_n_Update(self, closure, d, g_prev, loss)
        x = TOU.gather_flat_param(self)
        q = g.clone()

        S = self._update_var_array(win_size, s_prev, x - x_prev)
        if LA.norm(S[-1]) < epsilon_stop:
            return loss
        Y = self._update_var_array(win_size, y_prev, g - g_prev)
        N = self._update_var_array(win_size, n_prev, S[-1].dot(Y[-1]))

        q, A = self._run_loop_one(win_size, q, S, N, Y)
        q = self._run_loop_two(win_size, q, Y, N, S, A)

        if win_size < self.param_groups[0]["window_size"]:
            self.state['win_size'] += 1
        self.state['d'] = q.neg()
        self.state['s_prev'] = S
        self.state['y_prev'] = Y
        self.state['n_prev'] = N
        self.state['x_prev'] = x
        self.state['g_prev'] = g

        return loss

    def _run_loop_one(
        self, win_size: int, q: Tensor, S: list[Tensor], N: list[Tensor], Y: list[Tensor]
    ) -> tuple[Tensor, list[Tensor]]:
        """
        Returns:
            tuple[Tensor, list[Tensor]]: Descent direction after loop one, list of A.
        """
        A: list[Tensor] = []
        for i in reversed(range(win_size)):
            A.insert(0, S[i].dot(q).div(N[i]))
            q -= A[0].mul(Y[i])
        return q, A

    def _run_loop_two(
        self,
        win_size: int,
        q: Tensor,
        Y: list[Tensor],
        N: list[Tensor],
        S: list[Tensor],
        A: list[Tensor]
    ) -> Tensor:
        """
        Returns:
            Tensor: Descent direction after loop two.
        """
        for i in range(win_size):
            b = Y[i].dot(q).div(N[i])
            q = q + S[i].mul(A[i] - b)
        return q

    def _update_var_array(
        self, win_size: int, var: list[Tensor], new_val: Tensor
    ) -> list[Tensor]:
        if len(var) == self.param_groups[0]["window_size"]:
            var.pop(0)
        var.append(new_val)
        return var

