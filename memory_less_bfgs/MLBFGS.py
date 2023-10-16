import two_order_utils as TOU

from torch.optim import Optimizer
from torch import Tensor
from two_order_utils import _params_t
from typing import Any, Callable, Dict, Optional


class MLBFGS(Optimizer):
    def __init__(self, params: _params_t) -> None:
        defaults = {}
        super().__init__(params, defaults)
        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def step(self, closure: Callable[[], float] | None = ...) -> float | None:
        epsilon_stop = 1e-6
        if not self.state:      # Initialise during the first iteration
            self.state['step'] = 0
            g_prev = TOU._gather_flat_grad(self)
            d = g_prev.clone().neg()
            self.state['skip'] = False
            self.state['x_prev'] = TOU._gather_flat_param(self)
        else:
            self.state['step'] += 1
            x_prev = self.state.get('x_prev')
            g_prev = self.state.get('g_prev')
            d = self.state.get('d')

        self.update_param(1, d)

        x = TOU._gather_flat_param(self)
        s = self._calc_s(x, x_prev)

        g = TOU._gather_flat_grad(self)
        y = self._calc_y(g, g_prev)

        beta = self._calc_beta(s, g, y)
        alpha = self._calc_alpha(y, s, beta, g)
        d = self._calc_d(g, alpha, s, beta, d)

        self.state['x_prev'] = x
        self.state['g_prev'] = g
        self.state['d_prev'] = d

        return super().step(closure)

    def update_param(self, _step_size: float, _d: Tensor):
        TOU._add_grad(self, _step_size, _d)

    def _calc_s(_x: Tensor, _x_prev: Tensor) -> Tensor:
        """ s = x - x_prev """
        return _x - _x_prev

    def _calc_y(_g: Tensor, _g_prev: Tensor) -> Tensor:
        """ y = g - g_prev """
        return _g - _g_prev

    def _calc_beta(_s: Tensor, _g: Tensor, _y: Tensor) -> Tensor:
        """ beta = s^T * g / s^T * y """
        return _s.dot(_g) / _s.dot(_y)

    def _calc_alpha(_y: Tensor, _s: Tensor, _beta: Tensor, _g: Tensor) -> Tensor:
        """ alpha = -(1 + y^2 / (s^T * y)) + y^T*g / (s^T*y) """
        return -(1 + _y.dot(_y)/_s.dot(_y))*_beta + _y.dot(_g)/_s.dot(_y)

    def _calc_d(_g: Tensor, _alpha: float|Tensor, _s: Tensor, _beta: float|Tensor, _d: Tensor
                ) -> Tensor:
        """ d = -g + alpha * s + beta * d """
        return _g.neg() + _alpha*_s + _beta*_d
