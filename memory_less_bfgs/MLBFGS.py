import two_order_utils as TOU

from torch.optim import Optimizer
from torch import Tensor
from two_order_utils import _params_t
from typing import Callable
from torch import linalg as LA


class MLBFGS(Optimizer):
    def __init__(self, params: _params_t, weight_decay: float = 0) -> None:
        if weight_decay < 0:
            raise ValueError(f"Weight decay {weight_decay} should greater than 0.")
        defaults = {'weight_decay': weight_decay}
        super().__init__(params, defaults)
        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def step(self, closure: Callable[[], float] | None = ...) -> float | None:
        epsilon_stop = 1e-6
        group = self.param_groups[0]

        if not closure:
            raise ValueError('Closure must be provided and cannot be None.')
        loss = closure()
        if LA.norm(TOU.gather_flat_grad(self)) < epsilon_stop:
            return loss

        if not self.state:      # Initialise during the first iteration
            self.state['step'] = 0
            x_prev = TOU.gather_flat_param(self)
            g_prev = TOU.gather_flat_grad(self)
            d = g_prev.clone().neg()
            self.state['skip'] = False
        else:
            self.state['step'] += 1
            x_prev = self.state.get('x_prev')
            g_prev = self.state.get('g_prev')
            d = self.state.get('d')

        if not self.state.get('skip'):
            loss, g, t = TOU.LineSearch_n_Update(self, closure, d, g_prev, loss)

        # # g = TOU.gather_flat_grad(self)
        # if LA.norm(g) < epsilon_stop:
        #     self.state['skip'] = True
        #     return loss

        x = TOU.gather_flat_param(self)
        if group['weight_decay'] > 0:
            g.add_(x, alpha=group['weight_decay'])

        s = self._calc_s(x, x_prev)
        if LA.norm(s) < epsilon_stop:
            return loss

        y = self._calc_y(g, g_prev)
        beta = self._calc_beta(s, g, y)
        alpha = self._calc_alpha(y, s, beta, g)
        d = self._calc_d(g, alpha, s, beta, y)

        self.state['x_prev'] = x
        self.state['g_prev'] = g
        self.state['d'] = d
        self.state['skip'] = False

        return loss

    def _calc_s(self, _x: Tensor, _x_prev: Tensor) -> Tensor:
        """ s = x - x_prev """
        return _x - _x_prev

    def _calc_y(self, _g: Tensor, _g_prev: Tensor) -> Tensor:
        """ y = g - g_prev """
        return _g - _g_prev

    def _calc_beta(self, _s: Tensor, _g: Tensor, _y: Tensor) -> Tensor:
        """ beta = s^T * g / s^T * y """
        return _s.dot(_g) / (_s.dot(_y) + 1e-8)

    def _calc_alpha(self, _y: Tensor, _s: Tensor, _beta: Tensor, _g: Tensor) -> Tensor:
        """ alpha = -(1 + y^2 / (s^T * y)) + y^T*g / (s^T*y) """
        return -(1 + _y.dot(_y)/_s.dot(_y).add(1e-8))*_beta + _y.dot(_g)/_s.dot(_y).add(1e-8)

    def _calc_d(self, _g: Tensor, _alpha: float|Tensor, _s: Tensor, _beta: float|Tensor, _y: Tensor
                ) -> Tensor:
        """ d = -g + alpha * s + beta * y """
        return _g.neg() + _alpha*_s + _beta*_y
