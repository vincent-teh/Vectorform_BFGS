from typing import Any, Callable, Dict, get_args, Iterable, Literal, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lbfgs import _strong_wolfe
from functools import reduce
from torch import Tensor
from torch import linalg as LA

from two_order_utils import _strong_wolfe


_LIST_VARIANTS = Literal['FR', 'PRP', 'DY', 'HS']
_LIST_LINE_SEARCH = Literal["BackTrack", "StrongWolfe"]
_params_t = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]


class ConjGrad(Optimizer):
    def __init__(self, params: _params_t,
                 variant: _LIST_VARIANTS = 'PRP',
                 weight_decay: float = 0,
                 line_search: _LIST_LINE_SEARCH = 'StrongWolfe') -> None:
        """
        Nonlinear Conjugate Gradient

        Args:
            variant (_LIST_VARIANTS, optional): Variant of CG momentum. Defaults to 'PRP'.
            weight_decay (float | None, optional): Performs regularization when provided. Defaults to None.
            line_search (_LIST_LINE_SEARCH, optional): Line search algorithm to be used. Defaults to 'StrongWolfe'.
        """
        if variant not in get_args(_LIST_VARIANTS):
            raise ValueError(
                f'Variant must be of type {get_args(_LIST_VARIANTS)}')
        if line_search not in get_args(_LIST_LINE_SEARCH):
            raise ValueError(
                f'Line search must be of type {get_args(_LIST_VARIANTS)}')
        if weight_decay < 0:
            raise ValueError(
                f"Invalid weight decay value {weight_decay}. Must be positive.")

        defaults = {'variant': variant,
                    'line_search': line_search,
                    'weight_decay': weight_decay}

        super().__init__(params, defaults)
        self._params = self.param_groups[0]['params']
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

    def gather_flat_param(self):
        return torch.cat([p.data.view(-1) for p in self._params], 0)
        views = []
        for p in self._params:
            view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    @torch.no_grad()
    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self.gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def _add_grad(self, step_size: float, update: Tensor) -> None:
        '''Parameters update for flatten gradient
        ------
        Parameters
        step_size (float)   : Alpha value of the updates.
        update (Tensor)     : The descent direction used for updates.
        ------
        Return
            None
        '''
        offset = 0
        for p in self._params:
            numel = p.numel()
            # Assert if step size is a number
            # if not isinstance(step_size, (int, float, complex)):
            if isinstance(step_size, (Tensor,)) and step_size.numel() == 1:
                step_size = step_size.data
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def LineSearch_n_Update(self, closure: Callable[[], float],
                             d: Tensor,
                             g: Tensor,
                             loss: Any,
                             cond: str = 'BackTrack',
                             max_iter: int = 100) -> None:
        '''Line search algorithm
        ------
        Parameters
        closure (function)  : closure fx of standard PyTorch optimizer.
        d (tensor)  : Single row vector of the descent direction.
        g (tensor)  : Single row vector of the current gradient.
        loss (Any)  : The output of the fx.
        cond (str)  : Line search algorithm supported - BackTrack, StrongWolfe.
        max_iter (int)  : Maximum iteration allowed by the algorithm.
        ------
        Return
        (None)
        ------
        Description
        '''
        if cond == 'BackTrack':
            alpha_old = 1
            gamma = 0.0001
            beta = 0.8
            # Right hand side of the condition
            R = loss + gamma*alpha_old*g.dot(d)
            self._add_grad(1, d)
            i = 0
            while closure() > R and i < max_iter:
                alpha_new = alpha_old*beta
                for p in self._params:
                    # p - ((alpha_new - alpha_old)*d)
                    self._add_grad((alpha_new - alpha_old), d)
                alpha_old = alpha_new
                i = i + 1
        elif cond == 'StrongWolfe':
            alpha = 1
            x_init = self._clone_param()
            # directional derivative
            gtd = g.dot(d)  # g * d

            def obj_func(x, t, d):
                return self._directional_evaluate(closure, x, t, d)
            loss, g, alpha, ls_func_evals = _strong_wolfe(
                obj_func, x_init, alpha, d, loss, g, gtd)
            self._add_grad(alpha, d)
            return loss, g, alpha
        else:
            raise ValueError("The supplied line search was not supported")

    def _CalcStepSize(self, g: Tensor,
                      g_prev: Tensor,
                      d: Tensor = None,
                      epsilon: float = 1e-5,
                      option: str = 'PR') -> float:
        '''CG Momentum Calculation
        ------
        Parameters
        g (Tensor)      : Current flatten gradient tensor.
        g_prev (Tensor) : Previous flatten gradient tensor.
        d (Tensor)      : Steepest direction, optional.
        epsilon (float) : Small number to avoid division by zero.
        option (str)    : Choice of momentum algorithm, support 'FR', 'PRP','DY','HS'.
        ------
        Return
        (float) : Momentum value of the conjugate gradient.
        '''
        if option == 'FR':
            beta = LA.norm(g)**2 / (LA.norm(g_prev)**2 + epsilon)
        elif option == 'PRP':
            beta = g.dot(g - g_prev) / (LA.norm(g_prev)**2 + epsilon)
        elif option == 'DY':
            assert d is not None
            y = g - g_prev
            beta = LA.norm(g)**2 / (d.dot(y) + epsilon)
        elif option == 'HS':
            assert d is not None
            y = g - g_prev
            beta = g.dot(y) / (d.dot(y) + epsilon)
        else:
            raise ValueError("The supplied CG variant is not supported.")
        return max(beta, 0)

    def step(self, closure: Callable[[], float] | None = ...) -> float | None:
        ''' Conjugate Gradient Steps
        ----------------------------
        Constants
        d : descent direction
        p : param group, variables to be optimise
        '''
        epsilon_stop = 1e-6

        if closure is not None:
            loss = closure()

        if not self.state:      # Initialise during the first iteration
            self.state['step'] = 0
            g_prev = self.gather_flat_grad()
            d = g_prev.clone().neg()
            self.state['skip'] = False
        else:
            d = self.state['d']  # Previous descent direction
            g_prev = self.state['g']
            self.state['step'] = self.state['step'] + 1

        group = self.param_groups[0]

        if self.state['skip'] == False:
            _ = self.LineSearch_n_Update(
                closure, d, g_prev, loss, cond=group['line_search'])

        loss = closure()
        g = self.gather_flat_grad()
        if LA.norm(g) < epsilon_stop:   # Skipped update.
            self.state['skip'] = True
            return loss
        if group['weight_decay'] > 0:
            def gather_flat_param(): return torch.cat(
                [p.data.view(-1) for p in self._params], 0)
            g.add_(gather_flat_param(), alpha=group['weight_decay'])

        self.state['skip'] = False
        beta = self._CalcStepSize(g, g_prev, d, option=group['variant'])
        d = g.neg() + beta*d
        self.state['d'] = d
        self.state['g'] = g

        return loss
