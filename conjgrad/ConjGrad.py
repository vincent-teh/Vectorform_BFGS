from typing import Any, Callable, Dict, get_args, Iterable, Literal, Union

import torch
from torch.optim import Optimizer
from torch.optim.lbfgs import _strong_wolfe
from functools import reduce
from torch import Tensor
from torch import linalg as LA
from bfgs_base import BfgsBaseOptimizer


_LIST_VARIANTS = Literal['FR', 'PRP', 'DY', 'HS']
_LIST_LINE_SEARCH = Literal["BackTrack", "StrongWolfe"]
_params_t = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]


class ConjGrad(BfgsBaseOptimizer):
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
            raise ValueError(f'Variant must be of type {get_args(_LIST_VARIANTS)}')
        if line_search not in get_args(_LIST_LINE_SEARCH):
            raise ValueError(f'Line search must be of type {get_args(_LIST_VARIANTS)}')
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay value {weight_decay}. Must be positive.")

        defaults = {'variant': variant,
                    'line_search': line_search,
                    'weight_decay': weight_decay
                   }
        super().__init__(params, defaults)


    def _calc_step_size(self, g: Tensor,
                      g_prev: Tensor,
                      d: Tensor | None = None,
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
        return max(int(beta), 0)

    def step(self, closure: Callable[[], float]) -> float:
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
            _ = self.update(
                closure, d, g_prev, loss, cond=group['line_search'])

        loss = closure()
        g = self.gather_flat_grad()
        if LA.norm(g) < epsilon_stop:   # Skipped update.
            self.state['skip'] = True
            return loss
        if group['weight_decay'] > 0:
            g.add_(self.gather_flat_param(), alpha=group['weight_decay'])

        self.state['skip'] = False
        beta = self._calc_step_size(g, g_prev, d, option=group['variant'])
        d = g.neg() + beta*d
        self.state['d'] = d
        self.state['g'] = g

        return loss
