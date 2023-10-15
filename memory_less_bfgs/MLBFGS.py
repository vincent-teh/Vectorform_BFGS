from typing import Any, Callable, Dict, Optional
from torch.optim import Optimizer
from two_order_utils import _params_t


class MLBFGS(Optimizer):
    def __init__(self, params: _params_t, defaults: Dict[str, Any]) -> None:
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = ...) -> float | None:
        return super().step(closure)

