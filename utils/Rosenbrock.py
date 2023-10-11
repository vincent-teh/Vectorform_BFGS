from torch import Tensor
from torch.optim import Optimizer
import numpy as np
import torch
from tqdm import tqdm


def Rosenbrock(xy: tuple, a: int = 1, b: int = 100) -> float:
    """Evaluate Rosenbrock function
    Parameters:
    xy (tuple)  : x & y coordinates of type float
    a (int)     : 1st constant of the fx, determines the global minimum of the fx
    b (int)     : 2nd constant of the fx, default to 100
    ------
    Return:
    float: result of the rosenbrock function
    ------
    Desc: f(x,y) = (a-x)^2 + b(y-x^2)^2
    Gloabl minimum of f(x,y) = (a, a^2)
    Never set a == 0
    ------
    Reference: https://en.wikipedia.org/wiki/Rosenbrock_function
    """
    if type(xy) is Tensor:
        xy.requires_grad_(True)
    x, y = xy
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def run_optimization(xy_init: tuple,
                     optimizer_class: Optimizer,
                     n_iter: int,
                     epsilon: float = 1e-8,
                     **optimizer_kwargs: dict):
    """Run optimization finding the minimum of the Rosenbrock function.

    Parameters
    ----------
    xy_init : tuple
        Two floats representing the x resp. y coordinates.

    optimizer_class : object
        Optimizer class.

    n_iter : int
        Number of iterations to run the optimization for.

    epsilon : float
        Convergence tolerance

    optimizer_kwargs : dict
        Additional parameters to be passed into the optimizer.

    Returns
    -------
    path : np.ndarray
        2D array of shape `(n_iter + 1, 2)`. Where the rows represent the
        iteration and the columns represent the x resp. y coordinates.
    """
    xy_t = torch.tensor(xy_init, requires_grad=True)
    optimizer = optimizer_class([xy_t], **optimizer_kwargs)

    path = np.empty((n_iter + 1, 2))
    path[0, :] = xy_init

    for i in tqdm(range(1, n_iter + 1)):
        def closure():
            optimizer.zero_grad()
            loss = Rosenbrock(xy_t)
            loss.backward()
            return loss
        # torch.nn.utils.clip_grad_norm_(xy_t, 1.0)
        loss = optimizer.step(closure)

        path[i, :] = xy_t.detach().numpy()

        # if loss < epsilon:
        #     print("=============Convergence Reach=============")
        #     break

    return path
