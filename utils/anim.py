import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def create_animation(function,
                     paths,
                     colors,
                     names,
                     figsize=(12, 12),
                     x_lim=(-2, 2),
                     y_lim=(-1, 3),
                     n_seconds=5,
                     save: bool=True):
    """Create an animation.

    Parameters
    ----------
    paths : list
        List of arrays representing the paths (history of x,y coordinates) the
        optimizer went through.

    colors :  list
        List of strings representing colors for each path.

    names : list
        List of strings representing names for each path.

    figsize : tuple
        Size of the figure.

    x_lim, y_lim : tuple
        Range of the x resp. y axis.

    n_seconds : int
        Number of seconds the animation should last.

    Returns
    -------
    anim : FuncAnimation
        Animation of the paths of all the optimizers.
    """
    if not (len(paths) == len(colors) == len(names)):
        raise ValueError

    path_length = max(len(path) for path in paths)

    n_points = 300
    x = np.linspace(*x_lim, n_points)
    y = np.linspace(*y_lim, n_points)
    X, Y = np.meshgrid(x, y)
    Z = function([X, Y])

    minimum = (1.0, 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.contour(X, Y, Z, 90, cmap="jet")

    scatters = [ax.scatter(None,
                           None,
                           label=label,
                           c=c) for c, label in zip(colors, names)]

    ax.legend(prop={"size": 25})
    ax.plot(*minimum, "rD")

    def animate(i):
        for path, scatter in zip(paths, scatters):
            scatter.set_offsets(path[:i, :])

        ax.set_title(str(i))

    ms_per_frame = 1000 * n_seconds / path_length

    anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)
    if save == True:
        writergif = PillowWriter(fps=30)
        anim.save('result.gif', writer=writergif)

    return anim