from utils import run_optimization, create_animation, Rosenbrock
from conjgrad import ConjGrad
from memory_less_bfgs import MLBFGS


if __name__ == "__main__":
    INIT = (2.,2.)
    MAX_ITER = 30
    path_cg = run_optimization(
        INIT,ConjGrad,MAX_ITER, 1e-8, variant='HS',weight_decay=1e-4)
    path_mlbfgs = run_optimization(INIT,MLBFGS,MAX_ITER)
    print(path_cg, path_mlbfgs)
    paths = [path_cg, path_mlbfgs,]
    # anim  = create_animation(Rosenbrock,
    #                          paths,
    #                          colors=['black'],
    #                          names=['ConjGrad'],
    #                          x_lim=(-2, 3),
    #                          y_lim=(-1., 5.))
