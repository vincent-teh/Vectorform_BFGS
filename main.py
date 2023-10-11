from utils import run_optimization, create_animation, Rosenbrock
from conjgrad import ConjGrad


if __name__ == "__main__":
    INIT = (2.,2.)
    MAX_ITER = 35
    path_cg = run_optimization(INIT,ConjGrad,MAX_ITER, 1e-8, variant='HS')
    print(path_cg)
    paths = [path_cg,]
    # anim  = create_animation(Rosenbrock,
    #                          paths,
    #                          colors=['black'],
    #                          names=['ConjGrad'],
    #                          x_lim=(-2, 3),
    #                          y_lim=(-1., 5.))
