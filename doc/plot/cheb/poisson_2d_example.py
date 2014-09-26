import dec
from dec.spectral2 import *

def run():

    N = 10
    g = Grid_2D_Chebyshev(N, N)

    L0, L1, L0d, L1d = laplacian2(g)
    D0, D1, D0d, D1d = g.derivative()
    P0, P1, P2, P0d, P1d, P2d = g.projection()
    H0, H1, H2, H0d, H1d, H2d = g.hodge_star()
    add = lambda x, y: (x[0]+y[0], x[1]+y[1])
    BC0, BC1 = g.boundary_condition()

    shape = [x.shape for x in g.B1]

    #bc = map(real, H1d(BC0(lambda x, y: 1)) )
    bc = H1d(BC0(lambda x, y: 1))

    def unflat(f):
        fx, fy = split(f, [shape[0][0]*shape[0][1]])
        fx = fx.reshape(shape[0])
        fy = fy.reshape(shape[1])
        return (fx, fy)

    L = to_matrix(lambda x: flat(L1(unflat(x))), 2*N*(N-1))
    f = real(-dot(linalg.inv(L), flat(bc)))
    f = unflat(f)

    def vec(X, Y):
        U = 0
        V = 0
        for _f, _b in zip(flat(f), flat(g.B1)):
            u, v = _b(X, Y)
            U += _f*u
            V += _f*v
        return U, V

    X = linspace(-1, +1, 2*N-1)
    Y = linspace(-1, +1, 2*N-1)
    X, Y = meshgrid(X, Y)
    U, V = vec(X, Y)
    return g.verts, X, Y, U, V

dataname = "poisson_2d_example.json"
#dec.store_data(dataname, run())

import matplotlib
from matplotlib.pyplot import *
matplotlib.rcParams['legend.fancybox'] = True
import matplotlib.pyplot as plt

verts, X, Y, U, V = dec.get_data(dataname)
scatter(*verts)
quiver(X, Y, U, V)

show()

