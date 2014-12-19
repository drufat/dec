from dec.spectral2 import *
import itertools
import collections
from numpy.testing import assert_array_almost_equal

eq = assert_array_almost_equal

eq2 = lambda x, y: eq(flat(x), flat(y))

#def test_basis_functions():
#    N = 2
#    Grids = [sp1.Grid_1D_Periodic, sp1.Grid_1D_Chebyshev]
#    for Gx, Gy in itertools.product(Grids, Grids):
#        gx, gy = Gx(N), Gy(N)
#        g = Grid_2D_Cartesian(gx, gy)
#        for P, B in zip(g.projection(), g.basis_fn()):
#            B = flat(B)
#            eq( vstack(flat(P(b)) for b in B),
#                eye(len(B)) )

#def test_d():
#
#    def check_d0(g, f, df):
#        D0, D1, D0d, D1d = g.derivative()
#        eq2( D0(g.P0(f)), g.P1(df))
#        eq2(D0d(g.P0d(f)), g.P1d(df))
#
#    check_d0(Grid_2D_Periodic(4, 3),
#             lambda x, y: sin(x) ,
#             lambda x, y: (cos(x),0) )
#
#    check_d0(Grid_2D_Periodic(11, 13),
#             lambda x, y: sin(x)*cos(y) ,
#             lambda x, y: (cos(x)*cos(y),-sin(x)*sin(y)) )
#
#    def check_d1(g, f, df):
#        D0, D1, D0d, D1d = g.derivative()
#        eq2(D1(g.P1(f)), g.P2(df))
#        eq2(D1d(g.P1d(f)), g.P2d(df))
#
#    check_d1( Grid_2D_Periodic(5, 8),
#             lambda x, y: (sin(x), sin(y)) ,
#             lambda x, y: 0 )
#
#    check_d1( Grid_2D_Periodic(5, 8),
#             lambda x, y: (-sin(y), sin(x)) ,
#             lambda x, y: (cos(x) + cos(y)) )
#
#    def check_d0_bndry(g, f, df):
#        eq2( g.D0(g.P0(f)), g.P1(df))
#        eq( flat(g.D0d(g.P0d(f))) + flat(g.BC0(f)), flat(g.P1d(df)) )
#
#    check_d0_bndry(Grid_2D_Chebyshev(3, 5),
#             lambda x, y: x*y ,
#             lambda x, y: (y,x) )
#
#    check_d0_bndry(Grid_2D_Chebyshev(3, 3),
#             lambda x, y: x ,
#             lambda x, y: (1,0) )
#
#    def check_d1_bndry(g, f, df):
#        D0, D1, D0d, D1d = g.derivative()
#        eq2(D1(g.P1(f)), g.P2(df))
#        eq2(D1d(g.P1d(f)) + g.BC1(f), g.P2d(df))
#
#    check_d1_bndry( Grid_2D_Chebyshev(13, 17),
#             lambda x, y: (-sin(y), sin(x)) ,
#             lambda x, y: (cos(x) + cos(y)) )
#
#    check_d1_bndry( Grid_2D_Chebyshev(13, 17),
#             lambda x, y: (exp(-y), exp(x+y)) ,
#             lambda x, y: (exp(-y) + exp(x+y)) )
#
def test_hodge():

    def test0(g, f):
        H0, H1, H2, H0d, H1d, H2d = g.hodge_star()
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        eq2(H0(P0(f)), P2d(f))
        eq2(H2(P2(f)), P0d(f))
        eq2(H0d(P0d(f)), P2(f))
        eq2(H2d(P2d(f)), P0(f))

    def test1(g, u, v):
        H0, H1, H2, H0d, H1d, H2d = g.hodge_star()
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        eq2(H1(P1(lambda x, y: (u(x,y), v(x,y)))),
            P1d(lambda x, y: (-v(x,y), u(x,y))))
        eq2(H1d(P1d(lambda x, y: (-v(x,y), u(x,y)))),
            P1(lambda x, y: (-u(x,y), -v(x,y))))

    f = lambda x, y: sin(x)*sin(y)
    u = lambda x, y: sin(x)*sin(y)
    v = lambda x, y: cos(y)

    g = Grid_2D_Chebyshev(9, 9)
    test0(g, f)
    test1(g, u, v)

    g = Grid_2D_Periodic(3, 5)
    test0(g, f)
    test1(g, u, v)
