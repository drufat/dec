from dec.grid2 import *
import itertools
import collections
from numpy.testing import assert_array_almost_equal as eq
eq2 = lambda x, y: eq(flat(x), flat(y))

def test_d():
 
    def check_d0(g, f, df):
        D0, D1, D0d, D1d = g.derivative()
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        eq2( D0(P0(f)), P1(df))
        eq2(D0d(P0d(f)), P1d(df))
 
    check_d0(Grid_2D_Periodic(4, 3),
             lambda x, y: sin(x) ,
             lambda x, y: (cos(x),0) )
 
    check_d0(Grid_2D_Periodic(3, 5),
             lambda x, y: sin(x)*cos(y) ,
             lambda x, y: (cos(x)*cos(y),-sin(x)*sin(y)) )
 
    def check_d1(g, f, df):
        D0, D1, D0d, D1d = g.derivative()
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        eq2(D1(P1(f)), P2(df))
        eq2(D1d(P1d(f)), P2d(df))
 
    check_d1( Grid_2D_Periodic(5, 6),
             lambda x, y: (sin(x), sin(y)) ,
             lambda x, y: 0 )
 
    check_d1( Grid_2D_Periodic(5, 6),
             lambda x, y: (-sin(y), sin(x)) ,
             lambda x, y: (cos(x) + cos(y)) )
 
    def check_d0_bndry(g, f, df):
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        D0, D1, D0d, D1d = g.derivative()
        BC0, BC1 = g.boundary_condition() 
        eq2( D0(P0(f)), P1(df))
        eq( flat(D0d(P0d(f))) + flat(BC0(f)), flat(P1d(df)) )
 
    check_d0_bndry(Grid_2D_Chebyshev(3, 5),
             lambda x, y: x*y ,
             lambda x, y: (y,x) )
 
    check_d0_bndry(Grid_2D_Chebyshev(3, 3),
             lambda x, y: x ,
             lambda x, y: (1,0) )
 
    def check_d1_bndry(g, f, df):
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        D0, D1, D0d, D1d = g.derivative()
        BC0, BC1 = g.boundary_condition() 
        eq2(D1(P1(f)), P2(df))
        eq2(D1d(P1d(f)) + BC1(f), P2d(df))
 
    check_d1_bndry( Grid_2D_Chebyshev(3, 5),
             lambda x, y: (-sin(y), sin(x)) ,
             lambda x, y: (cos(x) + cos(y)) )
 
    check_d1_bndry( Grid_2D_Chebyshev(3, 5),
             lambda x, y: (exp(-y), exp(x+y)) ,
             lambda x, y: (exp(-y) + exp(x+y)) )
 
 
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
    assert( check(g, Grid_2D_Interface) )
    test0(g, f)
    test1(g, u, v)
 
    g = Grid_2D_Periodic(3, 5)
    test0(g, f)
    test1(g, u, v)
 
def test_contraction():
    from dec.symbolic import (x, y, contractions, lambdify2, sin, cos)
 
#     X = (0*x + 1, 0*x + 1)
#     alpha = (sin(x)*sin(y), cos(y))
#     Omega = 0*x + 1
 
#     X = (sin(x)*sin(y), cos(x))
#     alpha = (sin(x)*sin(y), cos(y))
#     Omega = sin(y)
 
    X = (sin(x), 0*x)
    alpha = (sin(x)*sin(y), cos(y))
    Omega = 0*x + 1
     
    l0, l1 = lambdify2()
    c1, c2 = contractions(X)
 
    G = Grid_2D_Periodic(6, 6)
    P0, P1, P2, P0d, P1d, P2d = G.projection()
    C1, C2 = G.contraction(P1(l1(X)))
     
    eq2(C1(P1(l1(alpha))), P0(l0(c1(alpha))))
    eq2(C2(P2(l0(Omega))), P1(l1(c2(Omega))))
