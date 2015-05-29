from dec.grid2 import *
import itertools
import collections
from numpy.testing import assert_array_almost_equal as eq

def test_N():

    def check_N(g, f, df):
        P = g.dec.P
        assert P[0, True](f).shape[0]   == g.N[0, True]
        assert P[1, True](df).shape[0]  == g.N[1, True]
        assert P[0, False](f).shape[0]  == g.N[0, False]
        assert P[1, False](df).shape[0] == g.N[1, False]
 
    check_N(Grid_2D.periodic(4, 3),
             lambda x, y: sin(x) ,
             lambda x, y: (cos(x),0) )

    check_N(Grid_2D.periodic(3, 5),
             lambda x, y: sin(x) ,
             lambda x, y: (cos(x),0) )

    check_N(Grid_2D.chebyshev(2, 3),
             lambda x, y: x,
             lambda x, y: (-y, x) )

    check_N(Grid_2D.periodic(5, 4),
             lambda x, y: x,
             lambda x, y: (-y, x) )

def test_basis_functions():

    def check_grid(g):
        for P, B in zip(g.projection(), g.basis_fn()):
            eq( vstack(P(b) for b in B), eye(len(B)) )

    for n in (2,3):
        check_grid(Grid_2D.periodic(n, n))
        #check_grid(Grid_2D.chebyshev(n, n+1))

def test_d():
 
    def check_d0(g, f, df):
        D0, D1, D0d, D1d = g.derivative()
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        eq( D0(P0(f)), P1(df))
        eq(D0d(P0d(f)), P1d(df))
 
    check_d0(Grid_2D.periodic(4, 3),
             lambda x, y: sin(x) ,
             lambda x, y: (cos(x),0) )
 
    check_d0(Grid_2D.periodic(3, 5),
             lambda x, y: sin(x)*cos(y) ,
             lambda x, y: (cos(x)*cos(y),-sin(x)*sin(y)) )
 
    def check_d1(g, f, df):
        D0, D1, D0d, D1d = g.derivative()
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        eq(D1(P1(f)), P2(df))
        eq(D1d(P1d(f)), P2d(df))
 
    check_d1( Grid_2D.periodic(5, 6),
             lambda x, y: (sin(x), sin(y)) ,
             lambda x, y: 0 )
 
    check_d1( Grid_2D.periodic(5, 6),
             lambda x, y: (-sin(y), sin(x)) ,
             lambda x, y: (cos(x) + cos(y)) )
 
    def check_d0_bndry(g, f, df):
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        D0, D1, D0d, D1d = g.derivative()
        BC0, BC1 = g.boundary_condition() 
        eq( D0(P0(f)), P1(df))
        eq( D0d(P0d(f)) + BC0(f), P1d(df) )
  
    check_d0_bndry(Grid_2D.chebyshev(3, 5),
             lambda x, y: x*y ,
             lambda x, y: (y,x) )
  
    check_d0_bndry(Grid_2D.chebyshev(3, 3),
             lambda x, y: x ,
             lambda x, y: (1,0) )
  
    def check_d1_bndry(g, f, df):
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        D0, D1, D0d, D1d = g.derivative()
        BC0, BC1 = g.boundary_condition() 
        eq(D1(P1(f)), P2(df))
        eq(D1d(P1d(f)) + BC1(f), P2d(df))
  
    check_d1_bndry( Grid_2D.chebyshev(3, 5),
             lambda x, y: (-sin(y), sin(x)) ,
             lambda x, y: (cos(x) + cos(y)) )
  
    check_d1_bndry( Grid_2D.chebyshev(3, 5),
             lambda x, y: (exp(-y), exp(x+y)) ,
             lambda x, y: (exp(-y) + exp(x+y)) )

def test_hodge():
  
    def test0(g, f):
        H0, H1, H2, H0d, H1d, H2d = g.hodge_star()
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        eq(H0(P0(f)), P2d(f))
        eq(H2(P2(f)), P0d(f))
        eq(H0d(P0d(f)), P2(f))
        eq(H2d(P2d(f)), P0(f))
  
    def test1(g, u, v):
        H0, H1, H2, H0d, H1d, H2d = g.hodge_star()
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        eq(H1(P1(lambda x, y: (u(x,y), v(x,y)))),
           P1d(lambda x, y: (-v(x,y), u(x,y))))
        eq(H1d(P1d(lambda x, y: (-v(x,y), u(x,y)))),
           P1(lambda x, y: (-u(x,y), -v(x,y))))
  
    f = lambda x, y: sin(x)*sin(y)
    u = lambda x, y: sin(x)*sin(y)
    v = lambda x, y: cos(y)
  
    g = Grid_2D.chebyshev(9, 9)
    test0(g, f)
    test1(g, u, v)
  
    g = Grid_2D.periodic(3, 5)
    test0(g, f)
    test1(g, u, v)
