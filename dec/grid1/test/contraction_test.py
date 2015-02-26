from dec.grid1 import *
from numpy import *
from numpy.testing import assert_array_almost_equal as eq

def test_contraction_periodic():

    g = Grid_1D_Periodic(14)
    P0, P1, P0d, P1d = g.projection()

    def v(x): return sin(4 * x)
    def a(x): return cos(4 * x)
    def b(x): return a(x)*v(x) 

    V = P1(v)
    alpha1 = P1(a)
    beta0 = P0(b)

    C1 = g.contraction(V)
    eq(C1(alpha1), beta0)

def test_contraction_chebyshev():
 
    g = Grid_1D_Chebyshev(14)
    P0, P1, P0d, P1d = g.projection()
 
    def v(x): return x**2
    def a(x): return 2*x
    def b(x): return a(x)*v(x) 
 
    V = P1(v)
    alpha1 = P1(a)
    beta0 = P0(b)
 
    C1 = g.contraction(V)
    eq(C1(alpha1), beta0)

    def v(x): return x**2
    def a(x): return x**6
    def b(x): return a(x)*v(x) 
 
    V = P1(v)
    alpha1 = P1(a)
    beta0 = P0(b)
 
    C1 = g.contraction(V)
    eq(C1(alpha1), beta0)
