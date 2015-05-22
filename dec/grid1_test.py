from dec.spectral import *
from dec.grid1 import *
from numpy.testing import assert_array_almost_equal as eq
import pytest
random.seed(seed=1)

def test_integrals():
    
    for N in (10, 11, 12, 13):

        g = Grid_1D.periodic(N, 0, 2*pi)
        pnts = concatenate([g.verts, [g.xmax]])
        for f in (sin,
                  cos):
            reference = slow_integration(g.edges[0], g.edges[1], f)
            eq( integrate_boole1(pnts, f), reference )
            eq( integrate_spectral_coarse(pnts, f), reference )
            eq( integrate_spectral(pnts, f), reference )

        g = Grid_1D.chebyshev(N, -1, +1)
        for f in ((lambda x: x),
                  (lambda x: x**3),
                  (lambda x: exp(x))):
            reference = slow_integration(g.edges[0], g.edges[1], f)
            eq( integrate_boole1(g.verts, f), reference )
            eq( integrate_chebyshev(g.verts, f), reference )

        g = Grid_1D.chebyshev(N, -1, +1)
        for f in ((lambda x: x),
                  (lambda x: x**3),
                  (lambda x: exp(x))):
            reference = slow_integration(g.edges_dual[0], g.edges_dual[1], f)
            x = concatenate(([-1], g.verts_dual, [+1]))
            eq( integrate_boole1(x, f), reference )
            eq( integrate_chebyshev_dual(x, f), reference )

def test_basis_functions():

    def check_grid(g):
        for P, B in zip(g.projection(), g.basis_fn()):
            eq( vstack(P(b) for b in B), eye(len(B)) )

    for n in range(3, 6):
        check_grid(Grid_1D.periodic(n, 0, 2*pi))
        check_grid(Grid_1D.chebyshev(n, -1, +1))
        check_grid(Grid_1D.regular(n, 0, pi))

def test_projection_reconstruction():
        
    def check_grid(g):
        for P, R, B in zip(g.projection(), g.reconstruction(), g.basis_fn()):
            y = random.rand(len(B))
            eq( P(R(y)), y )

    for n in (2, 3, 4):
        check_grid(Grid_1D.periodic(n, 0, 2*pi))
        check_grid(Grid_1D.regular(n, 0, pi))
        check_grid(Grid_1D.chebyshev(n, -1, 1))

def test_hodge_star_basis_fn():

    for n in range(2,4):
        g = Grid_1D.periodic(n)
        H0, H1, H0d, H1d = hodge_star_matrix(g.projection(), g.basis_fn())
        h0, h1, h0d, h1d = g.hodge_star()

        eq(H0d, to_matrix(h0d, n))
        eq(H1, to_matrix(h1, n))

        eq(H0, to_matrix(h0, n))
        eq(H1d, to_matrix(h1d, n))

    for n in range(2,5):
        g = Grid_1D.regular(n, 0, pi)
        H0, H1, H0d, H1d = hodge_star_matrix(g.projection(), g.basis_fn())
        h0, h1, h0d, h1d = g.hodge_star()
   
        eq(H0d, to_matrix(h0d, n-1))
        eq(H1, to_matrix(h1, n-1))
   
        eq(H0, to_matrix(h0, n))
        eq(H1d, to_matrix(h1d, n))

    for n in range(2,4):
        g = Grid_1D.chebyshev(n, -1, +1)
        H0, H1, H0d, H1d = hodge_star_matrix(g.projection(), g.basis_fn())
        h0, h1, h0d, h1d = g.hodge_star()
 
        eq(H0d, to_matrix(h0d, n-1))
        eq(H1, to_matrix(h1, n-1))
  
        eq(H0, to_matrix(h0, n))
        eq(H1d, to_matrix(h1d, n))

def test_hodge_star_inv():

    def check_equal(g):
        h0, h1, h0d, h1d = g.hodge_star()
        H0 = to_matrix(h0, len(g.verts))
        H1 = to_matrix(h1, len(g.delta))
        H0d = to_matrix(h0d, len(g.delta))
        H1d = to_matrix(h1d, len(g.verts))
        eq( H1d, linalg.inv(H0) )
        eq( H0d, linalg.inv(H1) )

    for n in range(3,7):
        check_equal(Grid_1D.periodic(n))
        check_equal(Grid_1D.regular(n))
        check_equal(Grid_1D.chebyshev(n))

def test_compare_chebyshev_and_lagrange_polynomials():
    '''
    The Chebyshev basis functions are equivalent to the
    Lagrange basis functions.
    '''

    for n in (10, 11, 12, 13):

        g = Grid_1D.chebyshev(n, -1, +1)

        x = linspace(g.xmin, g.xmax, 100)

        L = lagrange_polynomials(g.verts)
        for i in range(len(L)):
            eq(L[i](x), psi0(n, i, x))
        L = lagrange_polynomials(g.verts_dual)
        for i in range(len(L)):
            eq(L[i](x), psid0(n, i, x))

def test_d():

    def check_d(g, f, f_prime):
        D, DD = g.derivative()
        P0, P1, P0d, P1d = g.projection()
        eq(D(P0(f)), P1(f_prime))
        eq(DD(P0d(f)), P1d(f_prime))
        #dd = g.differentiation_toeplitz()
        #eq( dd*P0(f), P0(f_prime) )

    check_d( Grid_1D.periodic(10), sin, cos )
    check_d( Grid_1D.periodic(11), sin, cos )
    check_d( Grid_1D.periodic(10), cos, (lambda x: -sin(x)) )


    def check_d_bnd(g, f, f_prime):
        D, DD = g.derivative()
        P0, P1, P0d, P1d = g.projection()
        eq(D(P0(f)), P1(f_prime))
        bc = g.boundary_condition(f)
        eq( DD(P0d(f))+bc, P1d(f_prime) )

    check_d_bnd( Grid_1D.chebyshev(10), (lambda x: 2*x), (lambda x: 2+0*x) )
    check_d_bnd( Grid_1D.chebyshev(11), (lambda x: 2*x), (lambda x: 2+0*x) )
    check_d_bnd( Grid_1D.chebyshev(10), (lambda x: x**3), (lambda x: 3*x**2) )
    check_d_bnd( Grid_1D.chebyshev(11), (lambda x: x**3), (lambda x: 3*x**2) )

def test_hodge():

    F = lambda x: sin(4 * x)
    g = Grid_1D.periodic(10)
    H0, H1, _, _ = g.hodge_star()
    P0, P1, P0d, P1d = g.projection()

    eq(H0(P0(F)), P1d(F))
    eq(H1(P1(F)), P0d(F))

    #eq(H0(P0(F)), dot(g.hodge_star_toeplitz(), P0(F)))
    eq(H0(P0(F)), P1d(F))
    eq(H1(P1(F)), P0d(F))

def test_wedge():

    def α(x): return sin(4 * x)
    def β(x): return cos(4 * x)
    def γ(x): return sin(4 * x) * cos(4 * x)

    g = Grid_1D.periodic(13)
    P0, P1, P0d, P1d = g.projection()
    W = g.wedge()
    W00 = W[(0,True),(0,True), True]
    W01 = W[(0,True),(1,True), True]

    a0 = P0(α)
    b0 = P0(β)
    c0 = P0(γ)
    eq(W00(a0, b0), c0)

    a0 = P0(α)
    b1 = P1(β)
    c1 = P1(γ)
    eq(W01(a0, b1), c1)

def test_wedge_chebyshev():

    def α(x): return x**2
    def β(x): return (x+1/2)**3
    def γ(x): return x**2 * (x+1/2)**3

    g = Grid_1D.chebyshev(13)
    P0, P1, P0d, P1d = g.projection()
    W = g.wedge()
    W00 = W[(0,True),(0,True), True]
    W01 = W[(0,True),(1,True), True]

    a0 = P0(α)
    b0 = P0(β)
    c0 = P0(γ)
    eq(W00(a0, b0), c0)

    a0 = P0(α)
    b1 = P1(β)
    c1 = P1(γ)
    eq(W01(a0, b1), c1)

def test_leibniz():

    N = 7
    g = Grid_1D.periodic(N)
    D, DD = g.derivative()
    W = g.wedge()
    W00 = W[(0,True),(0,True), True]
    W01 = W[(0,True),(1,True), True]

    a0 = random.random_sample(N)
    b0 = random.random_sample(N)

    lhs = D(W00(a0, b0))
    rhs = W01(a0, D(b0)) + W01(b0, D(a0))
    eq(lhs, rhs)

@pytest.mark.xfail
def test_associativity_exact():
    ''' Associativity satisfied by exact forms. '''

    N = 5
    g = Grid_1D.periodic(N)
    W = g.wedge()
    W00 = W[(0,True),(0,True), True]
    W01 = W[(0,True),(1,True), True]
    D, _ = g.derivative()

    a0 = random.random_sample(N)
    b0 = random.random_sample(N)
    c1 = D(random.random_sample(N))

    eq1 = W01(a0, W01(b0, c1))
    eq2 = W01(b0, W01(a0, c1))
    eq3 = W01(W00(a0, b0), c1)

    eq(eq1, eq3)
    eq(eq2, eq3)
