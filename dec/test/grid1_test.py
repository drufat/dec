from dec.grid1 import *
from numpy.testing import assert_array_almost_equal
import dec.forms

eq = assert_array_almost_equal

# def test_DEC():
#     g = Grid_1D_Periodic(11)
#
#     P, R, D, H = dec.forms.DEC(g)
#
#     P0, P1, P0d, P1d = g.projection()
#     R0, R1, R0d, R1d = g.reconstruction()
#     H0, H1, H0d, H1d = g.hodge_star()
#     D0, D0d = g.derivative()
#
#     f = lambda x: sin(sin(x))
#
#     eq( P(f, 0, True ), P0(f)  )
#     eq( P(f, 1, True ), P1(f)  )
#     eq( P(f, 0, False), P0d(f) )
#     eq( P(f, 1, False), P1d(f) )
#
#     x = linspace(0, 2*pi, 100)
#     eq( R(P(f, 0, True ))(x), R0(P0(f))(x) )
#     eq( R(P(f, 1, True ))(x), R1(P1(f))(x)  )
#     eq( R(P(f, 0, False))(x), R0d(P0d(f))(x) )
#     eq( R(P(f, 1, False))(x), R1d(P1d(f))(x) )
#
#     x = P(f, 0, True)
#     eq( D(x), D0(x) )
#     x = P(f, 0, False)
#     eq( D(x), D0d(x) )
#
#     x = P(f, 0, True)
#     eq( H(x), H0(x) )
#     x = P(f, 1, True)
#     eq( H(x), H1(x) )

def test_d_equivalence():
    g = Grid_1D_Periodic(10)
    F = lambda x: exp(sin(13*x))

    M = g.derivative_matrix()
    d, dd = [(lambda f: M[0]*f), (lambda f: M[1]*f)]
    D, DD = g.derivative()

    P0, P1, P0d, P1d = g.projection()
    f = P0(F)
    eq(D(f), d(f))
    f = P0d(F)
    eq(DD(f), dd(f))

def test_one_form():

    N = 10; h = 2*pi/N
    g = Grid_1D_Periodic(N)
    P0, P1, P0d, P1d = g.projection()
    eq(P1(sin),
       P1d(lambda x: sin(x+.5*h)))
    for i, j in itertools.combinations((integrate_boole1,
                                        integrate_spectral_coarse,
                                        integrate_spectral), 2):
        for x0, x1 in (g.edges, g.edges_dual):
                x = concatenate((x0, [x1[-1]]))
                for f in (sin, cos, (lambda x: sin(sin(x)))):
                    eq(i(x, f), j(x, f))

def test_integrals():
    
    for N in (10, 11, 12, 13):

        g = Grid_1D_Periodic(N, 0, 2*pi)
        for f in (sin,
                  cos):
            reference = slow_integration(g.edges[0], g.edges[1], f)
            eq( integrate_boole1(g.pnts, f), reference )
            eq( integrate_spectral_coarse(g.pnts, f), reference )
            eq( integrate_spectral(g.pnts, f), reference )

        g = Grid_1D_Chebyshev(N, -1, +1)
        for f in ((lambda x: x),
                  (lambda x: x**3),
                  (lambda x: exp(x))):
            reference = slow_integration(g.edges[0], g.edges[1], f)
            eq( integrate_boole1(g.verts, f), reference )
            eq( integrate_chebyshev(g.verts, f), reference )

        g = Grid_1D_Chebyshev(N, -1, +1)
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
        check_grid(Grid_1D_Periodic(n, 0, 2*pi))
        check_grid(Grid_1D_Regular(n, 0, pi))
        check_grid(Grid_1D_Chebyshev(n, -1, +1))

def test_projection_reconstruction():
    
    random.seed(seed=1)
    
    def check_grid(g):
        for P, R, B in zip(g.projection(), g.reconstruction(), g.basis_fn()):
            y = random.rand(len(B))
            eq( P(R(y)), y )

    for n in (2, 3, 4):
        check_grid(Grid_1D_Periodic(n, 0, 2*pi))
        check_grid(Grid_1D_Regular(n, 0, pi))
        check_grid(Grid_1D_Chebyshev(n, -1, 1))

def test_hodge_star_basis_fn():

    for n in range(2,4):
        g = Grid_1D_Periodic(n)
        H0, H1, H0d, H1d = hodge_star_matrix(g.projection(), g.basis_fn())
        h0, h1, h0d, h1d = g.hodge_star()

        eq(H0d, to_matrix(h0d, n))
        eq(H1, to_matrix(h1, n))

        eq(H0, to_matrix(h0, n))
        eq(H1d, to_matrix(h1d, n))

    for n in range(2,5):
        g = Grid_1D_Regular(n, 0, pi)
        H0, H1, H0d, H1d = hodge_star_matrix(g.projection(), g.basis_fn())
        h0, h1, h0d, h1d = g.hodge_star()
   
        eq(H0d, to_matrix(h0d, n-1))
        eq(H1, to_matrix(h1, n-1))
   
        eq(H0, to_matrix(h0, n))
        eq(H1d, to_matrix(h1d, n))

    for n in range(2,4):
        g = Grid_1D_Chebyshev(n, -1, +1)
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
        check_equal(Grid_1D_Periodic(n))
        check_equal(Grid_1D_Regular(n))
        check_equal(Grid_1D_Chebyshev(n))

def test_compare_chebyshev_and_lagrange_polynomials():
    '''
    The Chebyshev basis functions are equivalent to the
    Lagrange basis functions.
    '''

    for n in (10, 11, 12, 13):

        g = Grid_1D_Chebyshev(n, -1, +1)

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

    check_d( Grid_1D_Periodic(10), sin, cos )
    check_d( Grid_1D_Periodic(11), sin, cos )
    check_d( Grid_1D_Periodic(10), cos, (lambda x: -sin(x)) )


    def check_d_bnd(g, f, f_prime):
        D, DD = g.derivative()
        P0, P1, P0d, P1d = g.projection()
        eq(D(P0(f)), P1(f_prime))
        bc = g.boundary_condition(f)
        eq( DD(P0d(f))+bc, P1d(f_prime) )

    check_d_bnd( Grid_1D_Chebyshev(10), (lambda x: 2*x), (lambda x: 2+0*x) )
    check_d_bnd( Grid_1D_Chebyshev(11), (lambda x: 2*x), (lambda x: 2+0*x) )
    check_d_bnd( Grid_1D_Chebyshev(10), (lambda x: x**3), (lambda x: 3*x**2) )
    check_d_bnd( Grid_1D_Chebyshev(11), (lambda x: x**3), (lambda x: 3*x**2) )

def test_hodge():

    F = lambda x: sin(4 * x)
    g = Grid_1D_Periodic(10)
    H0, H1, _, _ = g.hodge_star()
    P0, P1, P0d, P1d = g.projection()

    eq(H0(P0(F)), P1d(F))
    eq(H1(P1(F)), P0d(F))

    eq(H0(P0(F)), dot(g.hodge_star_toeplitz(), P0(F)))
    eq(H0(P0(F)), P1d(F))
    eq(H1(P1(F)), P0d(F))

def test_wedge():

    def alpha(x): return sin(4 * x)
    def beta(x): return cos(4 * x)
    def gamma(x): return sin(4 * x) * cos(4 * x)

    g = Grid_1D_Periodic(13)
    P0, P1, P0d, P1d = g.projection()
    W00, W01, _W01 = g.wedge()

    a0 = P0(alpha)
    b0 = P0(beta)
    c0 = P0(gamma)
    eq(W00(a0, b0), c0)

    a0 = P0(alpha)
    b1 = P1(beta)
    c1 = P1(gamma)
    eq(W01(a0, b1), c1)
    #eq(_w01(alpha0, beta1), gamma1)

def test_leibniz():

    N = 13
    g = Grid_1D_Periodic(N)
    D, DD = g.derivative()
    P0, P1, P0d, P1d = g.projection()
    W00, W01, _W01 = g.wedge()

    a0 = random.random_sample(N)
    b0 = random.random_sample(N)

    lhs = D(W00(a0, b0))
    rhs = W01(a0, D(b0)) + W01(b0, D(a0))
    eq(lhs, rhs)

#def test_associativity_exact():
#    ''' Associativity satisfied by exact forms. '''
#
#    N = 5
#    g = Grid_1D_Periodic(N)
#    W00, W01, _W01 = g.wedge()
#    D, _ = g.derivative()
#
#    a0 = random.random_sample(N)
#    b0 = random.random_sample(N)
#    c1 = D(random.random_sample(N))
#
#    eq1 = W01(a0, W01(b0, c1))
#    eq2 = W01(b0, W01(a0, c1))
#    eq3 = W01(W00(a0, b0), c1)
#
#    eq(eq1, eq3)
#    eq(eq2, eq3)

def test_associativity_old():
    ''' Associativity satisfied only by wedge with no refinement.'''

    N = 13
    g = Grid_1D_Periodic(N)
    W00, W01, _W01 = g.wedge()

    W01 = _W01 #Use old wedge

    a0 = random.random_sample(N)
    b0 = random.random_sample(N)
    c1 = random.random_sample(N)

    eq1 = W01(a0, W01(b0, c1))
    eq2 = W01(b0, W01(a0, c1))
    eq3 = W01(W00(a0, b0), c1)

    eq(eq1, eq2)
    eq(eq2, eq3)

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

# def test_contraction_chebyshev():
# 
#     g = Grid_1D_Chebyshev(14)
#     P0, P1, P0d, P1d = g.projection()
# 
#     def v(x): return x**2
#     def a(x): return 2*x
#     def b(x): return a(x)*v(x) 
# 
#     V = P1(v)
#     alpha1 = P1(a)
#     beta0 = P0(b)
# 
#     C1 = g.contraction(V)
#     eq(C1(alpha1), beta0)
