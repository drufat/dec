from numpy.testing import *
from dec.spectral import *
import itertools

eq = assert_array_almost_equal

def test_transforms():
    x = random.random(11)

    eq( x,          Sinv(S(x)) )
    eq( Hinv(x),    S(Hinv(Sinv(x))) )
    eq( H(S(x)),    S(H(x)) )

    x = Grid_1D_Periodic(12).verts
    h = diff(x)[0]

    f = lambda x: sin(x)
    fp = lambda x: cos(x)

    eq( H(fp(x)),  f(x+h/2) - f(x-h/2) )
    eq( fp(x), Hinv(f(x+h/2) - f(x-h/2)) )

    eq( I(fp(x), -1, 1),  f(x+1) - f(x-1) )
    eq( I(fp(x),  0, 1),  f(x+1) - f(x) )
    eq( fp(x), Iinv(f(x+1) - f(x-1), -1, 1) )
    eq( fp(x), Iinv(f(x+1) - f(x  ),  0, 1) )

    eq( S(    f(x)), f(x+h/2) )
    eq( Sinv( f(x)), f(x-h/2) )

def test_type_system():
    g = Grid_1D_Periodic(11)

    P, R, D, H = DEC(g)
    f = lambda x: sin(sin(x))

    eq( P(f, 0, True ), g.P0(f)  )
    eq( P(f, 1, True ), g.P1(f)  )
    eq( P(f, 0, False), g.P0d(f) )
    eq( P(f, 1, False), g.P1d(f) )

    x = linspace(0, 2*pi, 100)
    eq( R(P(f, 0, True ))(x), g.R0(g.P0(f))(x) )
    eq( R(P(f, 1, True ))(x), g.R1(g.P1(f))(x)  )
    eq( R(P(f, 0, False))(x), g.R0d(g.P0d(f))(x) )
    eq( R(P(f, 1, False))(x), g.R1d(g.P1d(f))(x) )

    x = P(f, 0, True)
    eq( D(x), g.D0(x) )
    x = P(f, 0, False)
    eq( D(x), g.D0d(x) )

    x = P(f, 0, True)
    eq( H(x), g.H0(x) )
    x = P(f, 1, True)
    eq( H(x), g.H1(x) )

def test_d_equivalence():
    g = Grid_1D_Periodic(10)
    F = lambda x: exp(sin(13*x))

    M = g.derivative_matrix()
    d, dd = [(lambda f: M[0]*f), (lambda f: M[1]*f)]
    D, DD = g.derivative()

    f = g.P0(F)
    eq(D(f), d(f))
    f = g.P0d(F)
    eq(DD(f), dd(f))

def test_one_form():

    N = 10; h = 2*pi/N
    g = Grid_1D_Periodic(N)
    eq(g.P1(sin),
       g.P1d(lambda x: sin(x+.5*h)))
    for i, j in itertools.combinations((integrate_boole1,
                                        integrate_spectral_coarse,
                                        integrate_spectral), 2):
        for x0, x1 in (g.edges, g.edges_dual):
                x = concatenate((x0, [x1[-1]]))
                for f in (sin, cos, (lambda x: sin(sin(x)))):
                    eq(i(x, f), j(x, f))

def test_integrals():

    for N in (10, 11, 12, 13):

        x = Grid_1D_Periodic(N, 0, 2*pi).pnts
        for f in (sin,
                  cos):
            for I, J in itertools.combinations((
                        integrate_boole1,
                        integrate_spectral_coarse,
                        integrate_spectral), 2):
                eq(I(x, f), J(x, f))

        x = Grid_1D_Chebyshev(N, -1, +1).verts
        for f in ((lambda x: x),
                  (lambda x: x**3),
                  (lambda x: exp(x))):
            for I, J in itertools.combinations((
                        integrate_boole1,
                        integrate_chebyshev), 2):
                eq(I(x, f), J(x, f))

        x = Grid_1D_Chebyshev(N, -1, +1).verts_dual
        x = concatenate(([-1], x, [+1]))
        for f in ((lambda x: x),
                  (lambda x: x**3),
                  (lambda x: exp(x))):
            for I, J in itertools.combinations((
                        integrate_boole1,
                        integrate_chebyshev_dual), 2):
                eq(I(x, f), J(x, f))

def test_basis_functions():

    def check_periodic(n):
        g = Grid_1D_Periodic(n, 0, 2*pi)
        for P, B in zip(g.projection(), g.basis_fn()):
            eq( vstack(P(b) for b in B), eye(len(B)) )

    def check_regular(n):
        g = Grid_1D_Regular(n, 0, pi)
        for P, B in zip(g.projection(), g.basis_fn()):
            eq( vstack(P(b) for b in B), eye(len(B)) )

    def check_chebyshev(n):
        g = Grid_1D_Chebyshev(n, -1, +1)
        for P, B in zip(g.projection(), g.basis_fn()):
            eq( vstack(P(b) for b in B), eye(len(B)) )

    for n in (2, 3, 4):
        check_periodic(n)
        check_regular(n)
        check_chebyshev(n)

def test_hodge_star_inv():

    def check_equal(g):
        H0, H1, H0d, H1d = [to_matrix(H, len(g.verts)) for H in g.hodge_star()]

        eq( H1d, linalg.inv(H0) )
        eq( H0d, linalg.inv(H1) )

    for n in range(3,7):
        check_equal(Grid_1D_Periodic(n))
        check_equal(Grid_1D_Regular(n))
        check_equal(Grid_1D_Chebyshev(n))

def test_hodge_star_basis_fn():

    def check_equal(g):
        for A, B in zip(hodge_star_matrix(g.projection(), g.basis_fn()),
                          g.hodge_star()) :
            eq( A, to_matrix(B, n) )

    for n in range(2,7):
        check_equal(Grid_1D_Periodic(n, 0, 2*pi))

    for n in range(2,7):
        g = Grid_1D_Chebyshev(n, -1, +1)
        H0, H1, H0d, H1d = hodge_star_matrix(g.projection(), g.basis_fn())

        eq(H0d, to_matrix(g.H0d, n-1))
        eq(H1, to_matrix(g.H1, n-1))

        eq(H0, to_matrix(g.H0, n))
        eq(H1d, to_matrix(g.H1d, n))

def test_compare_chebyshev_and_lagrange_polynomials():
    """
    The Chebyshev basis functions are equivalent to the
    Lagrange basis functions.
    """

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
        eq(D(g.P0(f)), g.P1(f_prime))
        eq(DD(g.P0d(f)), g.P1d(f_prime))
        #dd = g.differentiation_toeplitz()
        #eq( dd*g.P0(f), g.P0(f_prime) )

    check_d( Grid_1D_Periodic(10), sin, cos )
    check_d( Grid_1D_Periodic(11), sin, cos )
    check_d( Grid_1D_Periodic(10), cos, (lambda x: -sin(x)) )


    def check_d_bnd(g, f, f_prime):
        D, DD = g.derivative()
        eq(D(g.P0(f)), g.P1(f_prime))
        bc = g.boundary_condition(f)
        eq( DD(g.P0d(f))+bc, g.P1d(f_prime) )

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
#    """ Associativity satisfied by exact forms. """
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
    """ Associativity satisfied only by wedge with no refinement."""

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

def test_contraction():

    g = Grid_1D_Periodic(14)
    P0, P1, P0d, P1d = g.projection()

    V = P1(lambda x: sin(4 * x))
    alpha1 = P1(lambda x: cos(4 * x))
    beta0 = P0(lambda x: sin(4 * x) * cos(4 * x))

    C1 = g.contraction(V)
    eq(C1(alpha1), beta0)
