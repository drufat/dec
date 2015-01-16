from dec.spectral import *

Grid_1D_Interface = '''
    dimension
    xmin xmax 
    delta delta_dual
    verts verts_dual
    edges edges_dual
    basis_fn
    projection
    derivative
    hodge_star
'''.split()

class Grid_1D_Periodic:

    def __init__(self, n, xmin=0, xmax=2*pi):
        assert xmax > xmin

        dimension = 1

        pnts = linspace(xmin, xmax, num=(n+1))
        lenx = abs(xmax - xmin)
        delta = diff(pnts)

        verts = pnts[:-1]
        edges = (pnts[:-1], pnts[1:])

        verts_dual = verts + 0.5*delta
        edges_dual = (roll(verts_dual,shift=1), verts_dual)
        edges_dual[0][0] -= lenx
        delta_dual = delta

        V = verts
        S0 = arange(len(V))
        S1 = (S0[:-1], S0[1:])

        Vd = verts_dual
        S0d = arange(len(Vd))
        S1d = (S0d[:-1], S0d[1:])

        self.dimension = dimension
        self.n = n
        self.xmin = xmin
        self.xmax = xmax
        self.delta = delta
        self.delta_dual = delta_dual
        self.pnts = pnts
        self.verts = verts
        self.verts_dual = verts_dual
        self.edges = edges
        self.edges_dual = edges_dual

    def projection(self):
        P0 = lambda f: f(self.verts)
        P1 = lambda f: split_args(integrate_spectral)(
                        self.edges[0], self.edges[1], f)
        P0d = lambda f: f(self.verts_dual)
        P1d = lambda f: split_args(integrate_spectral)(
                        self.edges_dual[0], self.edges_dual[1], f)
        return P0, P1, P0d, P1d

    def basis_fn(self):
        n = self.n
        B0 = [lambda x, i=i: phi0(n, i, x) for i in range(n)]
        B1 = [lambda x, i=i: phi1(n, i, x) for i in range(n)]
        B0d = [lambda x, i=i: phid0(n, i, x) for i in range(n)]
        B1d = [lambda x, i=i: phid1(n, i, x) for i in range(n)]
        return B0, B1, B0d, B1d

    def reconstruction(self):
        R0, R1, R0d, R1d = reconstruction(self.basis_fn())
        return R0, R1, R0d, R1d

    def derivative(self):
        D0  = lambda f: roll(f, shift=-1) - f
        D0d = lambda f: roll(D0(f), shift=+1)
        return D0, D0d

    def hodge_star(self):
        H0 = lambda x: real(H(x))
        H1 = lambda x: real(Hinv(x))
        H0d = H0
        H1d = H1
        return H0, H1, H0d, H1d

    def derivative_matrix(self):
        n = self.n
        rng = arange(n)
        ons = ones(n)
        d = row_stack((
                   column_stack((
                     rng,
                     roll(rng, shift= -1),
                     +ons)),
                   column_stack((
                     rng,
                     rng,
                     -ons))
                   ))
        D = sparse_matrix(d, n, n)
        return D, -D.T

    def differentiation_toeplitz(self):
        raise NotImplemented
        #TODO: fix this implementation
        n = self.n
        h = 2*pi/n
        assert n % 2 == 0
        column = concatenate(( [ 0 ], (-1)**arange(1,n) / tan(arange(1,n)*h/2)  ))
        row = concatenate(( column[:1], column[1:][::-1] ))
        D = toeplitz(column, row)
        return D

    def hodge_star_toeplitz(self):
        '''
        The Hodge-Star using a Toeplitz matrix.
        '''
        P0, P1, P0d, P1d = self.projection()
        column = P1d(lambda x: alpha0(self.n, x))
        row = concatenate((column[:1], column[1:][::-1]))
        return toeplitz(column, row)

    def wedge(self):
        '''
        Return \alpha ^ \beta. Keep only for primal forms for now.
        '''
        def w00(alpha, beta):
            return alpha * beta
        def _w01(alpha, beta):
            return S(H( alpha * Hinv(Sinv(beta)) ))
        def w01(alpha, beta):
#            a = interweave(alpha, T(alpha, [S]))
#            b = interweave(T(beta, [Hinv, Sinv]), T(beta, [Hinv]))
            a = refine(alpha)
            b = refine(Hinv(Sinv(beta)))
            c = S(H(a * b))
            return c[0::2] + c[1::2]
        return w00, w01, _w01

    def contraction(self, V):
        '''
        Return i_V. Keep only for primal forms for now.
        '''
        def C1(alpha):
            return Hinv(Sinv(V)) * Hinv(Sinv(alpha))
        return C1


class Grid_1D_Regular:
    '''
    Keep symmetric bases functions for 1-forms, so that the hodge star operators below are actually
    the correct ones. 
    '''
    
    def __init__(self, n, xmin=0, xmax=pi):
    
        assert xmax > xmin
    
        dimension = 1
    
        N = n
        pnts = linspace(xmin, xmax, num=n)
        lenx = abs(xmax - xmin)
        delta = diff(pnts)
    
        verts = pnts
        verts_dual = verts[:-1] + 0.5*delta
    
        edges = (pnts[:-1], pnts[1:])
        tmp = concatenate(([xmin], verts_dual, [xmax]))
        delta_dual = diff(tmp)
        edges_dual = (tmp[:-1], tmp[1:])
        
        self.dimension = dimension
        self.n = n
        self.xmin = xmin
        self.xmax = xmax
        self.delta = delta
        self.delta_dual = delta_dual
        self.pnts = pnts
        self.verts = verts
        self.verts_dual = verts_dual
        self.edges = edges
        self.edges_dual = edges_dual
        
    def projection(self):
        P0 = lambda f: f(self.verts)
        P1 = lambda f: slow_integration(self.edges[0], self.edges[1], f)
        P0d = lambda f: f(self.verts_dual)
        P1d = lambda f: slow_integration(self.edges_dual[0], self.edges_dual[1], f)
        return P0, P1, P0d, P1d

    def basis_fn(self):
        n = self.n
        B0 = [lambda x, i=i: kappa0(n, i, x) for i in range(n)]
        B1 = [lambda x, i=i: kappa1_symm(n, i, x) for i in range(n-1)]
        B0d = [lambda x, i=i: kappad0(n, i, x) for i in range(n-1)]
        B1d = [lambda x, i=i: kappad1_symm(n, i, x) for i in range(n)]
        return B0, B1, B0d, B1d

    def reconstruction(self):
        R0, R1, R0d, R1d = reconstruction(self.basis_fn())
        return R0, R1, R0d, R1d

    def derivative(self):
        D0 = lambda f: diff(f)
        D0d = lambda f: diff(concatenate(([0], f, [0])))
        return D0, D0d

    def hodge_star(self):
        H0 = H0_regular
        H1 = H1_regular
        H0d = H0d_regular
        H1d = H1d_regular
        return H0, H1, H0d, H1d

class Grid_1D_Chebyshev:
    
    def __init__(self, n, xmin=-1, xmax=+1):

        N = n
        dimension = 1
    
        # 2n-1 points: n primal, n-1 dual
        x = sin(linspace(-pi/2, pi/2, 2*n-1))
        p = 0.5*(xmin*(1-x) + xmax*(1+x))
    
        verts = p[::2]
        delta = diff(verts)
        edges = (verts[:-1], verts[1:])
    
        verts_dual = p[1::2]
        tmp = concatenate(([p[0]], verts_dual, [p[-1]]))
        delta_dual = diff(tmp)
        edges_dual = (tmp[:-1], tmp[1:])
    
        V = verts
        S0 = arange(len(V))
        S1 = (S0[:-1], S0[1:])
    
        Vd = verts_dual
        S0d = arange(len(Vd))
        S1d = (S0d[:-1], S0d[1:])

        self.dimension = dimension
        self.n = n
        self.xmin = xmin
        self.xmax = xmax
        self.delta = delta
        self.delta_dual = delta_dual
        self.pnts = p
        self.verts = verts
        self.verts_dual = verts_dual
        self.edges = edges
        self.edges_dual = edges_dual

    def projection(self):
        P0 = lambda f: f(self.verts)
        P1 = lambda f: slow_integration(self.edges[0], self.edges[1], f)
        #P1 = lambda f: integrate_chebyshev(self.verts, f)
        P0d = lambda f: f(self.verts_dual)
        #P1d = lambda f: integrate_chebyshev_dual(
        #        concatenate(([-1], self.verts_dual, [+1])), f)
        P1d = lambda f: slow_integration(self.edges_dual[0], self.edges_dual[1], f)
        return P0, P1, P0d, P1d

    def basis_fn(self):
        n = self.n
        B0  = [lambda x, i=i:  psi0(n, i, x) for i in range(n)]
        B1  = [lambda x, i=i:  psi1(n, i, x) for i in range(n-1)]
        B0d = [lambda x, i=i: psid0(n, i, x) for i in range(n-1)]
        B1d = [lambda x, i=i: psid1(n, i, x) for i in range(n)]
        return B0, B1, B0d, B1d

    def reconstruction(self):
        R0, R1, R0d, R1d = reconstruction(self.basis_fn())
        return R0, R1, R0d, R1d

    def boundary_condition(self, f):
        bc = zeros((self.n, ))
        bc[ 0] = -f(self.xmin)
        bc[-1] = +f(self.xmax)
        return bc

    def derivative(self):
        D0 = lambda f: diff(f)
        D0d = lambda f: diff(concatenate(([0], f, [0])))
        return D0, D0d

    def hodge_star(self):
        H0 = H0_cheb
        H1 = H1_cheb
        H0d = H0d_cheb
        H1d = H1d_cheb
        return H0, H1, H0d, H1d

    def contraction(self, V):
        '''
        Implement contraction where V is the one-form corresponding to the vector field.
        '''
        def C1(alpha):
            return None
        return C1
