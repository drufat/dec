from dec.spectral import *

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
        P1 = lambda f: slow_integration(self.edges[0], self.edges[1], f)
#         P1 = lambda f: split_args(integrate_spectral)(
#                         self.edges[0], self.edges[1], f)
        P0d = lambda f: f(self.verts_dual)
        P1d = lambda f: slow_integration(self.edges_dual[0], self.edges_dual[1], f)
#         P1d = lambda f: split_args(integrate_spectral)(
#                         self.edges_dual[0], self.edges_dual[1], f)
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

    def switch(self):
        '''
        Switch between primal and dual 0-forms. The operator is only invertible 
        if they have the same size.
        '''
        return S, Sinv

    def contraction(self, V):
        '''
        Return i_V. Keep only for primal forms for now.
        '''
        S, Sinv = self.switch()
        H0, H1, H0d, H1d = self.hodge_star()
        def C1(f): return Sinv(H1(V)) * Sinv(H1(f))
        return C1


