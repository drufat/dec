import numpy as np
import dec.spectral as sp
from dec.grid1 import projections, dec_operators
from collections import namedtuple

dec_operators = namedtuple('dec_operators', 'P B D H')

def periodic_fn(cls, n, xmin=0, xmax=2*np.pi):

    pnts = np.linspace(xmin, xmax, num=(n+1))
    lenx = abs(xmax - xmin)
    delta = np.diff(pnts)

    verts = pnts[:-1]
    edges = (pnts[:-1], pnts[1:])

    verts_dual = verts + 0.5*delta
    edges_dual = (np.roll(verts_dual,shift=1), verts_dual)
    edges_dual[0][0] -= lenx
    delta_dual = delta

    B0  = lambda i, x: sp.phi0(n, i, x)
    B1  = lambda i, x: sp.phi1(n, i, x)
    B0d = lambda i, x: sp.phid0(n, i, x)
    B1d = lambda i, x: sp.phid1(n, i, x)
    
    P0,  P1  = projections((verts, edges))
    P0d, P1d = projections((verts_dual, edges_dual))
    
    D0  = lambda f: np.roll(f, shift=-1) - f
    D1  = lambda f: 0 
    D0d = lambda f: np.roll(D0(f), shift=+1)
    D1d = lambda f: 0 

    H0 = lambda x: np.real(sp.H(x))
    H1 = lambda x: np.real(sp.Hinv(x))
    H0d = H0
    H1d = H1
    
    d = cls(n=n,
            xmin=xmin,
            xmax=xmax,
            pnts=pnts,
            delta=delta_dual, 
            N = (n, n),
            simp=(verts_dual, edges_dual),
            dec=dec_operators(P=(P0d, P1d),
                                 B=(B0d, B1d),
                                 D=(D0d, D1d),
                                 H=(H0d, H1d)),
            dual=None)
    g = cls(n=n,
            xmin=xmin, 
            xmax=xmax,
            pnts=pnts, 
            delta=delta,
            N = (n, n),
            simp=(verts, edges),
            dec=dec_operators(P=(P0, P1),
                                 B=(B0, B1),
                                 D=(D0, D1),
                                 H=(H0, H1)),
            dual=None)  
    g.dual, d.dual = d, g
    return g

def projections(simp):

    def P0(f):
        return f(simp[0])
    
    def P1(f):
        return sp.slow_integration(simp[1][0], simp[1][1], f)
        #return sp.split_args(sp.integrate_spectral)(grid.simp[1][0], grid.simp[1][1])
        
    return P0, P1

def derivatives(grid):
    D0 = grid.dec.D[0]
    D0d = grid.dual.dec.D[0]
    return D0, D0d

def hodge_star(grid):
    H0 = grid.dec.H[0]
    H1 = grid.dec.H[1]
    H0d = grid.dual.dec.H[0]
    H1d = grid.dual.dec.H[1]
    return H0, H1, H0d, H1d

class Grid_1D(object):
    '''
    >>> g = Grid_1D.periodic(5)
    '''
    
    def __init__(self, n, xmin, xmax, pnts, delta, N, simp, dec, dual):
        self.dimension = 1
        self.n = n
        self.xmin = xmin
        self.xmax = xmax
        self.pnts = pnts
        self.delta = delta
        self.N = N
        self.simp = simp
        self.dual = dual
        self.dec = dec
        
    def __repr__(self):
        return 'Grid_1D{}'.format((self.n, self.xmin, self.xmax))
    
Grid_1D.periodic = classmethod(periodic_fn)

class Grid_1D_Periodic:

    def __init__(self, n, xmin=0, xmax=2*np.pi):
        assert xmax > xmin

        dimension = 1

        pnts = np.linspace(xmin, xmax, num=(n+1))
        lenx = abs(xmax - xmin)
        delta = np.diff(pnts)

        verts = pnts[:-1]
        edges = (pnts[:-1], pnts[1:])

        verts_dual = verts + 0.5*delta
        edges_dual = (np.roll(verts_dual,shift=1), verts_dual)
        edges_dual[0][0] -= lenx
        delta_dual = delta

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
        P1 = lambda f: sp.slow_integration(self.edges[0], self.edges[1], f)
#         P1 = lambda f: split_args(integrate_spectral)(
#                         self.edges[0], self.edges[1], f)
        P0d = lambda f: f(self.verts_dual)
        P1d = lambda f: sp.slow_integration(self.edges_dual[0], self.edges_dual[1], f)
#         P1d = lambda f: split_args(integrate_spectral)(
#                         self.edges_dual[0], self.edges_dual[1], f)
        return P0, P1, P0d, P1d

    def basis_fn(self):
        n = self.n
        B0 = [lambda x, i=i: sp.phi0(n, i, x) for i in range(n)]
        B1 = [lambda x, i=i: sp.phi1(n, i, x) for i in range(n)]
        B0d = [lambda x, i=i: sp.phid0(n, i, x) for i in range(n)]
        B1d = [lambda x, i=i: sp.phid1(n, i, x) for i in range(n)]
        return B0, B1, B0d, B1d

    def reconstruction(self):
        R0, R1, R0d, R1d = sp.reconstruction(self.basis_fn())
        return R0, R1, R0d, R1d

    def derivative(self):
        D0  = lambda f: np.roll(f, shift=-1) - f
        D0d = lambda f: np.roll(D0(f), shift=+1)
        return D0, D0d

    def hodge_star(self):
        H0 = lambda x: np.real(sp.H(x))
        H1 = lambda x: np.real(sp.Hinv(x))
        H0d = H0
        H1d = H1
        return H0, H1, H0d, H1d

    def wedge(self):
        '''
        Return \alpha ^ \beta. Keep only for primal forms for now.
        '''
        def w00(alpha, beta):
            return alpha * beta
        def _w01(alpha, beta):
            return sp.S(sp.H( alpha * sp.Hinv(sp.Sinv(beta)) ))
        def w01(alpha, beta):
            #a = interweave(alpha, T(alpha, [S]))
            #b = interweave(T(beta, [Hinv, Sinv]), T(beta, [Hinv]))
            a = sp.refine(alpha)
            b = sp.refine(sp.Hinv(sp.Sinv(beta)))
            c = sp.S(sp.H(a * b))
            return c[0::2] + c[1::2]
        return w00, w01, _w01

    def switch(self):
        '''
        Switch between primal and dual 0-forms. The operator is only invertible 
        if they have the same size.
        '''
        return sp.S, sp.Sinv

    def contraction(self, V):
        return sp.contraction1(self, V)

def derivative_matrix(n):
    rng = np.arange(n)
    ons = np.ones(n)
    d = np.row_stack((
               np.column_stack((
                 rng,
                 np.roll(rng, shift= -1),
                 +ons)),
               np.column_stack((
                 rng,
                 rng,
                 -ons))
               ))
    D = np.sparse_matrix(d, n, n)
    return D, -D.T

def differentiation_toeplitz(n):
    raise NotImplemented
    #TODO: fix this implementation
    h = 2*np.pi/n
    assert n % 2 == 0
    column = np.concatenate(( [ 0 ], (-1)**np.arange(1,n) / np.tan(np.arange(1,n)*h/2)  ))
    row = np.concatenate(( column[:1], column[1:][::-1] ))
    D = toeplitz(column, row)
    return D

def hodge_star_toeplitz(g):
    '''
    The Hodge-Star using a Toeplitz matrix.
    '''
    P0, P1, P0d, P1d = g.projection()
    column = P1d(lambda x: sp.alpha0(g.n, x))
    row = np.concatenate((column[:1], column[1:][::-1]))
    return toeplitz(column, row)

