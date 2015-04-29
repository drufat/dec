from numpy import *
import dec.spectral as sp
from dec import d_
from dec.forms import discreteform
from collections import namedtuple

Grid1D = namedtuple('Grid1D', 'n xmin xmax pnts delta verts edges dual')

F0 = discreteform('F0')
F1 = discreteform('F1')

F0d = discreteform('F0d')
F1d = discreteform('F1d')

def grid_1d_periodic(n, xmin=0, xmax=2*pi):

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
    
    d = Grid1D(n=n,
               xmin=xmin,
               xmax=xmax,
               pnts=pnts,
               delta=delta_dual, 
               verts=verts_dual,
               edges=edges_dual,
               dual=None)
    
    g = Grid1D(n=n,
               xmin=xmin, 
               xmax=xmax,
               pnts=pnts, 
               delta=delta, 
               verts=verts, 
               edges=edges, 
               dual=d)
    
    return g

@d_(F0)
def D(f):
    return F1(roll(f, shift=-1) - f)

@d_(F0d)
def D(f):
    return F1d(f - roll(f, shift=+1))
 
def P0(g, f):
    return F0(g, f(g.verts))

def P1(g, f):
    f_ = sp.slow_integration(g.edges[0], g.edges[1], f)
    return F1(g, f_)

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
        D0  = lambda f: roll(f, shift=-1) - f
        D0d = lambda f: roll(D0(f), shift=+1)
        return D0, D0d

    def hodge_star(self):
        H0 = lambda x: real(sp.H(x))
        H1 = lambda x: real(sp.Hinv(x))
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

def differentiation_toeplitz(n):
    raise NotImplemented
    #TODO: fix this implementation
    h = 2*pi/n
    assert n % 2 == 0
    column = concatenate(( [ 0 ], (-1)**arange(1,n) / tan(arange(1,n)*h/2)  ))
    row = concatenate(( column[:1], column[1:][::-1] ))
    D = toeplitz(column, row)
    return D

def hodge_star_toeplitz(g):
    '''
    The Hodge-Star using a Toeplitz matrix.
    '''
    P0, P1, P0d, P1d = g.projection()
    column = P1d(lambda x: alpha0(g.n, x))
    row = concatenate((column[:1], column[1:][::-1]))
    return toeplitz(column, row)

