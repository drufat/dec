from numpy import *
import dec.spectral as sp
from dec.forms import dec_operators

def make(proj, cls, n, xmin=0, xmax=2*pi):

    pnts = linspace(xmin, xmax, num=(n+1))
    lenx = abs(xmax - xmin)
    delta = diff(pnts)

    verts = pnts[:-1]
    edges = (pnts[:-1], pnts[1:])

    verts_dual = verts + 0.5*delta
    edges_dual = (roll(verts_dual,shift=1), verts_dual)
    edges_dual[0][0] -= lenx
    delta_dual = delta

    g = cls(n=n,
            xmin=xmin, 
            xmax=xmax,
            pnts=pnts, 
            delta=delta,
            N = (n, n),
            simp=(verts, edges),
            dec=None,
            dual=None)  

    d = cls(n=n,
            xmin=xmin,
            xmax=xmax,
            pnts=pnts,
            delta=delta_dual, 
            N = (n, n),
            simp=(verts_dual, edges_dual),
            dec=None,
            dual=None)
        
    g.dual, d.dual = d, g

    B0  = lambda i, x: sp.phi0(n, i, x)
    B1  = lambda i, x: sp.phi1(n, i, x)
    B0d = lambda i, x: sp.phid0(n, i, x)
    B1d = lambda i, x: sp.phid1(n, i, x)
        
    D0  = lambda f: roll(f, shift=-1) - f
    D0d = lambda f: roll(D0(f), shift=+1)
    D1  = lambda f: 0
    D1d = lambda f: 0 

    H0 = lambda x: real(sp.H(x))
    H1 = lambda x: real(sp.Hinv(x))
    H0d = H0
    H1d = H1
    
    g.dec = dec_operators(P=proj(g),
                          B=(B0, B1),
                          D=(D0, D1),
                          H=(H0, H1))
    
    d.dec = dec_operators(P=proj(d),
                          D=(D0d, D1d),
                          B=(B0d, B1d),
                          H=(H0d, H1d))
    
    import types
    g.wedge = types.MethodType(wedge, g)
    g.switch = types.MethodType(switch, g)
   
    return g

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

def derivative_matrix(n):
    import dec.helper
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
    D = dec.helper.sparse_matrix(d, n, n)
    return D, -D.T

def differentiation_toeplitz(n):
    raise NotImplemented
    #TODO: fix this implementation
    import scipy.linalg
    h = 2*pi/n
    assert n % 2 == 0
    column = concatenate(( [ 0 ], (-1)**arange(1,n) / tan(arange(1,n)*h/2)  ))
    row = concatenate(( column[:1], column[1:][::-1] ))
    D = scipy.linalg.toeplitz(column, row)
    return D

def hodge_star_toeplitz(g):
    '''
    The Hodge-Star using a Toeplitz matrix.
    '''
    import scipy.linalg
    P0, P1, P0d, P1d = g.projection()
    column = P1d(lambda x: sp.alpha0(g.n, x))
    row = concatenate((column[:1], column[1:][::-1]))
    return scipy.linalg.toeplitz(column, row)

