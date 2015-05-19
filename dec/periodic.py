from numpy import *
import dec.spectral as sp
from dec.helper import bunch

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
            delta=delta,
            N = (n, n),
            simp=(verts, edges),
            dec=None,
            dual=None)  

    d = cls(n=n,
            xmin=xmin,
            xmax=xmax,
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
    
    g.dec = bunch(P=proj(g),
                  B=(B0, B1),
                  D=(D0, D1),
                  H=(H0, H1),
                  W=None, 
                  C=None)

    d.dec = bunch(P=proj(d),
                  D=(D0d, D1d),
                  B=(B0d, B1d),
                  H=(H0d, H1d))
    
    import types
    g.wedge = types.MethodType(wedge, g)
    g.switch = types.MethodType(switch, g)
   
    return g

def to_refine():
    '''
    >>> T0, T1, T0d, T1d = to_refine()
    >>> U0, U1, U0d, U1d = from_refine()
    >>> for f in (random.rand(8), random.rand(9), arange(10)):
    ...    assert allclose(f, U0(T0(f)))
    ...    assert allclose(f, U1(T1(f)))
    ...    assert allclose(f, U0d(T0d(f)))
    ...    assert allclose(f, U1d(T1d(f)))    
    '''
    def T0(f):
        return sp.interweave(f, sp.S(f)) 
    def T0d(f):
        return sp.interweave(sp.Sinv(f), f) 
    def T1(f):
        return T0d(sp.Hinv(f))
    def T1d(f):
        return T0(sp.Hinv(f))
    return T0, T1, T0d, T1d

def from_refine():
    def U0(f):  return f[0::2]
    def U0d(f): return f[1::2]
    def U1(f):
        f = sp.S(sp.H(f))
        return f[0::2] + f[1::2]
    def U1d(f):
        f = sp.Sinv(sp.H(f))
        return f[0::2] + f[1::2]
    return U0, U1, U0d, U1d

import itertools
import dec.symbolic

def binary_operators():

    T0, T1, T0d, T1d = to_refine()
    U0, U1, U0d, U1d = from_refine()
    P = lambda *x: tuple(itertools.product(*x))
    
    T = {(0, True):  T0, 
         (1, True):  T1, 
         (0, False): T0d, 
         (1, False): T1d,}
    U = {(0, True):  U0, 
         (1, True):  U1, 
         (0, False): U0d, 
         (1, False): U1d,}

    p = P((0, 1), (True, False))
    
    Ws = dec.symbolic.wedge_1d()
    Cs = dec.symbolic.contraction_1d()
    
    W, C = {}, {}
    
    def get_w(d0, p0, d1, p1, p2):
        def w(a, b):
            a = T[d0, p0](a)
            b = T[d1, p1](b)
            (c,) = Ws[d0, d1]((a,), (b,))
            return U[d0+d1, p2](c)
        return w

    def get_c(p0, d1, p1, p2):
        def c(a, b):
            a = T[1, p0](a)
            b = T[d1, p1](b)
            (c,) = Cs[d1]((a,), (b,))
            return U[d1-1, p2](c)
        return c

    for ((d0, p0), (d1, p1), p2) in P(p, p,(True, False)):
        if d0 + d1 > 1: continue
        if p0==p1==p2 and d0==d1==0:
            W[(d0, p0), (d1, p1), p2] = lambda a, b: a*b
            continue
        W[(d0, p0), (d1, p1), p2] = get_w(d0, p0, d1, p1, p2)    

    for (p0, (d1, p1), p2) in P((True, False), p, (True, False)):
        if d1-1 < 0: continue
        C[p0, (d1, p1), p2] = get_c(p0, d1, p1, p2)    

    return W, C
        
def wedge(self):
    '''
    Return \alpha ^ \beta. Keep only for primal forms for now.
    '''
    def w00(alpha, beta):
        return alpha * beta
    def _w01(alpha, beta):
        ''' This is the associative implementation. '''
        return sp.S(sp.H( alpha * sp.Hinv(sp.Sinv(beta)) ))
    def w01(alpha, beta):
        ''' This is the non-associative implementation. '''
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

