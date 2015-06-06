from numpy import *
import dec.spectral as sp
import dec.common
from dec.helper import bunch

def make(cls, n, xmin=-1, xmax=+1):

    # 2n-1 points: n primal, n-1 dual
    x = sin(linspace(-pi/2, pi/2, 2*n-1))
    pnts = 0.5*(xmin*(1-x) + xmax*(1+x))

    verts = pnts[::2]
    delta = diff(verts)
    edges = (verts[:-1], verts[1:])

    verts_dual = pnts[1::2]
    tmp = concatenate(([pnts[0]], verts_dual, [pnts[-1]]))
    delta_dual = diff(tmp)
    edges_dual = (tmp[:-1], tmp[1:])

    N = {(0, True)  : n, 
         (1, True)  : n-1,
         (0, False) : n-1,
         (1, False) : n}

    Delta = {True  : delta,
             False : delta_dual}
    
    cells = {(0, True)  : verts, 
             (1, True)  : edges,
             (0, False) : verts_dual,
             (1, False) : edges_dual,}

    B={(0, True)  : lambda i: lambda x: sp.psi0(n, i, x), 
       (1, True)  : lambda i: lambda x: sp.psi1(n, i, x),
       (0, False) : lambda i: lambda x: sp.psid0(n, i, x),
       (1, False) : lambda i: lambda x: sp.psid1(n, i, x),}

    D={(0, True)  : lambda f: diff(f), 
       (1, True)  : lambda f: 0,
       (0, False) : lambda f: diff(concatenate(([0], f, [0]))),
       (1, False) : lambda f: 0,}
    
    H={(0, True)  : H0_cheb, 
       (1, True)  : H1_cheb,
       (0, False) : H0d_cheb,
       (1, False) : H1d_cheb,}

    refine=dec.common.wrap_refine(to_refine, from_refine)
   
    return cls(n=n,
               xmin=xmin, 
               xmax=xmax,
               delta=Delta,
               N=N,
               cells=cells,
               dec=bunch(P=dec.common.projection(cells),
                         B=B,
                         D=D,
                         H=H,
                         W=dec.common.wedge(refine),
                         C=dec.common.contraction(refine)),
               refine=refine)  

def to_refine():
    '''
    >>> T0, T1, T0d, T1d = to_refine()
    >>> U0, U1, U0d, U1d = from_refine()
    >>> for f in (random.rand(8), random.rand(9), arange(10)):
    ...    assert len(T0(f))+2 == len(T1(f)) == len(T0d(f)) == len(T1d(f))+2
    ...    assert allclose(f, U0(T0(f)))
    ...    assert allclose(f, U1(T1(f)))
    ...    assert allclose(f, U0d(T0d(f)))
    ...    assert allclose(f, U1d(T1d(f)))    
    '''
    def T0(f):
        return sp.interweave(f, S_cheb(f)) 
    def T0d(f):
        return sp.interweave(S_cheb_pinv(f), f) 
    def T1(f):
        return T0d(H1_cheb(f))
    def T1d(f):
        return T0(H1d_cheb(f))
    return T0, T1, T0d, T1d

def from_refine():
    def U0(f):  return f[0::2]
    def U0d(f): return f[1::2]
    def U1(f):
        f = H0d_cheb(S_cheb(f))
        return f[0::2] + f[1::2]
    def U1d(f):
        f = H0d_cheb(S_cheb(f))
        m = f[1:-1]
        return concatenate([[f[0]], m[0::2] + m[1::2], [f[-1]]])
    return U0, U1, U0d, U1d

def switch():
    return S_cheb, S_cheb_pinv

def H0d_cheb(f):
    f = sp.mirror1(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = sp.F(f)
    f = sp.fourier_S(f, -h/2)
    f = sp.fourier_K(f, 0, h)
    f = sp.Finv(f)
    f = sp.unmirror1(f)
    return real(f)

def H1_cheb(f):
    f = sp.mirror1(f, -1)
    N = f.shape[0]; h = 2*pi/N
    f = sp.F(f)
    f = sp.fourier_K_inv(f, 0, h)
    f = sp.fourier_S(f, +h/2)
    f = sp.Finv(f)
    f = sp.unmirror1(f)
    return real(f)

def H0_cheb(f):
    '''
    >>> sp.to_matrix(H0_cheb, 2)
    array([[ 0.75,  0.25],
           [ 0.25,  0.75]])
    '''
    f = sp.mirror0(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = sp.F(f)
    f = sp.fourier_K(f, 0, h/2)
    f = sp.Finv(f)
    f = sp.fold0(f, -1)
    return real(f)

def H1d_cheb_new(f):
     
#     f=f.copy()
#     aa, bb = f[0], f[-1]
#     f[0], f[-1] = 0, 0
#     f = mirror0(f, -1)
#     N = f.shape[0]; h = 2*pi/N
#     f = F(f)
#     f = fourier_K_inv(f, -h/2, h/2)
#     f = Finv(f)
#     f = unmirror0(f)
#     return real(f)
    
    f = sp.unfold0(f)
    N = f.shape[0]; h = 2*pi/N
    f = sp.F(f)
    f = sp.fourier_K_inv(f, 0, h/2)
    f = sp.Finv(f)
    f = sp.unmirror0(f)
    return real(f)

def H1d_cheb(f):
    '''
    >>> sp.to_matrix(H1d_cheb, 2)
    array([[ 1.5, -0.5],
           [-0.5,  1.5]])
    '''
    N = f.shape[0]; h = pi/(N-1)
    # Is this essentially Schur's complement?
    def endpoints(f):
        f0 = sp.mirror0(sp.matC(f), -1)
        aa = f - sp.unmirror0(sp.I_space(0, h/2)(sp.I_space_inv(-h/2, h/2)(f0)))
        bb = f - sp.unmirror0(sp.I_space(-h/2, 0)(sp.I_space_inv(-h/2, h/2)(f0)))
        return sp.matB(aa) + sp.matB1(bb)
    def midpoints(f):
        f = sp.mirror0(sp.matC(f), -1)
        # Shift function with S, Sinv to avoid division by zero at x=0, x=pi
        f = sp.I_space_inv(-h/2, h/2)(f)
        f = sp.T_space(+h/2)(f)
        f = f/sp.Omega_d(f.shape[0])
        f = sp.T_space(-h/2)(f)
        f = sp.unmirror0(f)
        return f
    f = midpoints(f) + endpoints(f)
    return real(f)
    
def S_cheb(f):
    '''
    Interpolate from primal to dual vertices.
    >>> S_cheb(array([-1, 0, 1]))
    array([-0.70710678,  0.70710678])
    >>> S_cheb(array([1, 0, -1]))
    array([ 0.70710678, -0.70710678])
    >>> sp.to_matrix(S_cheb, 3).round(3)
    array([[ 0.604,  0.5  , -0.104],
           [-0.104,  0.5  ,  0.604]])
    '''
    f = sp.mirror0(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = sp.Finv(sp.F(f)*sp.S_diag(N, h/2))
    f = sp.unmirror1(f)
    return real(f)

def S_cheb_pinv(f):
    '''
    Interpolate from dual to primal vertices.
    Since there are smaller number of dual vertices, 
    this is only a pseudo inverse of S_cheb, and not an 
    exact inverse.
    >>> allclose(  S_cheb_pinv(array([ -1/sqrt(2),  1/sqrt(2)])), 
    ...            array([ -1, 0, 1]))
    True
    >>> allclose(  S_cheb_pinv(array([ 1/sqrt(2),  -1/sqrt(2)])), 
    ...            array([ 1, 0, -1]))
    True
    >>> sp.to_matrix(S_cheb_pinv, 2).round(3)
    array([[ 1.207, -0.207],
           [ 0.5  ,  0.5  ],
           [-0.207,  1.207]])
    '''
    f = sp.mirror1(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = sp.Finv(sp.F(f)*sp.S_diag(N, -h/2))
    f = sp.unmirror0(f)
    return real(f)
