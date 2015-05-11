from numpy import *
import dec.spectral as sp
from dec.forms import dec_operators

def make(proj, cls, n, xmin=-1, xmax=+1):

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

    g = cls(n=n,
            xmin=xmin, 
            xmax=xmax,
            pnts=pnts, 
            delta=delta,
            N = (n, n-1),
            simp=(verts, edges),
            dec=None,
            dual=None)  

    d = cls(n=n-1,
            xmin=xmin,
            xmax=xmax,
            pnts=pnts,
            delta=delta_dual, 
            N = (n-1, n),
            simp=(verts_dual, edges_dual),
            dec=None,
            dual=None)

    g.dual, d.dual = d, g

    B0  = lambda i, x: sp.psi0(n, i, x)
    B1  = lambda i, x: sp.psi1(n, i, x)
    B0d = lambda i, x: sp.psid0(n, i, x)
    B1d = lambda i, x: sp.psid1(n, i, x)
        
    D0  = lambda f: diff(f)
    D0d = lambda f: diff(concatenate(([0], f, [0])))
    D1  = lambda f: 0
    D1d = lambda f: 0 

    H0 = H0_cheb
    H1 = H1_cheb
    H0d = H0d_cheb
    H1d = H1d_cheb
    
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
    
    def w00(a, b):
        return a*b
    
    def w01(a, b):
        raise NotImplemented
    
    return w00, w01

def switch(self):
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
    return midpoints(f) + endpoints(f)
    
def S_cheb(f):
    '''
    Shift from primal to dual 0-forms.
    >>> S_cheb(array([-1, 0, 1]))
    array([-0.70710678,  0.70710678])
    >>> S_cheb(array([1, 0, -1]))
    array([ 0.70710678, -0.70710678])
    '''
    f = sp.mirror0(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = sp.Finv(sp.F(f)*sp.S_diag(N, h/2))
    f = sp.unmirror1(f)
    return real(f)

def S_cheb_pinv(f):
    '''
    This is a pseudo inverse of S_cheb.     
    >>> allclose(  S_cheb_pinv(array([ -1/sqrt(2),  1/sqrt(2)])), 
    ...            array([ -1, 0, 1]))
    True
    >>> allclose(  S_cheb_pinv(array([ 1/sqrt(2),  -1/sqrt(2)])), 
    ...            array([ 1, 0, -1]))
    True
    '''
    f = sp.mirror1(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = sp.Finv(sp.F(f)*sp.S_diag(N, -h/2))
    f = sp.unmirror0(f)
    return real(f)
