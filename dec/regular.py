from numpy import *
import dec.spectral as sp
from dec.forms import form_operators
'''
Keep symmetric bases functions for 1-forms, so that the hodge star operators below are actually
the correct ones. 
'''

def make(proj, cls, n, xmin=0, xmax=pi):

    assert xmax > xmin
    
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

    g = cls(n=n,
            xmin=xmin, 
            xmax=xmax,
            delta=delta,
            N = (n, n-1),
            simp=(verts, edges),
            dec=None,
            dual=None)  

    d = cls(n=n-1,
            xmin=xmin,
            xmax=xmax,
            delta=delta_dual, 
            N = (n-1, n),
            simp=(verts_dual, edges_dual),
            dec=None,
            dual=None)

    g.dual, d.dual = d, g

    B0  = lambda i, x: sp.kappa0(n, i, x)
    B1  = lambda i, x: sp.kappa1_symm(n, i, x)
    B0d = lambda i, x: sp.kappad0(n, i, x)
    B1d = lambda i, x: sp.kappad1_symm(n, i, x)
        
    D0  = lambda f: diff(f)
    D0d = lambda f: diff(concatenate(([0], f, [0])))
    D1  = lambda f: 0
    D1d = lambda f: 0 

    H0 = H0_regular
    H1 = H1_regular
    H0d = H0d_regular
    H1d = H1d_regular
    
    g.dec = form_operators(P=proj(g),
                          B=(B0, B1),
                          D=(D0, D1),
                          H=(H0, H1))
    
    d.dec = form_operators(P=proj(d),
                          D=(D0d, D1d),
                          B=(B0d, B1d),
                          H=(H0d, H1d)) 
    
    return g
    
def H1d_regular(f):
    r'''

    .. math::
        \widetilde{\mathbf{H}}^{1}=
            \mathbf{M}_{1}^{\dagger}
            {\mathbf{I}^{-\frac{h}{2},\frac{h}{2}}}^{-1}
            \mathbf{M}_{1}^{+}
            \mathbf{A}^{-1}
    '''
    f = f/sp.A_diag(f.shape[0])
    f = sp.mirror0(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = sp.I_space_inv(-h/2, +h/2)(f)
    f = sp.unmirror0(f)
    return f

def H0_regular(f):
    r'''

    .. math::
        \mathbf{H}^{0}=
            \mathbf{A}
            \mathbf{M}_{0}^{\dagger}
            \mathbf{I}^{-\frac{h}{2},\frac{h}{2}}
            \mathbf{M}_{0}^{+}
    '''
    f = sp.mirror0(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = sp.I_space(-h/2, h/2)(f)
    f = sp.unmirror0(f)
    f = f*sp.A_diag(f.shape[0])
    return  f

def H1_regular(f):
    r'''

    .. math::
        \mathbf{H}^{1}=
            \mathbf{M}_{1}^{\dagger}
            {\mathbf{I}^{-\frac{h}{2},\frac{h}{2}}}^{-1}
            \mathbf{M}_{1}^{+}
    '''
    f = sp.mirror1(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = sp.I_space_inv(-h/2, h/2)(f)
    f = sp.unmirror1(f)
    return f

def H0d_regular(f):
    r'''

    .. math::
        \widetilde{\mathbf{H}}^{0}=
            \mathbf{M}_{1}^{\dagger}
            \mathbf{I}^{-\frac{h}{2},\frac{h}{2}}
            \mathbf{M}_{1}^{+}
    '''
    f = sp.mirror1(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = sp.I_space(-h/2, h/2)(f)
    f = sp.unmirror1(f)
    return f
