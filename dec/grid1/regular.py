from dec.spectral import *
'''
Keep symmetric bases functions for 1-forms, so that the hodge star operators below are actually
the correct ones. 
'''

class Grid_1D_Regular:
    
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
    
    def contraction(self):
        raise NotImplemented
    
    def wedge(self):
        raise NotImplemented
    
    
def H1d_regular(f):
    r'''

    .. math::
        \widetilde{\mathbf{H}}^{1}=
            \mathbf{M}_{1}^{\dagger}
            {\mathbf{I}^{-\frac{h}{2},\frac{h}{2}}}^{-1}
            \mathbf{M}_{1}^{+}
            \mathbf{A}^{-1}
    '''
    f = f/A_diag(f.shape[0])
    f = mirror0(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = I_space_inv(-h/2, +h/2)(f)
    f = unmirror0(f)
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
    f = mirror0(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = I_space(-h/2, h/2)(f)
    f = unmirror0(f)
    f = f*A_diag(f.shape[0])
    return  f

def H1_regular(f):
    r'''

    .. math::
        \mathbf{H}^{1}=
            \mathbf{M}_{1}^{\dagger}
            {\mathbf{I}^{-\frac{h}{2},\frac{h}{2}}}^{-1}
            \mathbf{M}_{1}^{+}
    '''
    f = mirror1(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = I_space_inv(-h/2, h/2)(f)
    f = unmirror1(f)
    return f

def H0d_regular(f):
    r'''

    .. math::
        \widetilde{\mathbf{H}}^{0}=
            \mathbf{M}_{1}^{\dagger}
            \mathbf{I}^{-\frac{h}{2},\frac{h}{2}}
            \mathbf{M}_{1}^{+}
    '''
    f = mirror1(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = I_space(-h/2, h/2)(f)
    f = unmirror1(f)
    return f
