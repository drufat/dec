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
    
