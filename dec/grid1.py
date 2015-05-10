import dec.helper
import dec.periodic
import dec.chebyshev
import dec.spectral
import numpy as np

def projections(grid):

    def P0(f):
        return f(grid.simp[0])
    
    def P1(f):
        return dec.helper.slow_integration(grid.simp[1][0], grid.simp[1][1], f)
        #return dec.helper.split_args(sp.integrate_spectral)(grid.simp[1][0], grid.simp[1][1])
        
    return P0, P1

class Grid_1D(object):
    '''
    >>> g = Grid_1D.periodic(3)
    >>> g.n
    3
    >>> import dec.spectral as sp
    >>> sp.to_matrix(g.dec.D[0], g.N[0])
    array([[-1.,  1.,  0.],
           [ 0., -1.,  1.],
           [ 1.,  0., -1.]])
    >>> sp.to_matrix(g.dual.dec.D[0], g.dual.N[0])
    array([[ 1.,  0., -1.],
           [-1.,  1.,  0.],
           [ 0., -1.,  1.]])
    '''
    
    periodic = classmethod(lambda *args: dec.periodic.make(projections, *args))
    chebyshev = classmethod(lambda *args: dec.chebyshev.make(projections, *args))
    regular = classmethod(lambda *args: dec.regular.make(projections, *args))

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
    
    @property
    def verts(self):
        return self.simp[0]

    @property
    def edges(self):
        return self.simp[1]

    @property
    def verts_dual(self):
        return self.dual.simp[0]
    
    @property
    def edges_dual(self):
        return self.dual.simp[1]

    def boundary_condition(self, f):
        bc = np.zeros((self.n, ))
        bc[ 0] = -f(self.xmin)
        bc[-1] = +f(self.xmax)
        return bc

    def projection(self):
        P0 = self.dec.P[0]
        P1 = self.dec.P[1]
        P0d = self.dual.dec.P[0]
        P1d = self.dual.dec.P[1]
        return P0, P1, P0d, P1d

    def basis_fn(self):
#         B0 = self.dec.B[0]
#         B1 = self.dec.B[1]
#         B0d = self.dual.dec.B[0]
#         B1d = self.dual.dec.B[1]
        B0  = [lambda x, i=i:  self.dec.B[0](i, x) for i in range(self.N[0])]
        B1  = [lambda x, i=i:  self.dec.B[1](i, x) for i in range(self.N[1])]
        B0d = [lambda x, i=i:  self.dual.dec.B[0](i, x) for i in range(self.dual.N[0])]
        B1d = [lambda x, i=i:  self.dual.dec.B[1](i, x) for i in range(self.dual.N[1])]
        return B0, B1, B0d, B1d

    def reconstruction(self):
#         R0 = self.dec.R[0]
#         R1 = self.dec.R[1]
#         R0d = self.dual.dec.R[0]
#         R1d = self.dual.dec.R[1]
        R0, R1, R0d, R1d = dec.spectral.reconstruction(self.basis_fn())
        return R0, R1, R0d, R1d

    def derivative(self):
        D0 = self.dec.D[0]
        D0d = self.dual.dec.D[0]
        return D0, D0d
    
    def hodge_star(self):
        H0 = self.dec.H[0]
        H1 = self.dec.H[1]
        H0d = self.dual.dec.H[0]
        H1d = self.dual.dec.H[1]
        return H0, H1, H0d, H1d
    
    # the operators below are products - they will require refinements

#     def wedge(self):
#         W00 = self.dec.W[0, 0]
#         W01 = self.dec.W[0, 1]
#         return W00, W01
    
    def contraction(self, V):
        return dec.spectral.contraction1(self, V)
        
    