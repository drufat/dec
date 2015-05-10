from dec.periodic import Grid_1D_Periodic
from dec.regular import Grid_1D_Regular
from dec.chebyshev import Grid_1D_Chebyshev

Grid_1D_Interface = '''
    dimension
    xmin xmax 
    delta delta_dual
    verts verts_dual
    edges edges_dual
    basis_fn
    projection
    derivative
    hodge_star
    contraction
    wedge
'''.split()

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

from dec.periodic import periodic_fn

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
    
    periodic = classmethod(periodic_fn)

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
