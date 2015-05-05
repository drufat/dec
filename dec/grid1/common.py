import dec.spectral as sp
from collections import namedtuple

dec_operators = namedtuple('dec_operators', 'P B D H')

def projections(simp):

    def P0(f):
        return f(simp[0])
    
    def P1(f):
        return sp.slow_integration(simp[1][0], simp[1][1], f)
        #return sp.split_args(sp.integrate_spectral)(grid.simp[1][0], grid.simp[1][1])
        
    return P0, P1

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
