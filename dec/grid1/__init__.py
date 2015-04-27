from .periodic import Grid_1D_Periodic
from .regular import Grid_1D_Regular
from .chebyshev import Grid_1D_Chebyshev

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
