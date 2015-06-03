from dec.symbolic import *
from dec.grid2 import Grid_2D

c = Chart(x,y)

def compare_comps(g, f, isprimal):
    #TODO: Why aren't the arrays equal to higher tolerance?
    df = f.P(g, isprimal)

    assert np.allclose(f.lambdify(*g.points),
                       g.refine.T[f.degree, df.isprimal](df.array),
                       atol=1e-7)

    assert np.allclose(df.array,
                       g.refine.U[f.degree, df.isprimal](f.lambdify(*g.points)),
                       atol=1e-7)

def test_refine():

    g = Grid_2D.chebyshev(3, 3)
    compare_comps(g, form(0, c, (x+y,)), True)
    compare_comps(g, form(0, c, (x+y,)), False)
    
    compare_comps(g, form(1, c, ( x, y)), True)
    compare_comps(g, form(1, c, ( x, y)), False)
    compare_comps(g, form(1, c, (-y, x)), True)
    compare_comps(g, form(1, c, (-y, x)), False)

    compare_comps(g, form(2, c, ( x+y,)), True)
    compare_comps(g, form(2, c, ( x+y,)), False)
    compare_comps(g, form(2, c, ( x,)), True)
    compare_comps(g, form(2, c, ( y,)), False)

#     g = Grid_2D.periodic(3, 3)
#     compare_comps(g, form(0, c, (sin(x)+cos(y),)), True)
#     compare_comps(g, form(0, c, (sin(x)+cos(y),)), False)
#     compare_comps(g, form(1, c, (sin(x),cos(y),)), True)
#     compare_comps(g, form(1, c, (sin(x),cos(y),)), False)
#     compare_comps(g, form(2, c, (sin(x),)), True)
#     compare_comps(g, form(2, c, (sin(x),)), False)
#     compare_comps(g, form(1, c, (-cos(y),cos(x),)), True)
#     compare_comps(g, form(1, c, (-cos(y),cos(x),)), False)
