from dec.symbolic import *
from dec.grid2 import Grid_2D

c = Chart(x,y)

def compare_comps(g, f, isprimal):
    
    # discrete form
    fd = f.P(g, isprimal)
    
    # lambda form
    fλ = f.lambdify

    # Test refinement
    assert np.allclose(fλ(*g.points),
                       g.refine.T[f.degree, fd.isprimal](fd.array))

    # Test unrefinement
    assert np.allclose(fd.array,
                       g.refine.U[f.degree, fd.isprimal](fλ(*g.points)))

def test_refine():
  
    g = Grid_2D.chebyshev(3, 3)
    compare_comps(g, form(0, c, (x+y,)),  True)
    compare_comps(g, form(0, c, (x+y,)),  False)
      
    compare_comps(g, form(1, c, ( x, y)), True)
    compare_comps(g, form(1, c, ( x, y)), False)
    compare_comps(g, form(1, c, (-y, x)), True)
    compare_comps(g, form(1, c, (-y, x)), False)
  
    compare_comps(g, form(2, c, ( x+y,)), True)
    compare_comps(g, form(2, c, ( x+y,)), False)
    compare_comps(g, form(2, c, ( x,)),   True)
    compare_comps(g, form(2, c, ( y,)),   False)
  
    g = Grid_2D.periodic(3, 3)
    compare_comps(g, form(0, c, (sin(x)+cos(y),)), True)
    compare_comps(g, form(0, c, (sin(x)+cos(y),)), False)
    compare_comps(g, form(1, c, (sin(x),sin(y),)), True)
    compare_comps(g, form(1, c, (sin(x),sin(y),)), False)
    compare_comps(g, form(1, c, (cos(x),cos(y),)), True)
    compare_comps(g, form(1, c, (cos(x),cos(y),)), False)
#    compare_comps(g, form(1, c, (-sin(y),sin(x),)), True)
#    compare_comps(g, form(1, c, (-sin(y),sin(x),)), False)
#    compare_comps(g, form(1, c, (-cos(y),cos(x),)), True)
#    compare_comps(g, form(1, c, (-cos(y),cos(x),)), False)
#    compare_comps(g, form(2, c, (sin(x),)), True)
#    compare_comps(g, form(2, c, (sin(x),)), False)

if __name__ == '__main__':
    test_refine()