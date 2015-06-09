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
    #TODO: fix integration
#    compare_comps(g, form(1, c, (-sin(y),sin(x),)), True)
#    compare_comps(g, form(1, c, (-sin(y),sin(x),)), False)
#    compare_comps(g, form(1, c, (-cos(y),cos(x),)), True)
#    compare_comps(g, form(1, c, (-cos(y),cos(x),)), False)
#    compare_comps(g, form(2, c, (sin(x),)), True)
#    compare_comps(g, form(2, c, (sin(x),)), False)

def test_P():

    g = Grid_2D.periodic(3, 3)

    f = form(0, c, (sin(x)+sin(y),))
    assert f.P(g, True) == g.P(f.degree, True, f.lambdify)
    assert f.P(g, False) == g.P(f.degree, False, f.lambdify)

    g = Grid_2D.chebyshev(3, 3)

    f = form(0, c, (x+y,))
    assert f.P(g, True) == g.P(f.degree, True, f.lambdify)
    assert f.P(g, False) == g.P(f.degree, False, f.lambdify)

    f = form(1, c, (x,y))
    assert f.P(g, True) == g.P(f.degree, True, f.lambdify)
    assert f.P(g, False) == g.P(f.degree, False, f.lambdify)

    f = form(2, c, (x+y,))
    assert f.P(g, True) == g.P(f.degree, True, f.lambdify)
    assert f.P(g, False) == g.P(f.degree, False, f.lambdify)


def test_D():
    
    g = Grid_2D.periodic(3, 3)
    f = form(0, c, (cos(x)+cos(y),))
    assert f.D.P(g, True) == f.P(g, True).D
    assert f.D.P(g, False) == f.P(g, False).D

#TODO: fix boundary conditoins
    
def test_H():
    
#TODO: Requires fixing integration first.
#     g = Grid_2D.periodic(5, 5)
#     f = form(0, c, (cos(x),))    
#     assert f.H.P(g, False) == f.P(g, True).H
#     assert f.H.P(g, True) == f.P(g, False).H
#     f = form(1, c, (sin(x),cos(y)))    
#     assert f.H.P(g, False) == f.P(g, True).H
#     assert f.H.P(g, True) == f.P(g, False).H

    g = Grid_2D.chebyshev(10, 10)
    
    f = form(0, c, (x**4,))
    assert f.H.P(g, False) == f.P(g, True).H
    assert f.H.P(g, True) == f.P(g, False).H

    f = form(1, c, (-y, x**2))
    assert f.H.P(g, False) == f.P(g, True).H
    assert f.H.P(g, True) == f.P(g, False).H

    f = form(2, c, (x**2*y,))
    assert f.H.P(g, False) == f.P(g, True).H
    assert f.H.P(g, True) == f.P(g, False).H

def test_W_C():
    
    g = Grid_2D.chebyshev(4, 4)
      
    f0 = form(0, c, (x*y,))
    f1 = form(1, c, (x**2,-y))
    f2 = form(2, c, (x+y,))
    f = [f0, f1, f2]
    
    for ((d1, p1), (d2, p2), p3) in g.dec.W.keys():
        assert (f[d1]^f[d2]).P(g, p3) == f[d1].P(g, p1).W(f[d2].P(g, p2), toprimal=p3)

    for (p1, (d2, p2), p3) in g.dec.C.keys():
        assert f[1].C(f[d2]).P(g, p3) == f[1].P(g, p1).C(f[d2].P(g, p2), toprimal=p3)
    
if __name__ == '__main__':
    test_refine()