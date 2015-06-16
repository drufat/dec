from dec.symbolic import *
from dec.grid2 import Grid_2D, projection_2d
import sympy as sy
from dec.decform import decform

c = Chart(x,y)

def compare_comps(g, f, isprimal):
    
    # discrete form
    fd = g.P(f, isprimal)
    
    # lambda form
    fλ = sy.lambdify(f.grid.coords, f.components, 'numpy')

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
    compare_comps(g, form(1, c, (-sin(y),sin(x),)), True)
    compare_comps(g, form(1, c, (-sin(y),sin(x),)), False)
    compare_comps(g, form(1, c, (-cos(y),cos(x),)), True)
    compare_comps(g, form(1, c, (-cos(y),cos(x),)), False)
    compare_comps(g, form(2, c, (sin(x),)), True)
    compare_comps(g, form(2, c, (sin(x),)), False)

def test_P():


    def P(form, isprimal, g):
        proj = projection_2d(g.cells)
        a = proj[form.degree, isprimal](form.lambdify)
        return decform(form.degree, isprimal, g, a)

    def primaldual(t):
        g = Grid_2D.periodic(5, 3)
        
        f = form(0, c, (sin(x)+sin(y),))
        assert P(f, t, g)  == g.P(f, t)
        f = form(1, c, (sin(x),sin(y),))
        assert P(f, t, g)  == g.P(f, t) 
        f = form(2, c, (sin(x)+sin(y),))
        assert P(f, t, g)  == g.P(f, t)
        
        g = Grid_2D.chebyshev(3, 6)
        
        f = form(0, c, (x+y,))
        assert P(f, t, g)  == g.P(f, t)
        f = form(1, c, (x,y))
        assert P(f, t, g)  == g.P(f, t)
        f = form(2, c, (x+y,))
        assert P(f, t, g)  == g.P(f, t)
        
    primaldual(True)
    primaldual(False)


def test_N():

    def check_N(g):
        f = form(0, c, (0,))
        assert g.P(f, True).array.shape[0] == g.N[0, True]
        assert g.P(f, False).array.shape[0] == g.N[0, False]
        f = form(1, c, (0,0))
        assert g.P(f, True).array.shape[0] == g.N[1, True]
        assert g.P(f, False).array.shape[0] == g.N[1, False]
        f = form(2, c, (0,))
        assert g.P(f, True).array.shape[0] == g.N[2, True]
        assert g.P(f, False).array.shape[0] == g.N[2, False]

    check_N(Grid_2D.periodic(4, 3))
    check_N(Grid_2D.periodic(3, 5))
    check_N(Grid_2D.chebyshev(2, 3))
    check_N(Grid_2D.chebyshev(5, 4))


def test_D():
    
    g = Grid_2D.periodic(3, 3)
    
    f = form(0, c, (cos(x)+cos(y),))
    assert f.D.P(g, True) == f.P(g, True).D
    assert f.D.P(g, False) == f.P(g, False).D

    f = form(1, c, (cos(x),cos(y),))
    assert f.D.P(g, True) == f.P(g, True).D
    assert f.D.P(g, False) == f.P(g, False).D

#TODO: fix boundary conditions
    g = Grid_2D.chebyshev(3, 3)
    f = form(0, c, (x+y,))
    assert f.D.P(g, True) == f.P(g, True).D
    #assert f.D.P(g, False) == f.P(g, False).D

    f = form(1, c, (-y,x))
    assert f.D.P(g, True) == f.P(g, True).D
    #assert f.D.P(g, False) == f.P(g, False).D
    
def test_H():
    
    g = Grid_2D.periodic(5, 5)

    f = form(0, c, (cos(x),))    
    assert f.H.P(g, False) == f.P(g, True).H
    assert f.H.P(g, True) == f.P(g, False).H
    
    f = form(1, c, (sin(x),cos(y)))    
    assert f.H.P(g, False) == f.P(g, True).H
    assert f.H.P(g, True) == f.P(g, False).H

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
      
    f0 = form(0, c, (x,))
    f1 = form(1, c, (x**2,-y))
    f2 = form(2, c, (x+y,))
    f = [f0, f1, f2]

    h0 = form(0, c, (y,))
    h1 = form(1, c, (-y,x))
    h2 = form(2, c, (x-y,))
    h = [h0, h1, h2]
    
    for ((d1, p1), (d2, p2), p3) in g.dec.W.keys():
        assert (f[d1]^h[d2]).P(g, p3) == f[d1].P(g, p1).W(h[d2].P(g, p2), toprimal=p3)

    for (p1, (d2, p2), p3) in g.dec.C.keys():
        assert f[1].C(h[d2]).P(g, p3) == f[1].P(g, p1).C(h[d2].P(g, p2), toprimal=p3)
    
if __name__ == '__main__':
    test_refine()