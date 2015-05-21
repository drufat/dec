from dec.symbolic import *
from dec.grid1 import Grid_1D
from dec.forms import decform

c = Chart(x,)

def test_P():

    g = Grid_1D.periodic(11)
    
    f = form(0, c, (sin(x),))
    assert f.decform(g, True) == decform.P(f.degree, True, g, f.lambdify)
    assert f.decform(g, False) == decform.P(f.degree, False, g, f.lambdify)
    
    f = form(1, c, (cos(x),))
    assert f.decform(g, True) == decform.P(f.degree, True, g, f.lambdify)
    assert f.decform(g, False) == decform.P(f.degree, False, g, f.lambdify)

    g = Grid_1D.chebyshev(11)
    
    f = form(0, c, (x,))
    assert f.decform(g, True) == decform.P(f.degree, True, g, f.lambdify)
    assert f.decform(g, False) == decform.P(f.degree, False, g, f.lambdify)
    
    f = form(1, c, (x**3,))
    assert f.decform(g, True) == decform.P(f.degree, True, g, f.lambdify)
    assert f.decform(g, False) == decform.P(f.degree, False, g, f.lambdify)

def test_R():

    g = Grid_1D.periodic(11)
    pnts = np.linspace(g.xmin, g.xmax, 50)
    
    f = form(0, c, (sin(x),))
    assert np.allclose(f.lambdify(pnts), f.decform(g, True).R(pnts))
    assert np.allclose(f.lambdify(pnts), f.decform(g, False).R(pnts))

    f = form(1, c, (sin(x),))
    assert np.allclose(f.lambdify(pnts), f.decform(g, True).R(pnts))
    assert np.allclose(f.lambdify(pnts), f.decform(g, False).R(pnts))

    g = Grid_1D.chebyshev(11)
    # do not include boundaries
    pnts = np.linspace(g.xmin, g.xmax, 50)[1:-1] 
    
    f = form(0, c, (x**2,))
    assert np.allclose(f.lambdify(pnts), f.decform(g, True).R(pnts))
    assert np.allclose(f.lambdify(pnts), f.decform(g, False).R(pnts))

    f = form(1, c, (x**2,))
    assert np.allclose(f.lambdify(pnts), f.decform(g, True).R(pnts))
    assert np.allclose(f.lambdify(pnts), f.decform(g, False).R(pnts))

def test_D():
    
    g = Grid_1D.periodic(11)
    f = form(0, c, (cos(x),))    
    assert f.D.decform(g, True) == f.decform(g, True).D

    g = Grid_1D.chebyshev(11)
    f = form(0, c, (x,))    
    assert f.D.decform(g, True) == f.decform(g, True).D
    f = form(0, c, (x**3,))    
    assert f.D.decform(g, True) == f.decform(g, True).D
    
def test_H():

    g = Grid_1D.periodic(11)
    f = form(0, c, (cos(3*x),))    
    assert f.H.decform(g, False) == f.decform(g, True).H
    assert f.H.decform(g, True) == f.decform(g, False).H
    f = form(1, c, (sin(x)+cos(x),))    
    assert f.H.decform(g, False) == f.decform(g, True).H
    assert f.H.decform(g, True) == f.decform(g, False).H

    g = Grid_1D.chebyshev(11)
    f = form(0, c, (x**4,))    
    assert f.H.decform(g, False) == f.decform(g, True).H
    assert f.H.decform(g, True) == f.decform(g, False).H
    f = form(1, c, (x**2 + 1,))    
    assert f.H.decform(g, False) == f.decform(g, True).H
    assert f.H.decform(g, True) == f.decform(g, False).H

def test_W_C():

    g = Grid_1D.periodic(11)
    
    f0 = form(0, c, (cos(x),))
    f1 = form(1, c, (sin(x),))

    assert (f0^f0).decform(g, True) == f0.decform(g, True).W(f0.decform(g, True))
    assert (f0^f1).decform(g, True) == f0.decform(g, True).W(f1.decform(g, True))
    assert (f0^f1).decform(g, False) == f0.decform(g, True).W(f1.decform(g, True), toprimal=False)
    assert (f0^f1).decform(g, False) == f0.decform(g, False).W(f1.decform(g, False), toprimal=False)
    
    assert (f1.C(f1)).decform(g, True) == f1.decform(g, True).C(f1.decform(g, True))
    assert (f1.C(f1)).decform(g, False) == f1.decform(g, False).C(f1.decform(g, False), toprimal=False)
    
    g = Grid_1D.chebyshev(11)
    
    f0 = form(0, c, (x,))
    f1 = form(1, c, (x**2,))

    assert (f0^f0).decform(g, True) == f0.decform(g, True).W(f0.decform(g, True))
    assert (f0^f1).decform(g, True) == f0.decform(g, True).W(f1.decform(g, True))
    assert (f0^f1).decform(g, False) == f0.decform(g, True).W(f1.decform(g, True), toprimal=False)
    assert (f0^f1).decform(g, False) == f0.decform(g, False).W(f1.decform(g, False), toprimal=False)
    
    assert (f1.C(f1)).decform(g, True) == f1.decform(g, True).C(f1.decform(g, True))
    assert (f1.C(f1)).decform(g, False) == f1.decform(g, False).C(f1.decform(g, False), toprimal=False)
