import numpy as np
from scipy.integrate import quad, dblquad
import sympy as sy
from dec.data import memoize

##########################
# Symbolic Integration
##########################

@memoize('memoize/sympy.json', lambda expr: repr(expr), lambda expr: sy.sympify(expr))
def process(expr):
    rslt = sy.simplify(expr.doit())
    if rslt.has(sy.Integral):
        raise ValueError('Unable to evaluate {}.'.format(rslt))    
    return rslt

def enumerate_coords(coord, deg):
    '''
    >>> x, y = sy.symbols('x, y')
    >>> enumerate_coords((x,), 0)
    (x0,)
    >>> enumerate_coords((x,), 1)
    (x0, x1)
    >>> enumerate_coords((x,), 2)
    (x0, x1, x2)
    >>> enumerate_coords((x, y), 2)
    (x0, y0, x1, y1, x2, y2)
    '''
    return sy.symbols(tuple('{}{}'.format(c.name, i) for i in range(deg+1) for c in coord))

def averages_1d(x):
    '''
    >>> x = sy.symbols('x')
    >>> x0, x1 = enumerate_coords((x,), 1)
    >>> A0, A1 = averages_1d(x)
    >>> assert A0(1)     == 1
    >>> assert A0(x)     == x0
    >>> assert A1(1)     == 1
    >>> assert A1(x)     == (x1 + x0)/2
    >>> assert A1(x**2)  == (x0**2 + x0*x1 + x1**2)/3
    >>> assert A1(x**3)  == (x0**3 + x0**2*x1 + x0*x1**2 + x1**3)/4
    '''
    x0, x1 = enumerate_coords((x,), 1)
    s, t = sy.symbols('s t')
    assert t != x != s

    def A0(f):
        f = sy.sympify(f)
        return f.subs(x, x0)

    def A1(f):
        f = sy.sympify(f)
        integrand = f.subs(x, x0*(1-s) + x1*s)
        iexpr = sy.Integral(integrand, (s, 0, 1))
        return process(iexpr)

    return A0, A1

def averages_2d(x, y):
    '''
    >>> x, y = sy.symbols('x, y')
    >>> x0, y0, x1, y1, x2, y2 = enumerate_coords((x, y), 2)
    >>> A0, A1, A2 = averages_2d(x, y)
    >>> assert A0(1) == 1
    >>> assert A0(x) == x0
    >>> assert A1(1) == 1
    >>> assert A1(x) == (x0 + x1)/2
    >>> assert A1(y) == (y0 + y1)/2
    >>> assert A2(1) == 1
    >>> assert A2(x) == (x0 + x1 + x2)/3
    >>> assert A2(y) == (y0 + y1 + y2)/3
    '''
    x0, y0, x1, y1, x2, y2 = enumerate_coords((x, y), 2)
    s, t = sy.symbols('s t')
    assert t != x != s
    assert t != y != s

    def A0(f):
        f = sy.sympify(f)
        return f.subs({x:x0, y:y0})
     
    def A1(f):
        f = sy.sympify(f)
        subst = ((x, x0*(1-s) + x1*s),
                 (y, y0*(1-s) + y1*s))
        integrand = f.subs(subst)
        iexpr = sy.Integral(integrand, (s, 0, 1))
        return process(iexpr)
     
    def A2(f):
        f = sy.sympify(f)
        subst = ((x, x0*(1-s-t) + x1*s + x2*t),
                 (y, y0*(1-s-t) + y1*s + y2*t))
        integrand = 2*f.subs(subst)
        iexpr = sy.Integral(integrand, (t, 0, 1-s), (s, 0, 1))
        return process(iexpr)
    
    return A0, A1, A2

def integration_1d(x):
    '''
    >>> x = sy.symbols('x')
    >>> P0, P1 = integration_1d(x)
    >>> x0, x1 = enumerate_coords((x,), 1)
    >>> assert P0((x,)) == x0
    >>> assert P1((x,)) == x1**2/2 - x0**2/2
    >>> assert P1((1,)) == x1 - x0
    '''
    x0, x1 = enumerate_coords((x,), 1)

    def P0(f):
        f, = sy.sympify(f)
        return f.subs(x, x0)

    def P1(f):
        f, = sy.sympify(f)
        iexpr = sy.Integral(f, (x, x0, x1))
        return process(iexpr)
    
    return P0, P1

def integration_2d(x, y):
    '''
    >>> x, y = sy.symbols('x, y')
    >>> x0, y0, x1, y1, x2, y2 = enumerate_coords((x, y), 2)
    >>> P0, P1, P2 = integration_2d(x, y)
    >>> assert P0((x*y,)) == x0*y0
    >>> assert P1((x, 0)) == -x0**2/2 + x1**2/2
    >>> assert P1((1, 0)) == -x0 + x1
    >>> assert P1((1, 1)) == -x0 + x1 - y0 + y1
    >>> assert P2((0,)) == 0
    
    The expression below corresponds to the area of a triangle
    >>> from sympy import expand
    >>> assert P2((1,)) == expand( ((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))/2 )
    '''
    
    x0, y0, x1, y1, x2, y2 = enumerate_coords((x, y), 2)

    def P0(f):
        f, = sy.sympify(f)
        return f.subs({x:x0, y:y0})
     
    def P1(f):
        ux, uy = sy.sympify(f)
        s = sy.Symbol('s')
        lx, ly = x1 - x0, y1 - y0
        subst = ((x, x0*(1-s) + x1*s),
                 (y, y0*(1-s) + y1*s))
        integrand = (ux.subs(subst)*lx +
                     uy.subs(subst)*ly)
        iexpr = sy.Integral(integrand, (s, 0, 1))
        return process(iexpr)
     
    def P2(f):
        omega, = sy.sympify(f)
        s, t = sy.Symbol('s'), sy.Symbol('t')
        A = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)
        subst = ((x, x0*(1-s-t) + x1*s + x2*t),
                 (y, y0*(1-s-t) + y1*s + y2*t))
        integrand = (omega.subs(subst)*A)
        iexpr = sy.Integral(integrand, (t, 0, 1-s), (s, 0, 1))
        return process(iexpr)
    
    return P0, P1, P2

def integration_2d_regular(x, y):
    '''
    
    (x3, y3)--------(x2, y2)
       |                |
       |                |
    (x0, y0)--------(x1, y1)

    (x0, y1)--------(x1, y1)
       |                |
       |                |
    (x0, y0)--------(x1, y0)

    >>> x, y = sy.symbols('x, y')
    >>> x0, y0, x1, y1, x2, y2 = enumerate_coords((x, y), 2)
    >>> P0, P1, P2 = integration_2d_regular(x, y)
    >>> assert P0((1,)) == 1
    >>> assert P0((x*y,)) == x0*y0
    >>> assert P1((1, 0)) == ((x1-x0), 0)
    >>> assert P1((0, 1)) == (0, (y1-y0))
    >>> assert P1((y, x)) == (y*(x1-x0), x*(y1-y0))
    >>> assert P2((0,)) == 0
    >>> assert P2((1,)) == (x0-x1)*(y0-y1)
    '''
    
    x0, y0, x1, y1 = enumerate_coords((x, y), 1)

    def P0(f):
        f, = sy.sympify(f)
        return f.subs({x:x0, y:y0})

    def I(x, a, b):
        def I_(f):
            iexpr = sy.Integral(f, (x, a, b))
            return process(iexpr)
        return I_
    
    Ix = I(x, x0, x1) 
    Iy = I(y, y0, y1) 
    
    def P1(f):
        fx, fy = f
        return (Ix(fx), 
                Iy(fy))
     
    def P2(f):
        f, = sy.sympify(f)
        return Ix(Iy(f))
    
    return P0, P1, P2


#######################
# Numeric Intergration
#######################

def integrate_simpson(a, b, f):
    '''
    Simpson's 3-point rule O(h**4)
    http://en.wikipedia.org/wiki/Simpson%27s_rule

    >>> integrate_simpson(0, 1, lambda x: x)
    0.5
    >>> integrate_simpson(-1, -0.5, lambda x: x)
    -0.375
    '''
    I = ((b-a)/6.0)*(f(a) + 4*f((a+b)/2.0) + f(b))
    return I

def integrate_boole(x1, x5, f):
    '''
    Boole's 5-point rule O(h**7)
    http://en.wikipedia.org/wiki/Boole%27s_rule

    >>> integrate_boole(0, 1, lambda x: x)
    0.5
    >>> integrate_boole(0, 1, lambda x: x**4)
    0.2
    '''
    h = (x5 - x1)/4.0
    x2 = x1 + h
    x3 = x1 + 2*h
    x4 = x1 + 3*h
    #assert(x5 == x1 + 4*h)
    I = (2*h/45.0)*(7*f(x1) + 32*f(x2) + 12*f(x3) + 32*f(x4) + 7*f(x5))
    return I

def integrate_boole1(x, f):
    '''
    >>> integrate_boole1([0, 1], lambda x: x)
    array([ 0.5])
    '''
    x = np.asanyarray(x)
    return integrate_boole(x[:-1], x[1:], f)

def integrate_boole2(x1, x5, f):
    '''
    Boole's 5-point rule O(h**7) in 2D
    '''
    h = (x5 - x1)/4.0
    x2 = x1 + h
    x3 = x1 + 2*h
    x4 = x1 + 3*h
    I = (2*np.sqrt(h[0]**2 + h[1]**2)/45.0)*\
        (7*f(*x1) + 32*f(*x2) + 12*f(*x3) + 32*f(*x4) + 7*f(*x5))
    return I

def integrate_1form(edge, f):
    '''
    Integrate a continuous one-form **f** along an **edge** 
    ((x0, y0), (x1, y1))
    >>> integrate_1form( ((0,0), (1,0)), lambda x, y: (1, 0) )[0]
    array(1.0)
    >>> integrate_1form( ((0,0), (1,0)), lambda x, y: (0, 1) )[0]
    array(0.0)
    '''
    def tmp(x0, y0, x1, y1):
        dx = x1 - x0
        dy = y1 - y0
        def _f(t):
            x = x0 + t*dx
            y = y0 + t*dy
            fx, fy = f(x, y)
            return fx*dx + fy*dy
        return quad(_f, 0, 1)

    ((x0, y0), (x1, y1)) = edge
    return np.vectorize(tmp)(x0, y0, x1, y1)

def _integrate_2form(face, f):
    g = lambda x: 0
    h = lambda x: 1-x

    def tmp(x0, y0, x1, y1, x2, y2):
        dx1 = x1 - x0
        dy1 = y1 - y0
        dx2 = x2 - x0
        dy2 = y2 - y0
        _f = lambda u, v: f(x0 + u*dx1 + v*dx2,
                            y0 + u*dy1 + v*dy2)*(dx1*dy2 - dy1*dx2)
        return dblquad(_f, 0, 1, g, h)

    ((x0, y0), (x1, y1), (x2, y2)) = face
    return np.vectorize(tmp)(x0, y0, x1, y1, x2, y2)

def integrate_2form(face, f):
    '''
    Integrate a continuous two-form **f** on a **face** 
    ((x0, y0), (x1, y1), (x2, y2), ...)
    >>> integrate_2form( ((0,0), (2,0), (0,2)), lambda x, y: 1 )[0]
    2.0
    >>> integrate_2form( ((0,0), (1,0), (1,1), (0,1)), lambda x, y: 1 )[0]
    1.0
    '''
    integral = 0.0
    error = 0.0

    a = face[0]
    for b, c in zip(face[1:-1], face[2:]):
        i, e = _integrate_2form((a,b,c), f)
        integral += i
        error += e

    return integral, error

def slow_integration(a, b, f):
    return np.array([quad(f, _a, _b)[0] for _a, _b in zip(a, b)])

