'''
    Module for symbolic computations.    
'''
import numpy as np
from sympy import (symbols, Function, diff, lambdify, simplify,
                   sympify, Dummy, Symbol,
                   integrate, Integral,
                   sin, cos)
from dec.helper import bunch, nCr

# Coordinates
x, y, z = symbols('x y z')
# Vector Fields
u, v = Function('u')(x,y), Function('v')(x,y)
# Scalar Fields
f, g = Function('f')(x,y), Function('g')(x,y)
# Coordinates of Simplex Vertices
x0, y0, x1, y1, x2, y2 = symbols('x0, y0, x1, y1, x2, y2')

"""
V represents the velocity vector field.
"""
V = [
    (-2*sin(y)*cos(x/2)**2,
      2*sin(x)*cos(y/2)**2),
    (-cos(x/2)*sin(y/2), sin(x/2)*cos(y/2)),
    (-sin(y), sin(x)),
    (-sin(2*y), sin(x)),
    (1, 0)
]

"""
p represents the pressure scalar field.
"""
p  = [
    (-cos(2*x)*(5+4*cos(y))-5*(4*cos(y)+cos(2*y))-4*cos(x)*(5+5*cos(y)+cos(2*y)))/20,
    -(cos(x)+cos(y))/4,
    -cos(x)*cos(y),
    -4*cos(x)*cos(2*y)/5,
    0
]

class Chart:
    '''
    >>> c = Chart(x)
    >>> assert c.dimension == 1
    >>> c = Chart(x, y)
    >>> assert c.dimension == 2    
    '''
    
    def __init__(self, *coords):
        self.coords = coords
        self.dimension = len(coords)
        if len(coords) == 1:
            x, = coords
            dec = bunch(D=derivative_1d(x),
                        P=projections_1d(x),
                        H=hodge_star_1d(),
                        W=wedge_1d(),
                        C=contraction_1d(),
                        )
        elif len(coords) == 2:
            x, y = coords
            dec = bunch(D=derivative_2d(x, y),
                        P=projections_2d(x, y),
                        H=hodge_star_2d(),
                        W=wedge_2d(),
                        C=contraction_2d(),
                        )
        else:
            raise NotImplementedError
        self.dec = dec
    
    def simpl_coords(self, deg):
        assert deg <= self.dimension
        return enumerate_coords(self.coords, deg)
    
    def __repr__(self):
        return "Chart{}".format(self.coords)

def Chart_1d(x=x):
    '''
    >>> Chart_1d()
    Chart(x,)
    '''
    return Chart(x,)

def Chart_2d(x=x, y=y):
    '''
    >>> Chart_2d()
    Chart(x, y)
    '''
    return Chart(x, y)

def enumerate_coords(coord, deg):
    '''
    >>> enumerate_coords((x,), 0)
    (x0,)
    >>> enumerate_coords((x,), 1)
    (x0, x1)
    >>> enumerate_coords((x,), 2)
    (x0, x1, x2)
    >>> enumerate_coords((x, y), 2)
    (x0, y0, x1, y1, x2, y2)
    '''
    return symbols(tuple('{}{}'.format(c.name, i) for i in range(deg+1) for c in coord))

try:

    from pythematica import Pythematica
    mathematica = Pythematica()
        
    def Integrate(*args, Assumptions=None):
        return mathematica.Integrate(
                *args, 
                Assumptions=Assumptions)

except ImportError:
    
    def Integrate(*args, Assumptions=None):
        expr, *bounds = args
        return integrate(expr, *reversed(bounds))


def projections_1d(x):
    '''
    >>> P0, P1 = projections_1d(x)
    >>> assert P0((x,)) == x0
    >>> assert P1((x,)) == x1**2/2 - x0**2/2
    >>> assert P1((1,)) == x1 - x0
    '''
    x0, x1 = enumerate_coords((x,), 1)

    def P0(f):
        f = sympify(f[0])
        return f.subs(x, x0)

    def P1(f):
        f = sympify(f[0])
        iexpr = integrate(f, (x, x0, x1))
        if iexpr.has(Integral):
            raise ValueError('Unable to evaluate {}.'.format(iexpr))
        return simplify(iexpr)

    return P0, P1

def projections_2d(x, y):
    '''
    >>> P0, P1, P2 = projections_2d(x, y)
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
    assum = (x0 != x1) | (y0 != y1)

    def P0(f):
        f, = sympify(f)
        return f.subs({x:x0, y:y0})
     
    def P1(f):
        ux, uy = sympify(f)
        s = Symbol('s')
        lx, ly = x1 - x0, y1 - y0
        subst = ((x, x0*(1-s) + x1*s),
                 (y, y0*(1-s) + y1*s))
        integrand = (ux.subs(subst)*lx +
                     uy.subs(subst)*ly)
        iexpr = Integrate(integrand, (s, 0, 1), Assumptions=assum)
        if iexpr.has(Integral):
            raise ValueError('Unable to evaluate {}.'.format(iexpr))
        return simplify(iexpr)
     
    def P2(f):
        omega, = sympify(f)
        s, t = Symbol('s'), Symbol('t')
        A = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)
        subst = ((x, x0*(1-s-t) + x1*s + x2*t),
                 (y, y0*(1-s-t) + y1*s + y2*t))
        integrand = (omega.subs(subst)*A)
        iexpr = Integrate(integrand, (s, 0, 1), (t, 0, 1-s), Assumptions=assum)
        if iexpr.has(Integral):
            raise ValueError('Unable to evaluate {}.'.format(iexpr))
        return simplify(iexpr)
    
    return P0, P1, P2

def run_mathematica():
    from sympy import sin, cos
    P0, P1, P2 = projections_2d(x, y)

    P1_d = {}    
    for f in ((1,0),
              (0,1),
              (sin(x),0),
              (sin(y),0),
              (-sin(y), sin(x)),
              (sin(x), sin(y)),
              (sin(x), cos(2*y)),
              ):
        P1_d[repr(f)] = repr(P1(f))
        print()
        print(f)
        print(P1_d[repr(f)])
        
    P2_d = {}    
    for f in (1,
              sin(x),
              sin(y),
              cos(x),
              cos(y),
              sin(x+y),
              sin(x+2*y),
              x, y, x**2, y**2, x*y, 
              x**3, x**2*y, x*y**2, y**3,
              ):
        P2_d[repr((f,))] = repr(P2((f,)))
        print()
        print(f)
        print(P2_d[repr((f,))])
    
    dataname = 'memoize/projections.json'
    import dec
    dec.store_data(dataname, {1:P1_d, 2:P2_d})
    return

def derivative_1d(x):    
    '''
    >>> D = derivative_1d(x)
    >>> D[0]( (x,) )
    (1,)
    >>> D[0]( (x*y,) )
    (y,)
    >>> D[1]( (x,) )
    0
    '''
    D0 = lambda f: (diff(f[0], x),)
    D1 = lambda f: 0
    return D0, D1

def derivative_2d(x, y):    
    '''
    >>> D = derivative_2d(x, y)
    >>> D[0]( (x,) )
    (1, 0)
    >>> D[0]( (x*y,) )
    (y, x)
    >>> D[1]( (-y, x) )
    (2,)
    >>> D[2]( (x,) )
    0
    '''
    Dx = lambda f: diff(f, x)
    Dy = lambda f: diff(f, y)
    D0 = lambda f: (Dx(f[0]), 
                    Dy(f[0]))
    D1 = lambda f: (-Dy(f[0]) + Dx(f[1]),)
    D2 = lambda f: 0
    return D0, D1, D2

def hodge_star_1d():
    '''
    >>> H = hodge_star_1d()
    >>> H[0]((x,))
    (x,)
    >>> H[1]((x,))
    (x,)
    '''
    H0 = lambda f: f
    H1 = lambda f: f
    return H0, H1

def hodge_star_2d():
    '''
    >>> H = hodge_star_2d()
    >>> H[0]((x,))
    (x,)
    >>> H[1]((x,y))
    (-y, x)
    >>> H[2]((x,))
    (x,)
    '''
    H0 = lambda f: f
    H1 = lambda f: (-f[1], f[0])
    H2 = lambda f: f
    return H0, H1, H2

def antisymmetrize_wedge(W):
    r'''
    :math:`\alpha\wedge\beta` is **anticommutative**:
    :math:`\alpha\wedge\beta=(-1)^{kl}\beta\wedge\alpha`, where
    :math:`\alpha` is a :math:`k`-form and :math:`\beta` is an
    :math:`l`-form.
    '''
    keys = [key for key in W]
    for k, l in keys:
        if k == l: continue
        W[l, k] = (lambda k, l:
                    lambda a, b: 
                        tuple( c * (-1)**(k*l) for c in W[k, l](b, a)) 
                    )(k, l)

def wedge_1d():
    '''
    >>> u, v, f, g = symbols('u v f g')
    >>> W = wedge_1d()
    >>> W[0,0]((f,),(g,))
    (f*g,)
    >>> W[0,1]((f,),(u,))
    (f*u,)
    >>> W[1,0]((u,),(f,))
    (f*u,)
    '''
    W = {}
    W[0,0] = lambda a, b: (a[0]*b[0], )
    W[0,1] = lambda a, b: (a[0]*b[0], )
    antisymmetrize_wedge(W)
    return W

def wedge_2d():
    '''
    >>> u, v, f, g = symbols('u v f g')
    >>> W = wedge_2d()
    >>> W[0,0]((f,),(g,))
    (f*g,)
    >>> W[0,1]((f,),(u,v))
    (f*u, f*v)
    >>> W[1,0]((u,v),(f,))
    (f*u, f*v)
    >>> W[1,1]((u,v),(f,g))
    (-f*v + g*u,)
    >>> W[0,2]((f,),(g,))
    (f*g,)
    >>> W[2,0]((g,),(f,))
    (f*g,)
    '''
    W = {}
    W[0,0] = lambda a, b: (a[0]*b[0], )
    W[0,1] = lambda a, b: (a[0]*b[0], a[0]*b[1])
    W[0,2] = lambda a, b: (a[0]*b[0],)
    W[1,1] = lambda a, b: (a[0]*b[1]-a[1]*b[0],)    
    antisymmetrize_wedge(W)
    return W

def contraction_1d():
    '''
    Contraction
    >>> C = contraction_1d()
    >>> u, v, f, g = symbols('u v f g')
    >>> X = (u,)
    >>> C[0](X, (f,))
    0
    >>> C[1](X, (f,))
    (f*u,)
    '''
    C0 = lambda X, f: 0
    C1 = lambda X, f: ( X[0]*f[0], )
    return C0, C1
    
def contraction_2d():
    '''
    Contraction
    >>> C = contraction_2d()
    >>> u, v, f, g = symbols('u v f g')
    >>> X = (u, v)
    >>> C[0](X, (f,))
    0
    >>> C[1](X, (f,g))
    (f*u + g*v,)
    >>> C[2](X, (f,))
    (-f*v, f*u)
    '''
    C0 = lambda X, f: 0
    C1 = lambda X, f: ( X[0]*f[0]+X[1]*f[1], )
    C2 = lambda X, f: (-X[1]*f[0],
                        X[0]*f[0],)
    return C0, C1, C2

def form_factory(name):
    '''
    >>> form = form_factory('form')

    >>> f, g, u, v = symbols('f g u v')

    >>> c = Chart(x, y)
    >>> α = form(1, c, (f, g))
    >>> φ = form(1, c, (u, v))
    
    >>> -φ
    form(1, Chart(x, y), (-u, -v))
    >>> φ + φ
    form(1, Chart(x, y), (2*u, 2*v))
    >>> φ + α
    form(1, Chart(x, y), (f + u, g + v))
    >>> φ - α
    form(1, Chart(x, y), (-f + u, -g + v))
    
    We can use ^ as the wedge product operator.    

    >>> assert φ ^ φ == form(2, c, (0,))
    >>> assert α ^ φ == - φ ^ α
    '''
    
    F = type(name, (object,), {})
    
    def __init__(self, degree, chart, components):
        # make sure the form has the correct number of components
        assert degree <= chart.dimension
        assert len(components) == nCr(chart.dimension, degree)
        self.components = tuple(sympify(_) for _ in components)
        self.chart = chart
        self.degree = degree

    def __repr__(self):
        t = (self.degree, self.chart, self.components)
        return name + t.__repr__()

    def __eq__(self, other):
        return (self.degree == other.degree and 
                self.chart == other.chart and
                self.components == other.components)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __rmul__(self, other):
        if callable(other):
            return other(self)
        else:
            return type(self)(self.degree, (c.__rmul__(other) for c in self.components))

    def __xor__(self, other):
        return self.W(other)
    
    def __getitem__(self, k):
        return self.components[k]
        
    def binary(name):
        def __fname__(self, other):
            if type(other) is type(self):
                assert self.degree == other.degree
                assert self.chart == other.chart
                comps = tuple(getattr(s, name)(o) for s, o in zip(self.components, other.components))
            else:    
                comps = tuple(getattr(s, name)(other) for s in self)
            return F(self.degree, self.chart, comps)
        return __fname__

    def unary(name):
        def __fname__(self):
            comps = tuple(getattr(s, name)() for s in self.components)
            return F(self.degree, self.chart, comps)
        return __fname__
    
    @property
    def P(self):
        d, ch, c = self.degree, self.chart, self.components
        return ch.dec.P[d](c)

    @property
    def D(self):
        d, ch, c = self.degree, self.chart, self.components
        c = ch.dec.D[d](c)
        if c is 0: return 0
        return F(d+1, ch, c)
    
    @property
    def H(self):
        d, ch, c = self.degree, self.chart, self.components
        c = ch.dec.H[d](c)
        if c is 0: return 0
        dim = ch.dimension
        return F(dim-d, ch, c)

    def W(self, other):
        d1, ch1, c1 = self.degree, self.chart, self.components
        d2, ch2, c2 = other.degree, other.chart, other.components
        assert ch1 == ch2
        return F(d1+d2, ch1, ch1.dec.W[d1, d2](c1, c2))

    def C(self, other):
        d1, ch1, c1 = self.degree, self.chart, self.components
        assert d1 == 1
        d2, ch2, c2 = other.degree, other.chart, other.components
        assert ch1 == ch2
        c = self.chart.dec.C[d2](c1, c2)
        if c == 0: return 0
        return F(d2-1, ch1, c)
    
    def decform(self, g, isprimal):
        import dec.forms
        d, ch, c = self.degree, self.chart, self.components
        assert g.dimension == ch.dimension
        cells = g.cells[d, isprimal]
        
        #Symbolic Integration
        λ = lambdify(ch.simpl_coords(d), self.P, 'numpy')
        if   d == 0 and ch.dimension == 1:
            x0 = cells
            a = λ(x0)
        elif d == 1 and ch.dimension == 1:
            x0, x1 = cells
            a = λ(x0, x1)
        elif d == 0 and ch.dimension == 2:
            x0, y0 = cells
            a = λ(x0, y0)
        elif d == 1 and ch.dimension == 2:
            (x0, y0), (x1, y1) = cells
            a = λ(x0, y0, x1, y1)
        elif d == 2 and ch.dimension == 2:
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = cells
            a = (λ(x0, y0, x1, y1, x2, y2) + 
                 λ(x0, y0, x2, y2, x3, y3))
            
        return dec.forms.decform(d, isprimal, g, a)
    
    @property
    def lambdify_(self):
        if len(self.components) == 1:
            return lambdify(self.chart.coords, self.components[0], 'numpy')
        else:
            return lambdify(self.chart.coords, self.components, 'numpy')

    for m in '''
            __init__
            __eq__
            __ne__
            __repr__
            __rmul__
            __xor__
            __getitem__
            P D H W C
            decform
            '''.split():
        setattr(F, m, locals()[m])
    for m in '''
            __add__
            __radd__
            __sub__
            __rsub__
            __div__
            __truediv__
            '''.split():
        setattr(F, m, binary(m))
    for m in '''
            __neg__
            '''.split():
        setattr(F, m, unary(m))
    setattr(F, 'lambdify', lambdify_)

    return F

def simplified_forms(F, chart):
    '''
    Helper functions to make constructing forms easier.

    >>> chart = Chart(x,)
    >>> F0, F1 = simplified_forms(form, chart)
    >>> assert F0(x) == form(0, chart, (x,))
    >>> assert F1(x) == form(1, chart, (x,))

    >>> chart = Chart(x, y)
    >>> F0, F1, F2 = simplified_forms(form, chart)
    >>> assert F0(x   ) == form(0, chart, (x,  ))
    >>> assert F1(x, y) == form(1, chart, (x, y))
    >>> assert F2(x   ) == form(2, chart, (x,  ))

    #>>> chart = Chart(x, y, z)
    #>>> F0, F1, F2, F3 = simplified_forms(form, chart)
    #>>> assert F0(x      ) == form(0, chart, (x,     ))
    #>>> assert F1(x, y, z) == form(1, chart, (x, y, z))
    #>>> assert F2(x, y, z) == form(2, chart, (x, y, z))
    #>>> assert F3(x      ) == form(3, chart, (x,     ))

    '''
    def getF(deg):
        if deg == 0 or deg == chart.dimension:
            return (lambda f: form(deg, chart, (f,)))
        else:
            return (lambda *f: form(deg, chart, f))
    return tuple(getF(deg) for deg in range(chart.dimension+1))
    
form = form_factory('form')
F0, F1, F2 = simplified_forms(form, Chart(x,y))

################################
# Projections
################################

def P(f):
    '''
    Projection

    Integrate a symbolic form (expressed in terms of coordinates x, y) on the simplices,
    and return the result in terms of simplex coordinates.

    >>> P(F0(x*y))
    x0*y0
    >>> P(F1(x, 0))
    -x0**2/2 + x1**2/2
    >>> P(F1(1, 0))
    -x0 + x1
    >>> P(F1(1, 1))
    -x0 + x1 - y0 + y1
    >>> from sympy import expand
    >>> assert P(F2(1)) == expand( ((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))/2 )
    '''
    return f.P

################################
# Derivative
################################

def D(f):
    '''
    Derivative     
    >>> assert D( F0(x) ) == F1(1, 0)
    >>> assert D( F0(x*y) ) == F1(y, x)
    >>> assert D( F1(-y, x) ) == F2(2)
    >>> assert D( F2(x) ) == 0
    '''
    if f is 0: return 0
    return f.D
 
################################
# Hodge Star
################################

def H(f):
    '''
    Hodge Star
    >>> assert H(F0(x)) == F2(x)
    >>> assert H(F1(x,y)) == F1(-y, x)
    >>> assert H(F2(x)) == F0(x)
    '''
    if f is 0:
        return 0
    return f.H

################################
# Wedge Product
################################

def W(a, b):
    '''
    Wedge Product
    >>> u, v, f, g = symbols('u v f g')
    >>> assert W(F0(f),F0(g)) == F0(f*g)
    >>> assert W(F0(f),F1(u,v)) == F1(f*u, f*v)
    >>> assert W(F1(u,v),F0(f)) == F1(f*u, f*v)
    >>> assert W(F1(u,v),F1(f,g)) == F2(-f*v + g*u)
    >>> assert W(F0(f),F2(g)) == F2(f*g)
    >>> assert W(F2(g),F0(f)) == F2(f*g)
    '''    
    return a.W(b)
    
################################
# Inner Product
################################

def Dot(a, b):
    '''
    Inner Product
    
    The inner product berween forms can be expressed as 

    .. math:: \langle \alpha, \beta \rangle = \star ( \alpha \wedge \star \beta )
    
    >>> Dot(F0(f), F0(g)) == F0(f*g) == Dot(F0(g), F0(f))
    True
    >>> Dot(F1(f,g), F1(u,v)) == F0(f*u + g*v) == Dot(F1(u,v), F1(f,g))
    True
    >>> Dot(F2(f), F2(g)) == F0(f*g) == Dot(F2(g), F2(f))
    True
    '''
    return H(W(a, H(b)))
 
################################
# Contraction
################################

def C(X, f):
    '''
    Contraction
    
    >>> u, v, f, g = symbols('u v f g')

    >>> X = F1(u, v)
    >>> C(X, F1(f,g)) == F0(f*u + g*v)
    True
    >>> C(X, F2(f)) == F1(-f*v, f*u)
    True
    '''
    if X is 0 or f is 0: 
        return 0
    return X.C(f)
 
################################
# Lie Derivative
################################

def Lie(X, f):
    '''
    Lie Derivative
    
    >>> from sympy import expand
    >>> d = diff
    >>> l = lambda f_: Lie(F1(u,v), f_)
    >>> l(F0(f)) == F0( u*d(f, x) + v*d(f, y) )
    True
    >>> l(F2(f)) == F2( expand( d(f*u,x) + d(f*v,y) ) )
    True
    >>> simplify(l(F1(f, g))[0]) == u*d(f,x) + v*d(f,y) + f*d(u,x) + g*d(v,x)
    True
    >>> simplify(l(F1(f, g))[1]) == u*d(g,x) + v*d(g,y) + f*d(u,y) + g*d(v,y)
    True
    >>> simplify(l(F1(u, v))[0]) == expand( d((u**2+v**2)/2, x) + u*d(u, x) + v*d(u, y) )
    True
    >>> simplify(l(F1(u, v))[1]) == expand( d((u**2+v**2)/2, y) + u*d(v, x) + v*d(v, y) )
    True
    '''
    return C(X, D(f)) + D(C(X, f))

################################
# Laplacian
################################

def Laplacian(f):
    '''
    Laplacian Operator

    >>> l = Laplacian
    >>> assert l(F0(f)) == F0( diff(f, x, x) + diff(f, y, y))
    >>> assert l(F1(f,g)) == F1(diff(f, x, x) + diff(f, y, y),
    ...                         diff(g, x, x) + diff(g, y, y))
    >>> assert l(F2(f)) == F2( diff(f, x, x) + diff(f, y, y))
    '''
    return H(D(H(D(f)))) + D(H(D(H(f))))

################################
# Misc
################################

def grad(f):
    '''
    Compute the gradient of a scalar field :math:`f(x,y)`.
    
    >>> assert grad(f) == (diff(f, x), diff(f, y))
    '''
    return tuple(D(F0(f)))

def div(V):
    '''
    Compute the divergence of a vector field :math:`V(x,y)`.
    
    >>> assert div((u,v)) == diff(u, x) + diff(v, y)
    '''
    f = F1(*V)
    return H(D(H(f)))[0]

def vort(V):
    '''
    Compute the vorticity of a vector field :math:`V(x,y)`.
    
    >>> assert vort((u,v)) == -diff(u, y) + diff(v, x)
    '''
    f = F1(*V)
    return H(D(f))[0]

def adv(V):
    '''
    >>> d = diff
    >>> assert simplify(adv((u,v))) == (u*d(u,x)+v*d(u,y), u*d(v,x)+v*d(v,y))
    '''
    G  = grad(V[0]**2 + V[1]**2)
    V_, G_ = F1(*V), F1(*G)
    return tuple(Lie(V_, V_) - G_/2)

def lambdify2():
    '''
    >>> l0, l1 = lambdify2()
    >>> assert l0(x*y)(1, 2) == (lambda x, y: x*y)(1, 2)
    >>> assert l1((x, y))(1, 2) == (lambda x, y: (x,y))(1, 2)
    '''

    def l0(f):
        return lambdify((x,y), f, 'numpy')

    def l1(f):
        def f_(x_, y_, f=f):
            return (lambdify((x,y), f[0], 'numpy')(x_, y_),
                    lambdify((x,y), f[1], 'numpy')(x_, y_))
        return f_

    return l0, l1

def plot(plt, V, p):

    # print(simplify( div(adv(V)) + div(grad(p)) )) # must be zero

    plt.figure(figsize=(8,8))

    scale = [-np.pi, np.pi]
    axes = [
        plt.subplot(221, aspect='equal'),
        plt.subplot(222, aspect='equal'),
        plt.subplot(223, aspect='equal'),
        plt.subplot(224, aspect='equal')]
    for ax in axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(scale)
        ax.set_ylim(scale)

    n = 20
    X, Y = np.meshgrid(
               np.linspace(scale[0], scale[1], n),
               np.linspace(scale[0], scale[1], n))
    u, v = lambdify((x,y), V, 'numpy')(X, Y)
    axes[0].quiver(X, Y, u + 0*X, v + 0*X)
    axes[0].set_title(r'$\mathbf{v}(x,y)$')

    vdot = [simplify(-adv(V)[0] - grad(p)[0]),
            simplify(-adv(V)[1] - grad(p)[1])]
    udot, vdot = lambdify((x,y), vdot, 'numpy')(X, Y)
    udot = udot + 0*X; vdot = vdot + 0*X;
    axes[2].quiver(X, Y, udot, vdot)
    axes[2].set_title(r'$\mathbf{\dot{v}}(x,y)$')

    omega = simplify(vort(V))

    n = 200
    X, Y = np.meshgrid(
               np.linspace(scale[0], scale[1], n),
               np.linspace(scale[0], scale[1], n))
    Omega = lambdify((x,y), omega, 'numpy')(X, Y) + 0*X
    axes[1].contourf(X, Y, Omega)
    axes[1].set_title(r'$\omega(x,y)$')

    P = lambdify((x,y), p, 'numpy')(X, Y) + 0*X
    axes[3].contourf(X, Y, P)
    axes[3].set_title(r'$p(x,y)$')

# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     for V_, p_ in zip(V, p):
#         plot(plt, V_, p_)
#     plt.show()
