'''
    Module for symbolic computations.    
'''
import numpy as np
from sympy import (symbols, Function, diff, lambdify, simplify,
                   sympify,
                   integrate, Integral,
                   sin, cos)

# Coordinates
x, y = symbols('x y')
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

def form_factory(name):
    '''
    >>> form = form_factory('form')

    >>> f, g, u, v = symbols('f g u v')

    >>> α = form(1, (f, g))
    >>> φ = form(1, (u, v))
    
    >>> -φ
    form(1, (-u, -v))
    >>> φ + φ
    form(1, (2*u, 2*v))
    >>> φ + α
    form(1, (f + u, g + v))
    >>> φ - α
    form(1, (-f + u, -g + v))
    
    
    We can use ^ as the wedge product operator.    

    >>> φ ^ φ == form(2, (0,))
    True
    >>> α ^ φ == - φ ^ α
    True
    
    '''
    
    F = type(name, (object,), {})
    
    def __init__(self, degree, components):
        comp = tuple(sympify(_) for _ in components)
        self.components = comp
        self.degree = degree

    def __repr__(self):
        t = (self.degree, self.components)
        return name + t.__repr__()

    def __eq__(self, other):
        return self.degree == other.degree and self.components == other.components
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __rmul__(self, other):
        if callable(other):
            return other(self)
        else:
            return type(self)(self.degree, (c.__rmul__(other) for c in self.components))

    def __xor__(self, other):
        return W(self, other)
    
    def __getitem__(self, k):
        return self.components[k]
        
    def binary(name):
        def __fname__(self, other):
            if type(other) is type(self):
                assert self.degree == other.degree
                comps = tuple(getattr(s, name)(o) for s, o in zip(self.components, other.components))
            else:    
                comps = tuple(getattr(s, name)(other) for s in self)
            return F(self.degree, comps)
        return __fname__

    def unary(name):
        def __fname__(self):
            comps = tuple(getattr(s, name)() for s in self.components)
            return F(self.degree, comps)
        return __fname__

    for m in '''
            __init__
            __eq__
            __ne__
            __repr__
            __rmul__
            __xor__
            __getitem__
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

    return F

def simplified_forms(F):
    '''
    >>> F0, F1, F2 = simplified_forms(form)
    >>> F0(x   ) == form(0, (x,  ))
    True
    >>> F1(x, y) == form(1, (x, y))
    True
    >>> F2(x   ) == form(2, (x,  ))
    True
    '''

    def factory(degree):
        name = 'F{}'.format(degree)
        def __init__(self, *args):  
            F.__init__(self, degree, args)
        if degree == 1:
            def rep(self): 
                return  '{}{}'.format(name, tuple(self))
        else:            
            def rep(self): 
                return '{}({})'.format(name, self[0])
        return type(name, (F,), {'__init__':__init__, '__repr__':rep})
    
    return factory(0), factory(1), factory(2)

form = form_factory('form')
F0, F1, F2 = simplified_forms(form)

#################################
# Multiple dispatch
#################################

def multipledispatch(T):    
    def apply_decorator(dispatch_fn):
        __multi__ = {}
        def _inner(*args):
            d = tuple(f.degree     for f in args)
            c = tuple(f.components for f in args)
            d_ = dispatch_fn(*d)
            c_ = __multi__[d](*c)
            return T(d_, c_)
        _inner.__multi__ = __multi__
        _inner.__default__ = None
        return _inner
    return apply_decorator

def register(dispatch_fn, *dispatch_key):
    def apply_decorator(fn):
        if not dispatch_key:
            dispatch_fn.__default__ = fn
        else:
            dispatch_fn.__multi__[dispatch_key] = fn
    return apply_decorator

################################
# Derivative
################################

def D(f):
    '''
    Derivative
     
    >>> D( F0(x) )
    F1(1, 0)
    >>> D( F0(x*y) )
    F1(y, x)
    >>> D( F1(-y, x) )
    F2(2)
    >>> D( F2(x) )
    0
    '''

    if f is 0:
        return 0
    
    if f.degree is 0:
        f ,= f
        return F1(diff(f, x), diff(f, y))
    
    if f.degree is 1:
        fx, fy = f
        return F2(-diff(fx, y) + diff(fy, x))    
    
    if f.degree is 2:
        return 0
 
################################
# Wedge Product
################################

def W(a, b):
    '''
    Wedge Product
    
    >>> u, v, f, g = symbols('u v f g')
    
    >>> W(F0(f),F0(g))
    F0(f*g)
    >>> W(F0(f),F1(u,v))
    F1(f*u, f*v)
    >>> W(F1(u,v),F0(f))
    F1(f*u, f*v)
    >>> W(F1(u,v),F1(f,g))
    F2(-f*v + g*u)
    >>> W(F0(f),F2(g))
    F2(f*g)
    >>> W(F2(g),F0(f))
    F2(f*g)
    '''
    
    deg = (a.degree, b.degree)
    
    if deg == (0, 0):
        a, = a
        b, = b
        return F0(a*b)

    if deg == (0, 1):
        a, = a
        bx, by = b
        return F1(a*bx, a*by)
     
    if deg == (0, 2):
        a, = a
        b, = b
        return F2(a*b)
     
    if deg == (1, 1):
        ax, ay = a
        bx, by = b
        return F2(ax*by - ay*bx)

    if deg == (1, 0):
        return W(b, a)

    if deg == (2, 0):
        return W(b, a)
 
################################
# Hodge Star
################################

def H(f):
    '''
    Hodge Star

    >>> H(F0(x))
    F2(x)
    >>> H(F1(x,y))
    F1(-y, x)
    >>> H(F2(x))
    F0(x)
    '''

    if f is 0:
        return 0
 
    if f.degree is 0:
        f, = f
        return F2(f)
      
    if f.degree is 1:
        fx, fy = f
        return F1(-fy, fx)
      
    if f.degree is 2:
        f, = f
        return F0(f)

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
    r'''
    Contraction
    
    >>> u, v, f, g = symbols('u v f g')

    >>> X = F1(u, v)
    >>> C(X, F1(f,g))
    F0(f*u + g*v)
    >>> C(X, F2(f))
    F1(-f*v, f*u)
    '''

    assert X.degree is 1
    Xx, Xy = X

    if f is 0:
        return 0

    if f.degree is 0:
        return 0

    if f.degree is 1:
        fx, fy = f
        return F0(Xx*fx + Xy*fy)
      
    if f.degree is 2:
        f, = f
        return F1(-Xy*f, Xx*f )
 
#TODO: Delete this
def contractions(X):
    '''
    .. warning::
        Deprecated
 
    >>> C1, C2 = contractions((u,v))
    >>> C1((f,g)) == f*u + g*v
    True
    >>> C2(f) == (-f*v, f*u)
    True
    '''
    def C1(f):
        return X[0]*f[0] + X[1]*f[1]
    def C2(f):
        return (-f*X[1], f*X[0])
    return C1, C2

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
    >>> l(F0(f)) == F0( diff(f, x, x) + diff(f, y, y))
    True
    >>> l(F1(f,g)) == F1(diff(f, x, x) + diff(f, y, y),
    ...                  diff(g, x, x) + diff(g, y, y))
    True
    >>> l(F2(f)) == F2( diff(f, x, x) + diff(f, y, y))
    True
    '''
    return H(D(H(D(f)))) + D(H(D(H(f))))

################################
# Projections
################################

def P0(f):
    return f[0].subs({x:x0, y:y0})
 
def P1(f):
    #ux, uy = sympify(f[0]), sympify(f[1])
    ux, uy = f
    s = symbols('s')
    lx, ly = x1 - x0, y1 - y0
    subst = ((x, x0*(1-s) + x1*s),
             (y, y0*(1-s) + y1*s))
    integrand = (ux.subs(subst)*lx +
                 uy.subs(subst)*ly)
    iexpr = integrate(integrand,  (s, 0, 1))
    if iexpr.has(Integral):
        raise ValueError('Unable to evaluate {}.'.format(iexpr))
    return iexpr
 
def P2(f):
    omega = sympify(f[0])
    s, t = symbols('s t')
    A = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)
    subst = ((x, x0*(1-s-t) + x1*s + x2*t),
             (y, y0*(1-s-t) + y1*s + y2*t))
    integrand = (omega.subs(subst)*A)
    iexpr = integrate(integrand, (t, 0, 1-s), (s, 0, 1))
    if iexpr.has(Integral):
        raise ValueError('Unable to evaluate {}.'.format(iexpr))
    return iexpr

def P(f):
    '''
    Projections 

    Integrate a symbolic form (expressed in terms of coordinates x, y) on the simplices,
    and return the result in terms of simplex coordiates.

    >>> P(F0(x*y))
    x0*y0
    >>> P(F1(x, 0))
    -x0**2/2 + x1**2/2
    >>> P(F1(1, 0))
    -x0 + x1
    >>> P(F1(1, 1))
    -x0 + x1 - y0 + y1
    >>> from sympy import expand
    >>> P(F2(1)) == expand( ((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))/2 )
    True
    '''
    return {0: P0, 
            1: P1, 
            2: P2}[f.degree](f)

################################
# Misc
################################

def grad(f):
    '''
    Compute the gradient of a scalar field :math:`f(x,y)`.
    
    >>> grad(f) == (diff(f, x), diff(f, y))
    True
    '''
    return tuple(D(F0(f)))

def div(V):
    '''
    Compute the divergence of a vector field :math:`V(x,y)`.
    
    >>> div((u,v)) == diff(u, x) + diff(v, y)
    True
    '''
    f = F1(*V)
    return H(D(H(f)))[0]

def vort(V):
    '''
    Compute the vorticity of a vector field :math:`V(x,y)`.
    
    >>> vort((u,v)) == -diff(u, y) + diff(v, x)
    True
    '''
    f = F1(*V)
    return H(D(f))[0]

def adv(V):
    '''
    >>> d = diff
    >>> simplify(adv((u,v))) == (u*d(u,x)+v*d(u,y), u*d(v,x)+v*d(v,y))
    True
    '''
    G  = grad(V[0]**2 + V[1]**2)
    V_, G_ = F1(*V), F1(*G)
    return tuple(Lie(V_, V_) - G_/2)

def projections1d():
    '''
    >>> P0, P1 = projections1d()
    >>> P0(x) == x0
    True
    >>> P1(x) == x1**2/2 - x0**2/2
    True
    >>> P1(1) == x1 - x0
    True
    '''
    x0, x1 = symbols('x0 x1')

    def P0(f):
        f = sympify(f)
        return f.subs(x, x0)

    def P1(f):
        f = sympify(f)
        iexpr = integrate(f, (x, x0, x1))
        if iexpr.has(Integral):
            raise ValueError('Unable to evaluate {}.'.format(iexpr))
        return iexpr

    return P0, P1

def lambdify2():
    '''
    >>> l0, l1 = lambdify2()
    >>> l0(x*y)(1, 2) == (lambda x, y: x*y)(1, 2)
    True
    >>> l1((x, y))(1, 2) == (lambda x, y: (x,y))(1, 2)
    True
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
