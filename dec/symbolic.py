r'''
    Module for symbolic computations.

    >>> u, v, f, g = symbols('u v f g')
    
    Derivative
    -----------
    >>> D( F0(x) )
    F1(1, 0)
    >>> D( F0(x*y) )
    F1(y, x)
    >>> D( F1(-y, x) )
    F2(2)
    >>> D( F2(x) )
    0

    Wedge
    ------
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

    Hodge Star
    -----------
    >>> H(F0(x))
    F2(x)
    >>> H(F1(x,y))
    F1(-y, x)
    >>> H(F2(x))
    F0(x)

    Contraction
    ------------
    >>> X = F1(u, v)
    >>> C(X, F1(f,g))
    F0(f*u + g*v)
    >>> C(X, F2(f))
    F1(-f*v, f*u)
    
    
    Inner Product
    --------------
    The inner product berween forms can be expressed as 

    .. math:: \alpha \wedge \star \beta = \langle \alpha, \beta \rangle \omega
        
    where :math:`\omega` is the volume form and :math:`\langle\cdot,\cdot\rangle`
    represent an inner product between forms.
    
    >>> W(F0(f), H(F0(g)))
    F2(f*g)
    >>> W(F1(f,g), H(F1(u,v)))
    F2(f*u + g*v)
    >>> W(F2(f), H(F2(g)))
    F2(f*g)

    Projections 
    ------------
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

    Other Functions
    ----------------
    
'''
from dec import d_
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
    
    #>>> φ ^ φ
    #form(2, (0,))
    #>>> α ^ φ
    #form(2, (-f*v + g*u,))
    #>>> φ ^ α
    #form(2, (f*v - g*u,))
    
    '''
    
    def __new__(cls, degree, components):
        comp = tuple(sympify(_) for _ in components)
        obj = tuple.__new__(cls, comp)
        obj.degree = degree
        return obj

    def __repr__(self):
        t = (self.degree, tuple(self))
        return name + t.__repr__()

    def __eq__(self, other):
        return self.degree == other.degree and tuple(self) == tuple(other)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __rmul__(self, other):
        if callable(other):
            return other(self)
        return NotImplemented

    def __xor__(self, other):
        return W(self, other)
        
    def binary(name):
        def __fname__(self, other):
            if type(other) is type(self):
                assert self.degree == other.degree
                comps = tuple(getattr(s, name)(o) for s, o in zip(self, other))
            else:    
                comps = tuple(getattr(s, name)(other) for s in self)
            return form(self.degree, comps)
        return __fname__

    def unary(name):
        def __fname__(self):
            comps = tuple(getattr(s, name)() for s in self)
            return form(self.degree, comps)
        return __fname__

    methods = {}
    for m in '''
            __new__
            __eq__
            __ne__
            __repr__
            __rmul__
            __xor__
            '''.split():
        methods[m] = locals()[m]
    for m in '''
            __add__
            __radd__
            __sub__
            __rsub__
            __div__
            __truediv__
            '''.split():
        methods[m] = binary(m)
    for m in '''
            __neg__
            '''.split():
        methods[m] = unary(m)

    return type(name, (tuple,), methods)

form = form_factory('form')

def simplified_forms(F):

    class F0(F):
        def __new__(cls, *args):  return super().__new__(cls, 0, args)
        def __repr__(self): return 'F0(' + self[0].__repr__() + ')'
    
    class F1(F):
        def __new__(cls, *args):  return super().__new__(cls, 1, args)
        def __repr__(self): return 'F1' + tuple(self).__repr__()
    
    class F2(F):
        def __new__(cls, *args):  return super().__new__(cls, 2, args)
        def __repr__(self): return 'F2(' + self[0].__repr__() + ')'
    
    return F0, F1, F2

F0, F1, F2 = simplified_forms(form)

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

# @multipledispatch(form)
# def D(k):
#     return k+1
# 
# @register(D, 0)
# def _(f):
#     f, = f
#     return (diff(f, x), diff(f, y))
#     
# @register(D, 1)
# def _(f):
#     fx, fy = f
#     return -diff(fy, x) + diff(fx, y),
# 
# @register(D, 2)
# def _(f):
#     return 0,
 
@d_(F0)
def D(f):
    f ,= f
    return F1(diff(f, x), diff(f, y))
  
@d_(F1)
def D(f):
    fx, fy = f
    return F2(-diff(fx, y) + diff(fy, x))
  
@d_(F2)
def D(f):
    return 0
  
@d_(object)
def D(f):
    assert f == 0
    return 0
 
################################
# Wedge Product
################################

# @multipledispatch(form)
# def W(k, m):
#     return k + m
# 
# @register(W, 0, 0)
# def _(a, b):
#     a, = a
#     b, = b
#     return a*b,
# 
# @register(W, 0, 1)
# def _(a, b):
#     a, = a
#     bx, by = b
#     return a*bx, a*by
# 
# @register(W, 0, 2)
# def _(a, b):
#     a, = a
#     b, = b
#     return a*b
# 
# @register(W, 1, 1)
# def _(a, b):
#     ax, ay = a
#     bx, by = b
#     return -ax*by + ay*bx,

@d_(F0, F0)
def W(α, β):
    α, = α
    β, = β
    return F0(α*β)
  
@d_(F0, F1)
def W(α, β):
    α, = α
    βx, βy = β
    return F1(α*βx, α*βy)
  
@d_(F0, F2)
def W(α, β):
    α, = α
    β, = β
    return F2(α*β)
  
@d_(F1, F1)
def W(α, β):
    αx, αy = α
    βx, βy = β
    return F2(αx*βy-βx*αy)
  
@d_(F1, F0)
def W(α, β):
    return W(β, α)
  
@d_(F2, F0)
def W(α, β):
    return W(β, α)
 
################################
# Hodge Star
################################

# @multipledispatch(form)
# def H(k):
#     return 2 - k
# 
# @register(H, 0)
# def _(f):
#     f, = f
#     return f,
# 
# @register(H, 1)
# def _(f):
#     fx, fy = f
#     return -fy, fx
# 
# @register(H, 2)
# def _(f):
#     f, = f
#     return f,
 
@d_(F0)
def H(f):
    f, = f
    return F2(f)
  
@d_(F1)
def H(f):
    fx, fy = f
    return F1(-fy, fx)
  
@d_(F2)
def H(f):
    f, = f
    return F0(f)
  
@d_(object)
def H(f):
    assert f == 0
    return 0
 
################################
# Contraction
################################

# @multipledispatch(form)
# def C(k, l):
#     assert k == 1
#     return l - 1
# 
# @register(C, 1, 1)
# def _(X, f):
#     Xx, Xy = X
#     fx, fy = f
#     return Xx*fx + Xy*fy,
# 
# @register(C, 1, 2)
# def _(X, f):
#     Xx, Xy = X
#     f, = f
#     return -Xy*f, Xx*f

@d_(F1, F0)
def C(X, f):
    return 0
  
@d_(F1, F1)
def C(X, f):
    return F0( X[0]*f[0] + X[1]*f[1] )
  
@d_(F1, F2)
def C(X, f):
    return F1( -X[1]*f[0], X[0]*f[0] )
  
@d_(F1, object)
def C(X, f):
    assert f == 0
    return 0
 
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

@d_(F0)
def P(f):
    return f[0].subs({x:x0, y:y0})
 
@d_(F1)
def P(f):
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
 
@d_(F2)
def P(f):
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
