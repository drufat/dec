'''
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
    -----------
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

_class_template = '''
class {typename}(np.ndarray):

    def __new__(cls, *args):
        return np.asarray(args).view(cls)
        #return tuple.__new__(cls, args)

    def __repr__(self):
        t = tuple(self)
        if len(t) == 1:
            s = '(' + t[0].__repr__() + ')'
        else:
            s = t.__repr__()
        return '{typename}' + s
'''

def formtuple(typename, verbose=False):

    class_definition = _class_template.format(typename=typename)
    namespace = dict(__name__='formtuple_{}'.format(typename), np=np)
    exec(class_definition, namespace)
    result = namespace[typename]
    if verbose:
        print(result._source)
    return result


F0 = formtuple('F0')
F1 = formtuple('F1')
F2 = formtuple('F2')

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

@d_(F0, F0)
def W(α, β):
    return α*β

@d_(F0, F1)
def W(α, β):
    α, = α
    βx, βy = β
    return F1(α*βx, α*βy)

@d_(F0, F2)
def W(α, β):
    return β*α

@d_(F1, F0)
def W(α, β):
    return W(β, α)

@d_(F1, F1)
def W(α, β):
    αx, αy = α
    βx, βy = β
    return F2(αx*βy-βx*αy)

@d_(F2, F0)
def W(α, β):
    return W(β, α)

################################

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
    Deprecated !!!
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

def Lie(X, f):
    '''
    >>> from sympy import expand
    >>> d = diff
    >>> l = lambda f_: Lie(F1(u,v), f_)
    >>> l(F0(f)) == F0( u*d(f, x) + v*d(f, y) )
    F0(True)
    >>> l(F2(f)) == F2( expand( d(f*u,x) + d(f*v,y) ) )
    F2(True)
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

def Laplacian(f):
    '''
    >>> l = Laplacian
    >>> l(F0(f)) == F0( diff(f, x, x) + diff(f, y, y))
    F0(True)
    >>> l(F1(f,g)) == F1(diff(f, x, x) + diff(f, y, y),
    ...                  diff(g, x, x) + diff(g, y, y))
    F1(True, True)
    >>> l(F2(f)) == F2( diff(f, x, x) + diff(f, y, y))
    F2(True)
    '''
    return H(D(H(D(f)))) + D(H(D(H(f))))

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
    >>> simplify(adv((u,v))) == (u*diff(u,x)+v*diff(u,y), u*diff(v,x)+v*diff(v,y))
    True
    '''
    V_ = F1(*V)
    return tuple(Lie(V_, V_) + grad(-(V[0]**2+V[1]**2)/2))

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

def projections2d():
    '''
    Integrate a symbolic form (expressed in terms of coordinates x, y) on the simplices,
    and return the result in terms of simplex coordiates.

    >>> P0, P1, P2 = projections2d()
    >>> P0(x*y) == x0*y0
    True
    >>> P1((x, 0)) == x1**2/2 - x0**2/2
    True
    >>> P1((1, 0)) == x1 - x0
    True
    >>> P1((1, 1)) == y1 - y0 + x1 - x0
    True
    >>> from sympy import expand
    >>> P2(1) == expand( ((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))/2 )
    True
    '''

    def P0(f):
        return f.subs({x:x0, y:y0})

    def P1(f):
        ux, uy = sympify(f[0]), sympify(f[1])
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
        omega = sympify(f)
        s, t = symbols('s t')
        A = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)
        subst = ((x, x0*(1-s-t) + x1*s + x2*t),
                 (y, y0*(1-s-t) + y1*s + y2*t))
        integrand = (omega.subs(subst)*A)
        iexpr = integrate(integrand, (t, 0, 1-s), (s, 0, 1))
        if iexpr.has(Integral):
            raise ValueError('Unable to evaluate {}.'.format(iexpr))
        return iexpr

    return P0, P1, P2

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

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    for V_, p_ in zip(V, p):
        plot(plt, V_, p_)
    plt.show()
