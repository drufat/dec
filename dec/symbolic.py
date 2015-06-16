'''
    Module for symbolic computations.    
'''
import numpy as np
from sympy import (symbols, Function, diff, lambdify, simplify,
                   sin, cos, sqrt)
from dec.helper import bunch
from dec.symform import form
from dec.integrate import enumerate_coords, integration_1d, integration_2d

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

class Chart(object):
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
                        H=hodge_star_1d(),
                        W=wedge_1d(),
                        C=contraction_1d(),
                        )
        elif len(coords) == 2:
            x, y = coords
            dec = bunch(D=derivative_2d(x, y),
                        H=hodge_star_2d(),
                        W=wedge_2d(),
                        C=contraction_2d(),
                        )
        else:
            raise NotImplementedError
        self.dec = dec
    
    def cell_coords(self, deg):
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

def simplex_measure(σ):

    n, k = len(σ[0]), len(σ)
    assert all(n==len(s) for s in  σ)
    
    if n == 1:
        if k == 1:
            return 1
        if k == 2:
            ((x0,), (x1,)) = σ
            return x1 - x0

    if n == 2:
        if k == 1:
            return 1
        if k == 2:
            ((x0,y0), (x1,y1)) = σ
            return sqrt((x1 - x0)**2 + (y1 -y0)**2)
        if k == 3:
            ((x0,y0), (x1,y1), (y1,y2)) = σ
            return ((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))/2

    if n == 3:
        if k == 1:
            return 1
        if k == 2:
            ((x0,y0,z0), (x1,y1,z1)) = σ
            return sqrt((x1 - x0)**2 + (y1 -y0)**2 + (z1 -z0)**2)
        if k == 3:
            ((x0,y0,z0), (x1,y1,z1), (x2,y2,z2)) = σ
            raise NotImplementedError
        if k == 4:
            ((x0,y0,z0), (x1,y1,z1), (x2,y2,z2), (x3,y3,z3)) = σ
            raise NotImplementedError

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

def simplified_forms(F, chart):
    '''
    Helper functions to make constructing forms easier.

    >>> grid = Chart(x,)
    >>> F0, F1 = simplified_forms(form, grid)
    >>> assert F0(x) == form(0, grid, (x,))
    >>> assert F1(x) == form(1, grid, (x,))

    >>> grid = Chart(x, y)
    >>> F0, F1, F2 = simplified_forms(form, grid)
    >>> assert F0(x   ) == form(0, grid, (x,  ))
    >>> assert F1(x, y) == form(1, grid, (x, y))
    >>> assert F2(x   ) == form(2, grid, (x,  ))

    #>>> grid = Chart(x, y, z)
    #>>> F0, F1, F2, F3 = simplified_forms(form, grid)
    #>>> assert F0(x      ) == form(0, grid, (x,     ))
    #>>> assert F1(x, y, z) == form(1, grid, (x, y, z))
    #>>> assert F2(x, y, z) == form(2, grid, (x, y, z))
    #>>> assert F3(x      ) == form(3, grid, (x,     ))

    '''
    def getF(deg):
        if deg == 0 or deg == chart.dimension:
            return (lambda f: form(deg, chart, (f,)))
        else:
            return (lambda *f: form(deg, chart, f))
    return tuple(getF(deg) for deg in range(chart.dimension+1))
    
F0, F1, F2 = simplified_forms(form, Chart(x,y))

################################
# Projections
################################

# def P(f):
#     '''
#     Projection
# 
#     Integrate a symbolic form (expressed in terms of coordinates x, y) on the simplices,
#     and return the result in terms of simplex coordinates.
# 
#     >>> P(F0(x*y))
#     x0*y0
#     >>> P(F1(x, 0))
#     -x0**2/2 + x1**2/2
#     >>> P(F1(1, 0))
#     -x0 + x1
#     >>> P(F1(1, 1))
#     -x0 + x1 - y0 + y1
#     >>> from sympy import expand
#     >>> assert P(F2(1)) == expand( ((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))/2 )
#     '''
#     return f.P

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
    
    The inner product between forms can be expressed as 

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

#if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     for V_, p_ in zip(V, p):
#         plot(plt, V_, p_)
#     plt.show()
