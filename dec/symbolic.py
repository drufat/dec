from sympy import *
import numpy as np

x, y = symbols('x y')
u, v = Function('u')(x,y), Function('v')(x,y)
f, g = Function('f')(x,y), Function('g')(x,y)
D = diff

V1 = (-2*sin(y)*cos(x/2)**2,
       2*sin(x)*cos(y/2)**2)
V2 = (-sin(y), sin(x))
V3 = (-sin(2*y), sin(x))
V4 = (1, 0)

p1 = (-cos(2*x)*(5+4*cos(y))-5*(4*cos(y)+cos(2*y))-4*cos(x)*(5+5*cos(y)+cos(2*y)))/20
p2 = -cos(x)*cos(y)
p3 = -4*cos(x)*cos(2*y)/5
p4 = 0

def derivatives():
    '''
    >>> D0, D1 = derivatives()
    >>> D0(x)
    (1, 0)
    >>> D0(x*y)
    (y, x)
    >>> D1([-y, x])
    2
    '''
    def D0(f):
        return (diff(f, x), diff(f, y))
    def D1(f):
        return -diff(f[0], y) + diff(f[1], x)
    return D0, D1

def hodge_stars():
    '''
    >>> H0, H1, H2 = hodge_stars()
    >>> H0(x)
    x
    >>> H1([x,y])
    (-y, x)
    '''
    def H0(f): return f
    def H1(f): return (-f[1], f[0])
    def H2(f): return f
    return H0, H1, H2

def contractions(X):    
    '''
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

def lie_derivatives(X):
    '''
    >>> L0, L1, L2 = lie_derivatives((u,v))
    >>> L0(f) == u*D(f, x) + v*D(f, y)
    True
    >>> L2(f) == expand( D(f*u,x) + D(f*v,y) )
    True
    >>> simplify(L1((f, g))[0]) == u*D(f,x) + v*D(f,y) + f*D(u,x) + g*D(v,x)
    True
    >>> simplify(L1((f, g))[1]) == u*D(g,x) + v*D(g,y) + f*D(u,y) + g*D(v,y)
    True
    >>> simplify(L1((u, v))[0]) == expand( D((u**2+v**2)/2, x) + u*D(u, x) + v*D(u, y) )
    True
    >>> simplify(L1((u, v))[1]) == expand( D((u**2+v**2)/2, y) + u*D(v, x) + v*D(v, y) )
    True
    '''
    D0, D1 = derivatives()
    C1, C2 = contractions(X)
    
    def L0(f): return C1(D0(f))
    def L1(f): return plus(C2(D1(f)), D0(C1(f)))
    def L2(f): return D1(C2(f))

    return L0, L1, L2

def plus(a, b): return (a[0]+b[0], a[1]+b[1])

def laplacians():
    '''
    >>> l0, l1, l2 = laplacians()
    >>> f, g = Function('f')(x,y), Function('g')(x,y)
    >>> l0(f) == D(f, x, x) + D(f, y, y)
    True
    >>> l2(f) == D(f, x, x) + D(f, y, y)
    True
    >>> l1((f,g)) == (D(f, x, x) + D(f, y, y), D(g, x, x) + D(g, y, y))
    True
    '''
    D0, D1 = derivatives()
    H0, H1, H2 = hodge_stars()
    
    def l0(f): return H2(D1(H1(D0(f))))
    def l1(f): return plus( H1(D0(H2(D1(f)))), D0(H2(D1(H1(f)))) )
    def l2(f): return D1(H1(D0(H2(f))))
    
    return l0, l1, l2

def grad(f):
    '''
    Compute the gradient of a scalar field :math:`f(x,y)`.
    '''
    D0, D1 = derivatives()
    return D0(f)

def div(V):
    '''
    Compute the divergence of a vector field :math:`V(x,y)`.
    >>> div((u,v)) == D(u, x) + D(v, y)
    True
    '''
    D0, D1 = derivatives()
    H0, H1, H2 = hodge_stars()
    return H2((D1(H1(V))))

def vort(V):
    '''
    Compute the vorticity of a vector field :math:`V(x,y)`.
    >>> vort((u,v)) == -D(u, y) + D(v, x)
    True
    '''
    D0, D1 = derivatives()
    H0, H1, H2 = hodge_stars()
    return H2(D1(V))
    
def adv(V):
    '''
    >>> simplify(adv((u,v))) == (u*D(u,x)+v*D(u,y), u*D(v,x)+v*D(v,y))
    True
    '''
    L1 = lie_derivatives(V)[1]    
    return plus( L1(V), grad(-(V[0]**2+V[1]**2)/2) )

def plot(plt, V, p):

    #print(simplify( div(adv(V)) + div(grad(p)) ))

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
