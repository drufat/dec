from sympy import *
import numpy as np

x, y = symbols('x y')

V1 = (-2*sin(y)*cos(x/2)**2,
       2*sin(x)*cos(y/2)**2)
V2 = (-sin(y), sin(x))
V3 = (-sin(2*y), sin(x))
V4 = (1, 0)

p1 = (cos(2*x)*(5+4*cos(y))+5*(4*cos(y)+cos(2*y))+4*cos(x)*(5+5*cos(y)+cos(2*y)))/20
p2 = cos(x)*cos(y)
p3 = 4*cos(x)*cos(2*y)/5
p4 = 0

def div(V):
    '''
    Compute the divergence of a vector field :math:`V(x,y)`.
    '''
    return diff(V[0], x) + diff(V[1], y)

def vort(V):
    '''
    Compute the vorticity of a vector field :math:`V(x,y)`.
    '''
    return diff(V[1], x) - diff(V[0], y)

def grad(f):
    '''
    Compute the gradient of a scalar field :math:`f(x,y)`.
    '''
    return (diff(f, x), diff(f, y))

def lie0(V, f):
    return V[0]*diff(f, x) + V[1]*diff(f, y)

def lie1(U, V):
    raise NotImplemented

def lie2(V, omega):
    raise NotImplemented

def adv(V):
    Ax = [V[0]* diff(Vc, x) for Vc in V]
    Ay = [V[1]* diff(Vc, y) for Vc in V]
    return (Ax[0]+Ay[0], Ax[1]+Ay[1])

def comm(U, V):
    raise NotImplemented

def cross(U, V):
    return -U[0]*V[1] + U[1]*V[0]

def dot(U, V):
    return U[0]*V[0] + U[1]*V[1]

def dens(V):
    return dot(V, V)/2

def laplace(f):
    return diff(diff(f, x), x) + diff(diff(f, y), y)

def plot(plt, V, p):

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
    axes[0].quiver(X, Y, u, v)
    axes[0].set_title(r'$\mathbf{v}(x,y)$')

    vdot = [simplify(-adv(V)[0] + grad(p)[0]),
            simplify(-adv(V)[1] + grad(p)[1])]
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
