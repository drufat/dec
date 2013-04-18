"""
Spectral DEC in 2D
=============================
"""
from __future__ import division
from numpy import *
from dec.helper import *
from dec.spectral import *
import scipy.integrate

def cartesian_product(X, Y):
    """
    >>> cartesian_product([0,1],[2,3])
    (array([0, 1, 0, 1]), array([2, 2, 3, 3]))
    """
    X = asarray(X)
    Y = asarray(Y)
    X, Y = map(lambda x: x.flatten(), meshgrid(X, Y))
    return X, Y

Grid_2D_Interface = """
    pnts, 
    verts, verts_dual,
    edges, edges_dual,
    faces, faces_dual,
    basis_fn,
    B0, B1, B2, B0d, B1d, B2d,
    projection,
    P0, P1, P2, P0d, P1d, P2d,
    reconstruction,
    R0, R1, R2, R0d, R1d, R2d, 
    derivative,
    D0, D1, D0d, D1d,
    hodge_star,
    H0, H1, H2, H0d, H1d, H2d,
    gx, gy
    """

def Grid_2D_Cartesian(gx, gy):
    
    dimension = gx.dimension + gy.dimension
    
    # For all meshgrids hence forth, first argument should have an x, second should have a y    
    verts = meshgrid(gx.verts, gy.verts)
    verts_dual = meshgrid(gx.verts_dual, gy.verts_dual)    
    edges = ((meshgrid(gx.edges[0], gy.verts),
              meshgrid(gx.edges[1], gy.verts)),
             (meshgrid(gx.verts, gy.edges[0]),
              meshgrid(gx.verts, gy.edges[1])))
    edges_dual = ((meshgrid(gx.edges_dual[0], gy.verts_dual),
                   meshgrid(gx.edges_dual[1], gy.verts_dual)),
                  (meshgrid(gx.verts_dual, gy.edges_dual[0]),
                   meshgrid(gx.verts_dual, gy.edges_dual[1])))
    faces = (meshgrid(gx.edges[0], gy.edges[0]),
             meshgrid(gx.edges[1], gy.edges[0]),
             meshgrid(gx.edges[1], gy.edges[1]),
             meshgrid(gx.edges[0], gy.edges[1]))
    faces_dual = (meshgrid(gx.edges_dual[0], gy.edges_dual[0]),
                  meshgrid(gx.edges_dual[1], gy.edges_dual[0]),
                  meshgrid(gx.edges_dual[1], gy.edges_dual[1]),
                  meshgrid(gx.edges_dual[0], gy.edges_dual[1]))

    P0  = lambda f: f(*verts)
    P0d = lambda f: f(*verts_dual)
    P2 = lambda f: integrate_2form(faces, f)[0]
    P2d = lambda f: integrate_2form(faces_dual, f)[0]
    P1  = lambda f: (integrate_1form(edges[0], f)[0],
                     integrate_1form(edges[1], f)[0])
    P1d = lambda f: (integrate_1form(edges_dual[0], f)[0],
                     integrate_1form(edges_dual[1], f)[0])
    def projection():
        return P0, P1, P2, P0d, P1d, P2d
    
    def basis_fn():
        vec = vectorize(lambda u, v: (lambda x, y: (u(x)*v(y))))
        mg = lambda x, y: meshgrid(x, y, copy=False, sparse=False)
        B0  = vec(*mg(gx.B0,  gy.B0))
        B0d = vec(*mg(gx.B0d, gy.B0d))
        B2  = vec(*mg(gx.B1,  gy.B1))
        B2d = vec(*mg(gx.B1d, gy.B1d))

        fx = vectorize(lambda u, v: (lambda x, y: (u(x)*v(y), 0)))
        fy = vectorize(lambda u, v: (lambda x, y: (0, u(x)*v(y))))
        B1  = (fx(*mg(gx.B1, gy.B0)),
               fy(*mg(gx.B0, gy.B1)))
        B1d = (fx(*mg(gx.B1d, gy.B0d)),
               fy(*mg(gx.B0d, gy.B1d)))
        return B0, B1, B2, B0d, B1d, B2d
    B0, B1, B2, B0d, B1d, B2d = basis_fn()
    
    R0, R1, R2, R0d, R1d, R2d = reconstruction(basis_fn())
    
    def derivative():

        def deriv(g, axis):
            d, dd = g.derivative()
            D  = lambda arr: apply_along_axis(d, axis, arr)
            DD = lambda arr: apply_along_axis(dd, axis, arr)
            return D, DD

        Dx, Ddx = deriv(gx, axis=1)
        Dy, Ddy = deriv(gy, axis=0)
        
        D0 = lambda f: (Dx(f), Dy(f))
        D0d = lambda f: (Ddx(f), Ddy(f))
        D1 = lambda f: -Dy(f[0]) + Dx(f[1]) 
        D1d = lambda f: -Ddy(f[0]) + Ddx(f[1])

        return D0, D1, D0d, D1d
    D0, D1, D0d, D1d = derivative()
    
    def boundary_condition():

        def BC0(f):
            ((x0, y0), (x1,y1)) = edges_dual[0]
            bc0 = zeros(x0.shape)
            ma = (x0==gx.xmin)
            bc0[ma] -= f(x0[ma], y0[ma])
            ma = (x1==gx.xmax)
            bc0[ma] += f(x1[ma], y1[ma])
    
            ((x0, y0), (x1,y1)) = edges_dual[1]        
            bc1 = zeros(x1.shape)
            ma = (y0==gy.xmin)
            bc1[ma] -= f(x0[ma], y0[ma])
            ma = (y1==gy.xmax)
            bc1[ma] += f(x1[ma], y1[ma])
            return bc0, bc1

        def BC1(f):
            ((x0, y0), (x1,y1), (x2, y2), (x3, y3)) = faces_dual
            bc = zeros(x0.shape)
            ma = (y0==gy.xmin)
            bc[ma] += integrate_1form( ((x0[ma], y0[ma]), (x1[ma], y1[ma])), f)[0]
            ma = (x1==gx.xmax)
            bc[ma] += integrate_1form( ((x1[ma], y1[ma]), (x2[ma], y2[ma])), f)[0]
            ma = (y2==gy.xmax)
            bc[ma] += integrate_1form( ((x2[ma], y2[ma]), (x3[ma], y3[ma])), f)[0]
            ma = (x3==gx.xmin)
            bc[ma] += integrate_1form( ((x3[ma], y3[ma]), (x0[ma], y0[ma])), f)[0]
            return bc

        return BC0, BC1
    BC0, BC1 = boundary_condition()
    
    def hodge_star():

        def hodge(g, axis):
            h0, h1, h0d, h1d = g.hodge_star()
            H0 = lambda arr: apply_along_axis(h0, axis, arr)
            H1 = lambda arr: apply_along_axis(h1, axis, arr)
            H0d = lambda arr: apply_along_axis(h0d, axis, arr)
            H1d = lambda arr: apply_along_axis(h1d, axis, arr)
            return H0, H1, H0d, H1d
    
        H0x, H1x, H0dx, H1dx = hodge(gx, axis=1)
        H0y, H1y, H0dy, H1dy = hodge(gy, axis=0)
        
        H0 = lambda f: H0x(H0y(f))
        H2 = lambda f: H1x(H1y(f))
        H0d = lambda f: H0dx(H0dy(f))
        H2d = lambda f: H1dx(H1dy(f))
        
        H1 = lambda (fx, fy): (-H0x(H1y(fy)), H0y(H1x(fx)))
        H1d = lambda (fx, fy): (-H0dx(H1dy(fy)), H0dy(H1dx(fx)))
        
        return H0, H1, H2, H0d, H1d, H2d
    H0, H1, H2, H0d, H1d, H2d = hodge_star()
    
    return bunch(**locals())

def Grid_2D_Periodic(N, M):
    return Grid_2D_Cartesian(Grid_1D_Periodic(N), Grid_1D_Periodic(M))

def Grid_2D_Chebyshev(N, M):
    return Grid_2D_Cartesian(Grid_1D_Chebyshev(N), Grid_1D_Chebyshev(M))

def Grid_2D_Regular(N, M):
    return Grid_2D_Cartesian(Grid_1D_Regular(N), Grid_1D_Regular(M))

def laplacian2(g):
    """ 
    2D Laplacian Operator
    """
    D0, D1, D0d, D1d = g.derivative()
    H0, H1, H2, H0d, H1d, H2d = g.hodge_star()
    add = lambda x, y: (x[0]+y[0], x[1]+y[1])
    
    L0 = lambda f: H2d(D1d(H1(D0(f))))
    L0d = lambda f: H2(D1(H1d(D0d(f))))
    L1 = lambda f: add(H1d(D0d(H2(D1(f)))), 
                       D0(H2d(D1d(H1(f)))))
    L1d = lambda f: add(H1(D0(H2d(D1d(f)))), 
                        D0d(H2(D1(H1d(f)))))
    
    return L0, L1, L0d, L1d

def _draw(plt, pnts, xytext=(10,10), color='k', fc='blue'):
    
    def average(pnts):
        Sx, Sy = map(lambda x: reduce(operator.add, x), zip(*pnts))
        Lx, Ly = map(len, zip(*pnts))
        return Sx/Lx, Sy/Ly
    
    X, Y = average(pnts)
    plt.scatter(X, Y,color=color)
                    
    for i, (x, y) in enumerate(zip(X.flat, Y.flat)):
        plt.annotate(
            '{0}'.format(i), 
            xy = (x, y), xytext = xytext,
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = fc, alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

def _plot_scatter(g):
    import matplotlib.pyplot as plt

    xytext = (20,20)
    _draw(plt, [g.verts,], xytext=xytext, fc='black', color='k')
#    _draw(plt, g.faces, xytext=xytext, fc='green', color='r')
    _draw(plt, g.edges[0], xytext=xytext, fc='green', color='r')
#    _draw(plt, g.edges[1], xytext=xytext, fc='green', color='r')

    xytext = (-20,-20)
    _draw(plt, [g.verts_dual,], xytext=xytext, fc='red', color='r')
    #_draw(plt, g.faces_dual, xytext=xytext, fc='orange', color='r')
    _draw(plt, g.edges_dual[0], xytext=xytext, fc='orange', color='r')
#    _draw(plt, g.edges_dual[1], xytext=xytext, fc='orange', color='r')
    plt.show()
