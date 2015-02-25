'''
Spectral DEC in 2D
=============================
'''
from numpy import *
from dec.helper import *
from dec.grid1 import *
from dec.spectral import *

def cartesian_product(X, Y):
    '''
    >>> cartesian_product([0,1],[2,3])
    (array([0, 1, 0, 1]), array([2, 2, 3, 3]))
    '''
    X = asarray(X)
    Y = asarray(Y)
    X, Y = [x.flatten() for x in meshgrid(X, Y)]
    return X, Y

Grid_2D_Interface = '''
    verts verts_dual
    edges edges_dual
    faces faces_dual
    basis_fn
    projection
    reconstruction
    derivative
    hodge_star
    gx gy
'''.split()

class Grid_2D_Cartesian:
    
    def __init__(self, gx, gy):

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

        self.dimension = dimension
        self.gx = gx
        self.gy = gy
        self.verts = verts
        self.verts_dual = verts_dual
        self.edges = edges
        self.edges_dual = edges_dual
        self.faces = faces
        self.faces_dual = faces_dual

    def projection(self):
        P0  = lambda f: f(*self.verts)
        P0d = lambda f: f(*self.verts_dual)
        P2 = lambda f: integrate_2form(self.faces, f)[0]
        P2d = lambda f: integrate_2form(self.faces_dual, f)[0]
        P1  = lambda f: (integrate_1form(self.edges[0], f)[0],
                         integrate_1form(self.edges[1], f)[0])
        P1d = lambda f: (integrate_1form(self.edges_dual[0], f)[0],
                         integrate_1form(self.edges_dual[1], f)[0])
        return P0, P1, P2, P0d, P1d, P2d

    def basis_fn(self):
        vec = vectorize(lambda u, v: (lambda x, y: (u(x)*v(y))))
        mg = lambda x, y: meshgrid(x, y, copy=False, sparse=False)
        gxB0, gxB1, gxB0d, gxB1d = self.gx.basis_fn()
        gyB0, gyB1, gyB0d, gyB1d = self.gy.basis_fn()
        B0  = vec(*mg(gxB0,  gyB0))
        B0d = vec(*mg(gxB0d, gyB0d))
        B2  = vec(*mg(gxB1,  gyB1))
        B2d = vec(*mg(gxB1d, gyB1d))

        fx = vectorize(lambda u, v: (lambda x, y: (u(x)*v(y), 0)))
        fy = vectorize(lambda u, v: (lambda x, y: (0, u(x)*v(y))))
        B1  = (fx(*mg(gxB1, gyB0)),
               fy(*mg(gxB0, gyB1)))
        B1d = (fx(*mg(gxB1d, gyB0d)),
               fy(*mg(gxB0d, gyB1d)))
        return B0, B1, B2, B0d, B1d, B2d

    def reconstruction(self):
        R0, R1, R2, R0d, R1d, R2d = reconstruction(self.basis_fn())
        return R0, R1, R2, R0d, R1d, R2d

    def derivative(self):

        def deriv(g, axis):
            d, dd = g.derivative()
            D  = lambda arr: apply_along_axis(d, axis, arr)
            DD = lambda arr: apply_along_axis(dd, axis, arr)
            return D, DD

        Dx, Ddx = deriv(self.gx, axis=1)
        Dy, Ddy = deriv(self.gy, axis=0)

        D0 = lambda f: (Dx(f), Dy(f))
        D0d = lambda f: (Ddx(f), Ddy(f))
        D1 = lambda f: -Dy(f[0]) + Dx(f[1])
        D1d = lambda f: -Ddy(f[0]) + Ddx(f[1])

        return D0, D1, D0d, D1d

    def boundary_condition(self):
        '''
        Two types of boundaries: Vertices (0) or Edges (1). 
        '''

        def BC0(f):
            ((x0, y0), (x1,y1)) = self.edges_dual[0]
            bc0 = zeros(x0.shape)
            ma = (x0==self.gx.xmin)
            bc0[ma] -= f(x0[ma], y0[ma])
            ma = (x1==self.gx.xmax)
            bc0[ma] += f(x1[ma], y1[ma])

            ((x0, y0), (x1,y1)) = self.edges_dual[1]
            bc1 = zeros(x1.shape)
            ma = (y0==self.gy.xmin)
            bc1[ma] -= f(x0[ma], y0[ma])
            ma = (y1==self.gy.xmax)
            bc1[ma] += f(x1[ma], y1[ma])
            return bc0, bc1

        def BC1(f):
            ((x0, y0), (x1,y1), (x2, y2), (x3, y3)) = self.faces_dual
            bc = zeros(x0.shape)
            ma = (y0==self.gy.xmin)
            bc[ma] += integrate_1form( ((x0[ma], y0[ma]), (x1[ma], y1[ma])), f)[0]
            ma = (x1==self.gx.xmax)
            bc[ma] += integrate_1form( ((x1[ma], y1[ma]), (x2[ma], y2[ma])), f)[0]
            ma = (y2==self.gy.xmax)
            bc[ma] += integrate_1form( ((x2[ma], y2[ma]), (x3[ma], y3[ma])), f)[0]
            ma = (x3==self.gx.xmin)
            bc[ma] += integrate_1form( ((x3[ma], y3[ma]), (x0[ma], y0[ma])), f)[0]
            return bc

        return BC0, BC1

    def hodge_star(self):

        H0x, H1x, H0dx, H1dx = apply_operators(self.gx.hodge_star(), axis=1)
        H0y, H1y, H0dy, H1dy = apply_operators(self.gy.hodge_star(), axis=0)

        H0 = lambda f: H0x(H0y(f))
        H2 = lambda f: H1x(H1y(f))
        H0d = lambda f: H0dx(H0dy(f))
        H2d = lambda f: H1dx(H1dy(f))

        def H1(f):
            fx, fy = f
            return -H0x(H1y(fy)), H0y(H1x(fx))
        
        def H1d(f):
            fx, fy = f
            return -H0dx(H1dy(fy)), H0dy(H1dx(fx))

        return H0, H1, H2, H0d, H1d, H2d
    
    def contraction(self, V):
        H0x, H1x, H0dx, H1dx = apply_operators(self.gx.hodge_star(), axis=1)
        H0y, H1y, H0dy, H1dy = apply_operators(self.gy.hodge_star(), axis=0)
        Sx, Sinvx = apply_operators(self.gx.switch(), axis=1)
        Sy, Sinvy = apply_operators(self.gy.switch(), axis=0)

        Vx, Vy = V
        vx, vy = Sinvx(H1x(Vx)), Sinvy(H1y(Vy))

        def C1(f):
            alphax, alphay = f
            ax, ay = Sinvx(H1x(alphax)), Sinvy(H1y(alphay))
            c = ax*vx + ay*vy
            return real(c)
        
        def C2(f):
            o = H1x(H1y(f))
            #TODO: since we are multiplying, should we not refine the grid first?
            ax, ay = -Sinvy(o)*vy, Sinvx(o)*vx
            cx, cy = H0dx(Sx(ax)), H0dy(Sy(ay))
            return ( real(cx), real(cy) )
        
        return C1, C2

def apply_operators(H, axis):
    return [lambda arr, h=h: apply_along_axis(h, axis, arr) for h in H]

def Grid_2D_Periodic(N, M):
    return Grid_2D_Cartesian(Grid_1D_Periodic(N), Grid_1D_Periodic(M))

def Grid_2D_Chebyshev(N, M):
    return Grid_2D_Cartesian(Grid_1D_Chebyshev(N), Grid_1D_Chebyshev(M))

def Grid_2D_Regular(N, M):
    return Grid_2D_Cartesian(Grid_1D_Regular(N), Grid_1D_Regular(M))

def laplacian2(g):
    '''
    2D Laplacian Operator
    '''
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
        Sx, Sy = [reduce(operator.add, x) for x in zip(*pnts)]
        Lx, Ly = list(map(len, list(zip(*pnts))))
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
