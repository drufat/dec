'''
Spectral DEC in 2D
=============================
'''
from numpy import *
from dec.helper import *
from dec.grid1 import *
from dec.spectral import *
from operator import mul

Π = lambda *x: tuple(itertools.product(*x))

def arange2(shape):
    '''
    >>> arange2((2,1))
    array([[0],
    ...    [1]])
    >>> arange2((3,4))
    array([[ 0,  1,  2,  3],
    ...    [ 4,  5,  6,  7],
    ...    [ 8,  9, 10, 11]])
    '''
    n = 1
    for s in shape: n*=s
    return arange(n, dtype=int).reshape(shape)

def cartesian_product(X, Y):
    '''
    >>> cartesian_product([0,1],[2,3])
    (array([0, 1, 0, 1]), array([2, 2, 3, 3]))
    '''
    X = asarray(X)
    Y = asarray(Y)
    X, Y = [x.flatten() for x in meshgrid(X, Y)]
    return X, Y

def apply_operators(H, axis):
    def get_apply(h):
        return lambda x: apply_along_axis(h, axis, x)
    return [get_apply(h) for h in H]

def unshape(X):
    '''
    >>> unshape(array([[0, 1, 2], 
    ...                [3, 4, 5]]))
    (array([0, 1, 2, 3, 4, 5]), (2, 3))
    >>> unshape((array([[0, 1, 2], 
    ...                 [3, 4, 5]]), 
    ...          array([[6, 7], 
    ...                 [8, 9]])))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), ((2, 3), (2, 2)))
    '''
    if type(X) is tuple:
        shape = []
        arr = []
        for x in X:
            shape.append(x.shape)
            arr.append(x.reshape(-1))
        return concatenate(arr), tuple(shape)
    elif type(X) is ndarray:
        return X.reshape(-1), X.shape
    else:
        raise TypeError

def reshape(x, shape):
    '''
    >>> reshape(array([0, 1, 2, 3, 4, 5]), (2, 3))
    array([[0, 1, 2],
    ...    [3, 4, 5]])
    >>> reshape(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), ((2, 3), (2, 2)))
    (array([[0, 1, 2],
    ...     [3, 4, 5]]), array([[6, 7],
    ...     [8, 9]]))
    '''
    if type(shape[0]) is tuple:
        lngth = array([reduce(mul, s) for s in shape], dtype=int)
        arr = split(x, cumsum(lngth)[:-1])
        return tuple(a.reshape(s) for a, s in zip(arr, shape))
    elif type(shape[0]) is int:
        return x.reshape(shape)
    else:
        raise TypeError

def reshapeP(shape, P):
    def get_new_p(p, k):
        def new_p(f):
            f = p(f)
            f, _ = unshape(f)
            return f
        return new_p
    newP = {k : get_new_p(P[k], k) for k in P}
    return newP

def reshapeO(shape, O, Otype):
    def get_new_op(op, typ):
        deg, isprimal = typ
        def new_op(f):
            f = reshape(f, shape[deg  , isprimal])
            f = op(f)
            f, _ = unshape(f)
            return f
        return new_op
    newO = {}
    for k in O:
        if Otype(*k) in shape:
            newO[k] = get_new_op(O[k], k)
        else: 
            newO[k] = O[k]
    return newO

class Grid_2D(object):
    
    def __init__(self, gx, gy, N, simp, shape, dec, refine):
        self.gx, self.gy = gx, gy
        self.dimension = gx.dimension + gy.dimension
        self.N = N
        self.simp = simp
        self.shape = shape
        self.dec = dec
        self.refine = refine
        
    @classmethod
    def product(cls, gx, gy):
        return cartesian_product_grids(gx, gy)
    
    @classmethod
    def periodic(cls, N, M):
        return cls.product(Grid_1D.periodic(N), Grid_1D.periodic(M))

    @classmethod
    def chebyshev(cls, N, M):
        return cls.product(Grid_1D.chebyshev(N), Grid_1D.chebyshev(M))
        
    @property
    def verts(self):
        return self.simp[0, True]

    @property
    def edges(self):
        return self.simp[1, True]
    
    @property
    def faces(self):
        return self.simp[2, True]

    @property
    def verts_dual(self):
        return self.simp[0, False]
    
    @property
    def edges_dual(self):
        return self.simp[1, False]
    
    @property
    def faces_dual(self):
        return self.simp[2, False]
    
    def projection(self):
        P = self.dec.P
        P0  = P[0, True]
        P1  = P[1, True]
        P2  = P[2, True]
        P0d = P[0, False]
        P1d = P[1, False]
        P2d = P[2, False]
        return P0, P1, P2, P0d, P1d, P2d
    
    def derivative(self):
        D = self.dec.D
        D0  = D[0, True]
        D1  = D[1, True]
        D0d = D[0, False]
        D1d = D[1, False]
        return D0, D1, D0d, D1d
    
    def hodge_star(self):
        H = self.dec.H
        H0  = H[0, True]
        H1  = H[1, True]
        H2  = H[2, True]
        H0d = H[0, False]
        H1d = H[1, False]
        H2d = H[2, False]
        return H0, H1, H2, H0d, H1d, H2d
    
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

    def boundary_condition(self):
        BC0, BC1 = boundary_condition(self.simp[1, False], 
                                      self.simp[2, False], 
                                      self.gx, self.gy)
        BC0_ = lambda f: unshape(BC0(f))[0]
        BC1_ = lambda f: unshape(BC1(f))[0]
        return BC0_, BC1_

def projection(simp):
    
    P = {(0, True ) : lambda f: f(*simp[0, True ]),
         (0, False) : lambda f: f(*simp[0, False]),
         (1, True ) : lambda f: integrate_1form(simp[1, True ], f)[0],
         (1, False) : lambda f: integrate_1form(simp[1, False], f)[0],
         (2, True ) : lambda f: integrate_2form(simp[2, True ], f)[0],
         (2, False) : lambda f: integrate_2form(simp[2, False], f)[0]}

    return P

def derivative(gx, gy):

    def deriv(g, axis):
        d, dd = g.derivative()
        D  = lambda arr: apply_along_axis(d, axis, arr)
        DD = lambda arr: apply_along_axis(dd, axis, arr)
        return D, DD

    Dx, Ddx = deriv(gx, axis=1)
    Dy, Ddy = deriv(gy, axis=0)

    D0  = lambda f: (Dx(f), Dy(f))
    D0d = lambda f: (Ddx(f), Ddy(f))
    D1  = lambda f: -Dy(f[0]) + Dx(f[1])
    D1d = lambda f: -Ddy(f[0]) + Ddx(f[1])

    D = {(0, True) : D0,
         (1, True) : D1,
         (2, True) : lambda f: 0,
         (0, False): D0d, 
         (1, False): D1d,
         (2, False): lambda f: 0}

    return D

def boundary_condition(edges_dual, faces_dual, gx, gy):
    '''
    Two types of boundaries: Vertices (0) or Edges (1). 
    '''

    def BC0(f):
        ((x0, y0), (x1,y1)) = edges_dual
        bc = zeros(x0.shape)
        ma = (x0==gx.xmin)
        bc[ma] -= f(x0[ma], y0[ma])
        ma = (x1==gx.xmax)
        bc[ma] += f(x1[ma], y1[ma])
        ma = (y0==gy.xmin)
        bc[ma] -= f(x0[ma], y0[ma])
        ma = (y1==gy.xmax)
        bc[ma] += f(x1[ma], y1[ma])
        return bc

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

def hodge_star(gx, gy):

    H0x, H1x, H0dx, H1dx = apply_operators(gx.hodge_star(), axis=1)
    H0y, H1y, H0dy, H1dy = apply_operators(gy.hodge_star(), axis=0)

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

    H = {(0, True) : H0,
         (1, True) : H1,
         (2, True) : H2,
         (0, False): H0d, 
         (1, False): H1d,
         (2, False): H2d}
    return H

def product_simplices(sx, sy):
    '''
    Representation of a cellular complex in 2D (block array):

        vertices: (x, y)
        edges:    (((x0h, y0h), (x1h, y1h)), ((x0v, y0v), (x1v, y1v)))
        faces:    ((x0, y0), (x1, y1), (x2, y2), (x3, y3))

    '''

    def vertices(vx, vy):
        x, y = meshgrid(vx, vy)
        return (x, y)
    
    def edges(vx, vy, ex, ey):
        x0h, y0h = meshgrid(ex[0], vy)
        x1h, y1h = meshgrid(ex[1], vy)
        x0v, y0v = meshgrid(vx, ey[0])
        x1v, y1v = meshgrid(vx, ey[1])        
        return (((x0h, y0h), (x1h, y1h)), 
                ((x0v, y0v), (x1v, y1v)))
    
    def faces(ex, ey):
        ((x0, y0), 
         (x1, y1), 
         (x2, y2), 
         (x3, y3)) = (meshgrid(ex[0], ey[0]),
                      meshgrid(ex[1], ey[0]),
                      meshgrid(ex[1], ey[1]),
                      meshgrid(ex[0], ey[1]))
        return  ((x0, y0), (x1, y1), (x2, y2), (x3, y3))
     
    t, f = True, False
    simp = {(0, t) : vertices(sx[0, t], sy[0, t]),
            (0, f) : vertices(sx[0, f], sy[0, f]),
            (1, t) : edges(sx[0, t], sy[0, t], sx[1, t], sy[1, t]),
            (1, f) : edges(sx[0, f], sy[0, f], sx[1, f], sy[1, f]),
            (2, t) : faces(sx[1, t], sy[1, t]),
            (2, f) : faces(sx[1, f], sy[1, f])}
    return simp

def product_simplices_flat(sx, sy):
    '''
    Representation of a cellular complex in 2D (flat array):

        vertices: (x, y)
        edges:    ((x0, y0), (x1, y1))
        faces:    ((x0, y0), (x1, y1), (x2, y2), (x3, y3))

    '''
    
    simp = product_simplices(sx, sy)
    simp_new = {}
    shape = {}
    
    def get_f(k):
        def f(x):
            x, shape[k] = unshape(x)
            return x
        return f

    for t in (True, False):

        (x, y) = simp[0, t]
        (x, y) = map(get_f((0, t)), 
        (x, y))
        simp_new[0, t] = (x, y)
        
        (((x0h, y0h), (x1h, y1h)), ((x0v, y0v), (x1v, y1v))) = simp[1, t]
        (x0, y0, x1, y1) = map(get_f((1,t)), ((x0h, x0v), (y0h, y0v), (x1h, x1v), (y1h, y1v)))
        simp_new[1, t] = ((x0, y0), (x1, y1))

        ((x0, y0), (x1, y1), (x2, y2), (x3, y3)) = simp[2, t]
        (x0, y0, x1, y1, x2, y2, x3, y3) = map(get_f((2, t)), (x0, y0, x1, y1, x2, y2, x3, y3))
        simp_new[2, t] = ((x0, y0), (x1, y1), (x2, y2), (x3, y3))
    
    return simp_new, shape

def cartesian_product_grids(gx, gy):

    assert gx.dimension is 1
    assert gy.dimension is 1
    
    simp, shape = product_simplices_flat(gx.simp, gy.simp)
    
#     shape = {(0, True) :  (gy.N[0, True], gx.N[0, True]),
#              (1, True) : ((gy.N[0, True], gx.N[1, True]), 
#                           (gy.N[1, True], gx.N[0, True])),
#              (2, True) :  (gy.N[1, True], gx.N[1, True]),
#              (0, False) :  (gy.N[0, False], gx.N[0, False]),
#              (1, False) : ((gy.N[0, False], gx.N[1, False]), 
#                            (gy.N[1, False], gx.N[0, False])),
#              (2, False) :  (gy.N[1, False], gx.N[1, False])}

    N = {}
    for deg, isprimal in shape:
        if deg == 1:
            (hx, hy), (vx, vy) = shape[deg, isprimal]
            N[deg, isprimal] = hx*hy + vx*vy            
        else:
            nx, ny = shape[deg, isprimal]
            N[deg, isprimal] = nx*ny
            
    decnf = bunch(D=derivative(gx, gy),
                  H=hodge_star(gx, gy),)

    dec = bunch(P=projection(simp),
                B=None,
                D=reshapeO(shape, decnf.D, (lambda d, p: (d+1, p))),
                H=reshapeO(shape, decnf.H, (lambda d, p: (2-d, not p))),
                W=None,
                C=None,)
    
    refine = None

    return Grid_2D(gx, gy, N, simp, shape, dec, refine)

def laplacian2(g):
    '''
    2D Laplacian Operator
    '''
    D0, D1, D0d, D1d = g.derivative()
    H0, H1, H2, H0d, H1d, H2d = g.hodge_star()
    
    L0 = lambda f: H2d(D1d(H1(D0(f))))
    L0d = lambda f: H2(D1(H1d(D0d(f))))
    L1 = lambda f: (H1d(D0d(H2(D1(f)))) + 
                    D0(H2d(D1d(H1(f)))))
    L1d = lambda f: (H1(D0(H2d(D1d(f)))) +
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
