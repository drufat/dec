'''
Spectral DEC in 2D
=============================
'''
from numpy import *
from dec.helper import *
from dec.grid1 import *
from operator import mul
from functools import reduce
import itertools
import operator
from scipy.interpolate.interpolate import interp2d

Π = lambda *x: tuple(itertools.product(*x))

def arange2(shape):
    '''
    >>> arange2((2,1))
    array([[0],
           [1]])
    >>> arange2((3,4))
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    >>> arange2(((2, 2), (3, 2)))
    (array([[0, 1],
           [2, 3]]), array([[4, 5],
           [6, 7],
           [8, 9]]))
    '''
    if type(shape[0]) is int:
        n = cumul_index((shape,))[-1]
    else:    
        n = cumul_index(shape)[-1]
    return reshape(arange(n, dtype=int), shape)

def unravel_idx(i, shape):
    '''
    >>> shape = ((11, 12), (13, 14))
    >>> A = arange2(shape)
    >>> k, (i, j) = unravel_idx(10, shape)
    >>> A[k][i,j]
    10
    >>> k, (i, j) = unravel_idx(0, shape)
    >>> A[k][i,j]
    0
    >>> k, (i, j) = unravel_idx(200, shape)
    >>> A[k][i,j]
    200
    '''
    cum = (0,) + cumul_index(shape)
    for (k, (a, b)) in enumerate(zip(cum[:-1], cum[1:])):
        if a <= i < b:
            return k, unravel_index(i-a, shape[k])
    raise ValueError

def cartesian_product(X, Y):
    '''
    >>> cartesian_product([0,1],[2,3])
    (array([0, 1, 0, 1]), array([2, 2, 3, 3]))
    '''
    X = asarray(X)
    Y = asarray(Y)
    X, Y = [x.flatten() for x in meshgrid(X, Y)]
    return X, Y

def cumul_index(shape):
    '''
    >>> cumul_index(((3,1), (4,5), (6,7)))
    (3, 23, 65)
    '''
    lngth = array([reduce(mul, s) for s in shape], dtype=int)
    return tuple(cumsum(lngth))

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
           [3, 4, 5]])
    >>> reshape(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), ((2, 3), (2, 2)))
    (array([[0, 1, 2],
           [3, 4, 5]]), array([[6, 7],
           [8, 9]]))
    '''
    if type(shape[0]) is tuple:
        arr = split(x, cumul_index(shape)[:-1])
        return tuple(a.reshape(s) for a, s in zip(arr, shape))
    elif type(shape[0]) is int:
        return x.reshape(shape)
    else:
        raise TypeError

def apply_operators(H, axis):
    def get_apply(h):
        return lambda x: apply_along_axis(h, axis, x)
    if type(H) is tuple or type(H) is list:
        return tuple(get_apply(h) for h in H)
    elif type(H) is dict:
        return {k:get_apply(H[k]) for k in H}
    else:
        raise TypeError
        
class Grid_2D(object):
    
    def __init__(self, gx, gy, N, cells, shape, dec, refine):
        self.gx, self.gy = gx, gy
        self.dimension = gx.dimension + gy.dimension
        self.N = N
        self.cells = cells
        self.shape = shape
        self.dec = dec
        self.refine = refine
        
    @classmethod
    def product(cls, gx, gy):
        g = cartesian_product_grids(gx, gy)
        return flatten_grid(g)
    
    @classmethod
    def periodic(cls, N, M):
        return cls.product(Grid_1D.periodic(N), Grid_1D.periodic(M))

    @classmethod
    def chebyshev(cls, N, M):
        return cls.product(Grid_1D.chebyshev(N), Grid_1D.chebyshev(M))
    
    @property
    def points(self):
        return meshgrid(self.gx.points, self.gy.points)
    
    @property
    def verts(self):
        return self.cells[0, True]

    @property
    def edges(self):
        return self.cells[1, True]
    
    @property
    def faces(self):
        return self.cells[2, True]

    @property
    def verts_dual(self):
        return self.cells[0, False]
    
    @property
    def edges_dual(self):
        return self.cells[1, False]
    
    @property
    def faces_dual(self):
        return self.cells[2, False]
    
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
        B = self.dec.B
        B0  = [B[0, True ](i) for i in range(self.N[0, True])]
        B1  = [B[1, True ](i) for i in range(self.N[1, True])]
        B2  = [B[2, True ](i) for i in range(self.N[2, True])]
        B0d = [B[0, False](i) for i in range(self.N[0, False])]
        B1d = [B[1, False](i) for i in range(self.N[1, False])]
        B2d = [B[2, False](i) for i in range(self.N[2, False])]
        return B0, B1, B2, B0d, B1d, B2d
    
    def reconstruction(self):
        R0, R1, R2, R0d, R1d, R2d = dec.spectral.reconstruction(self.basis_fn())
        return R0, R1, R2, R0d, R1d, R2d

    def boundary_condition(self):
        BC0, BC1 = boundary_condition(self.cells[1, False], 
                                      self.cells[2, False], 
                                      self.gx.xmin, self.gx.xmax, 
                                      self.gy.xmin, self.gy.xmax)
        BC0_ = lambda f: unshape(BC0(f))[0]
        BC1_ = lambda f: unshape(BC1(f))[0]
        return BC0_, BC1_
    
    def rand(self, deg, isprimal):
        '''
        Create a random form.
        '''
        return decform(deg, isprimal, self, np.random.rand(self.N[deg, isprimal]))

    def P(self, deg, isprimal, func):
        return decform(deg, isprimal, self, self.dec.P[deg, isprimal](func))

def projection(cells):
    
    P = {(0, True ) : lambda f: f(*cells[0, True ]),
         (0, False) : lambda f: f(*cells[0, False]),
         (1, True ) : lambda f: integrate_1form(cells[1, True ], f)[0],
         (1, False) : lambda f: integrate_1form(cells[1, False], f)[0],
         (2, True ) : lambda f: integrate_2form(cells[2, True ], f)[0],
         (2, False) : lambda f: integrate_2form(cells[2, False], f)[0]}

    return P

# def interpolate(T, px, py):
#     I = {(0, True )  : lambda f: interp2d(px, py, T[0, True](f)),
#          (0, False ) : lambda f: interp2d(px, py, T[0, False](f)),}
#     return I

def boundary_condition(edges_dual, faces_dual, xmin, xmax, ymin, ymax):
    '''
    Two types of boundaries: Vertices (0) or Edges (1). 
    '''

    def BC0(f):
        ((x0, y0), (x1,y1)) = edges_dual
        bc = zeros(x0.shape)
        ma = (x0==xmin)
        bc[ma] -= f(x0[ma], y0[ma])
        ma = (x1==xmax)
        bc[ma] += f(x1[ma], y1[ma])
        ma = (y0==ymin)
        bc[ma] -= f(x0[ma], y0[ma])
        ma = (y1==ymax)
        bc[ma] += f(x1[ma], y1[ma])
        return bc

    def BC1(f):
        ((x0, y0), (x1,y1), (x2, y2), (x3, y3)) = faces_dual
        bc = zeros(x0.shape)
        ma = (y0==ymin)
        bc[ma] += integrate_1form( ((x0[ma], y0[ma]), (x1[ma], y1[ma])), f)[0]
        ma = (x1==xmax)
        bc[ma] += integrate_1form( ((x1[ma], y1[ma]), (x2[ma], y2[ma])), f)[0]
        ma = (y2==ymax)
        bc[ma] += integrate_1form( ((x2[ma], y2[ma]), (x3[ma], y3[ma])), f)[0]
        ma = (x3==xmin)
        bc[ma] += integrate_1form( ((x3[ma], y3[ma]), (x0[ma], y0[ma])), f)[0]
        return bc

    return BC0, BC1

def basis_fn(Bx, By):

    B = {}
    def update_B(t):
        B[0, t] = lambda j, i: lambda x, y: Bx[0, t](i)(x)*By[0, t](j)(y)
        B[1, t] =(lambda j, i: lambda x, y: array((Bx[1, t](i)(x)*By[0, t](j)(y), 0), dtype=object),
                  lambda j, i: lambda x, y: array((0, Bx[0, t](i)(x)*By[1, t](j)(y)), dtype=object))
        B[2, t] = lambda j, i: lambda x, y: Bx[1, t](i)(x)*By[1, t](j)(y)
    update_B(True)
    update_B(False)
    return B

def derivative(dx, dy):

    def deriv(D, axis):
        d, dd = D[0, True], D[0, False]
        D  = lambda arr: apply_along_axis(d, axis, arr)
        DD = lambda arr: apply_along_axis(dd, axis, arr)
        return D, DD

    Dx, Ddx = deriv(dx, axis=1)
    Dy, Ddy = deriv(dy, axis=0)

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

def hodge_star(hx, hy):

    def expand(h):
        return (h[0, True], h[1, True], h[0, False], h[1, False])

    H0x, H1x, H0dx, H1dx = apply_operators(expand(hx), axis=1)
    H0y, H1y, H0dy, H1dy = apply_operators(expand(hy), axis=0)

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

def to_refine(tx, ty):
    
    Tx = apply_operators(tx, axis=1)
    Ty = apply_operators(ty, axis=0)
    
    T = {(0, True ): lambda f:(Ty[0, True ](Tx[0, True ](f)),),
         (1, True ): lambda f:(Ty[0, True ](Tx[1, True ](f[0])),
                               Ty[1, True ](Tx[0, True ](f[1])),),
         (2, True ): lambda f:(Ty[1, True ](Tx[1, True ](f)),),
         (0, False): lambda f:(Ty[0, False](Tx[0, False](f)),),
         (1, False): lambda f:(Ty[0, False](Tx[1, False](f[0])),
                               Ty[1, False](Tx[0, False](f[1])),),
         (2, False): lambda f:(Ty[1, False](Tx[1, False](f)),),}
    
    return T

def from_refine(ux, uy):
    
    Ux = apply_operators(ux, axis=1)
    Uy = apply_operators(uy, axis=0)
    
    U = {(0, True ): lambda f: Uy[0, True ](Ux[0, True ](f[0])),
         (1, True ): lambda f:(Uy[0, True ](Ux[1, True ](f[0])),
                               Uy[1, True ](Ux[0, True ](f[1]))),
         (2, True ): lambda f: Uy[1, True ](Ux[1, True ](f[0])),
         (0, False): lambda f: Uy[0, False](Ux[0, False](f[0])),
         (1, False): lambda f:(Uy[0, False](Ux[1, False](f[0])),
                               Uy[1, False](Ux[0, False](f[1]))),
         (2, False): lambda f: Uy[1, False](Ux[1, False](f[0])),}
    
    return U

def product_cells(sx, sy):
    '''
    Representation of a cellular complex in 2D (block array):

        vertices: (x, y)
        edges:    (((x0, x1), y), (x, (y0, y1)))
        faces:    ((x0, y0), (x1, y1))
    '''

    def vertices(vx, vy):
        x, y = meshgrid(vx, vy)
        return (x, y)
    
    def edges(vx, vy, ex, ey):
        x0, y = meshgrid(ex[0], vy)
        x1, y = meshgrid(ex[1], vy)
        x, y0 = meshgrid(vx, ey[0])
        x, y1 = meshgrid(vx, ey[1])
        return (((x0, x1), y), (x, (y0, y1)))
        
    def faces(ex, ey):
        x0, y0 = meshgrid(ex[0], ey[0])
        x1, y1 = meshgrid(ex[1], ey[1])
        return ((x0, y0), (x1, y1))
     
    t, f = True, False
    cells = {(0, t) : vertices(sx[0, t], sy[0, t]),
             (0, f) : vertices(sx[0, f], sy[0, f]),
             (1, t) : edges(sx[0, t], sy[0, t], sx[1, t], sy[1, t]),
             (1, f) : edges(sx[0, f], sy[0, f], sx[1, f], sy[1, f]),
             (2, t) : faces(sx[1, t], sy[1, t]),
             (2, f) : faces(sx[1, f], sy[1, f])}
    return cells

def flatten_cells(cells):
    '''
    Representation of a cellular complex in 2D (flat array):

        vertices: (x, y)
        edges:    ((x0, y0), (x1, y1))
        faces:    ((x0, y0), (x1, y1), (x2, y2), (x3, y3))

    '''
    cells_new = {}
    shape = {}
    
    def get_f(k):
        def f(x):
            x, shape[k] = unshape(x)
            return x
        return f

    for t in (True, False):

        (x, y) = cells[0, t]
        (x, y) = map(get_f((0, t)), (x, y))
        cells_new[0, t] = (x, y)
        
        (((x0, x1), y), (x, (y0, y1))) = cells[1, t]
        (x0, y0, x1, y1) = map(get_f((1,t)), ((x0, x), (y, y0), (x1, x), (y, y1)))
        cells_new[1, t] = ((x0, y0), (x1, y1))

        ((x0, y0), (x1, y1)) = cells[2, t]
        (x0, y0, x1, y1) = map(get_f((2, t)), (x0, y0, x1, y1))
        cells_new[2, t] = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    
    return cells_new, shape

def countshape(shape):
    '''
    >>> shape = {(0, False): (4, 4),
    ...          (0, True): (5, 5),
    ...          (1, False): ((4, 5), (5, 4)),
    ...          (1, True): ((5, 4), (4, 5)),
    ...          (2, False): (5, 5),
    ...          (2, True): (4, 4)}
    >>> assert countshape(shape) == {
    ...         (0, False): 16,
    ...         (0, True): 25,
    ...         (1, False): 40,
    ...         (1, True): 40,
    ...         (2, False): 25,
    ...         (2, True): 16}
    '''
    N = {}
    for deg, isprimal in shape:
        if deg == 1:
            (hx, hy), (vx, vy) = shape[deg, isprimal]
            N[deg, isprimal] = hx*hy + vx*vy            
        else:
            nx, ny = shape[deg, isprimal]
            N[deg, isprimal] = nx*ny
    return N

def cartesian_product_grids(gx, gy):

    assert gx.dimension is 1
    assert gy.dimension is 1
    
    cells = product_cells(gx.cells, gy.cells)    
    P = projection(cells)

    B = basis_fn(gx.dec.B, gy.dec.B)
    D = derivative(gx.dec.D, gy.dec.D)
    H = hodge_star(gx.dec.H, gy.dec.H)
    dec = bunch(P=P, B=B, D=D, H=H,)
    
    T = to_refine(  gx.refine.T, gy.refine.T)
    U = from_refine(gx.refine.U, gy.refine.U)
            
    refine = bunch(T=T, U=U)
    
    return Grid_2D(gx, gy, None, cells, None, dec, refine)


def reshapeB(shape, B):

    def getB(b, s):
        if type(b) is tuple:
            # 1form
            def Bi(i):
                k, ij = unravel_idx(i, s)
                return b[k](*ij)
        else:
            # 0form or 2form
            def Bi(i):
                ij = unravel_index(i, s)
                return b(*ij)
        return Bi
    return { k : getB(B[k], shape[k]) for k in B}

def reshapeP(shape, P):
    def get_new_p(p, k):
        def new_p(f):
            f = p(f)
            f, _ = unshape(f)
            return f
        return new_p
    newP = {k : get_new_p(P[k], k) for k in P}
    return newP

def reshapeT(shape, T):
    def get_new(p, k):
        def new_f(f):
            f = reshape(f, shape[k])
            f = p(f)
            return f
        return new_f
    return {k:get_new(T[k], k) for k in T}

def reshapeU(shape, U):
    def get_new(p, k):
        def new_f(f):
            f = p(f)
            f, _ = unshape(f)
            return f
        return new_f
    return {k:get_new(U[k], k) for k in U}

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

def flatten_grid(g):
    
    cells, shape = flatten_cells(g.cells)
    N = countshape(shape)

    refine = bunch(T=reshapeT(shape, g.refine.T),
                   U=reshapeU(shape, g.refine.U))

    dec = bunch(P=projection(cells),
                B=reshapeB(shape, g.dec.B),
                D=reshapeO(shape, g.dec.D, (lambda d, p: (d+1, p))),
                H=reshapeO(shape, g.dec.H, (lambda d, p: (2-d, not p))),
                W=wedge(refine.T, refine.U),
                C=contraction(refine.T, refine.U),)
    
    return Grid_2D(g.gx, g.gy, N, cells, shape, dec, refine)

import dec.symbolic

def wedge(T, U):
    
    Ws = dec.symbolic.wedge_2d()
    W = {}    
    
    def get_w(d0, p0, d1, p1, p2):
        def w(a, b):
            a = T[d0, p0](a)
            b = T[d1, p1](b)
            c = Ws[d0, d1](a, b)
            return U[d0+d1, p2](c)
        return w

    for ((d0, p0), (d1, p1), p2) in Π(Π((0, 1, 2), (True, False)), 
                                      Π((0, 1, 2), (True, False)),
                                      (True, False)):
        if d0 + d1 > 2: continue
        if p0==p1==p2 and d0==d1==0:
            #no refinement necessary, just multiply directly
            W[(d0, p0), (d1, p1), p2] = lambda a, b: a*b
        else:
            W[(d0, p0), (d1, p1), p2] = get_w(d0, p0, d1, p1, p2)

    return W

def contraction(T, U):

    Cs = dec.symbolic.contraction_2d()
    C = {}
    
    def get_c(p0, d1, p1, p2):
        def c(a, b):
            a = T[1, p0](a)
            b = T[d1, p1](b)
            c = Cs[d1](a, b)
            return U[d1-1, p2](c)
        return c

    for (p0, (d1, p1), p2) in Π((True, False), 
                                Π((0, 1, 2), (True, False)), 
                                (True, False)):
        if d1-1 < 0: continue
        C[p0, (d1, p1), p2] = get_c(p0, d1, p1, p2)    

    return C

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
    #_draw(plt, g.faces, xytext=xytext, fc='green', color='r')
    #_draw(plt, g.edges, xytext=xytext, fc='green', color='r')

    xytext = (-20,-20)
    _draw(plt, [g.verts_dual,], xytext=xytext, fc='red', color='r')
    #_draw(plt, g.faces_dual, xytext=xytext, fc='orange', color='r')
    #_draw(plt, g.edges_dual, xytext=xytext, fc='orange', color='r')
    plt.show()
