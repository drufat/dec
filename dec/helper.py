import numpy as np
from collections import namedtuple
from scipy.sparse import csr_matrix
import scipy.integrate

def approx(a, b):
    c = np.round_(a, 10) == np.round_(b, 10)
    if type(c)==np.ndarray:
        return c.all()
    return c

def to_matrix(f, N):
    '''
    >>> np.array_equal(
    ... to_matrix(lambda x: x, 2),
    ... [[ 1.,  0.],
    ...  [ 0.,  1.]])
    True
    >>> np.array_equal(
    ... to_matrix(lambda x: np.roll(x,+1), 2),
    ... [[ 0.,  1.],
    ...  [ 1.,  0.]])
    True
    '''
    M = np.vstack(f(b) for b in np.eye(N)).T
    if approx(np.real(M), M).all():
        M = np.real(M)
    return M

def to_matrix_by_index(f, N):
    '''
    >>> np.array_equal(
    ... to_matrix_by_index(lambda i, j: 10*i + j, 2),
    ... [[  0.,   1.],
    ...  [ 10.,  11.]])
    True
    '''
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M[i, j] = f(i, j)
    return M

def sparse_diag(x):
    '''
    >>> np.array_equal(
    ... sparse_diag([1,2,3]).todense(),
    ... np.matrix([[1, 0, 0],
    ...            [0, 2, 0],
    ...            [0, 0, 3]]))
    True
    '''
    x = np.asanyarray(x)
    N = len(x)
    ii = np.row_stack([np.arange(N)]*2)
    return csr_matrix((x,ii), shape=(N, N))

def sparse_matrix(ijk, N, M):
    '''
    >>> np.array_equal(
    ... sparse_matrix([[0,1,1], [1,0,2]], 2, 2).todense(),
    ... np.matrix([[0, 1],
    ...            [2, 0]]))
    True
    '''
    ijk = np.asanyarray(ijk)
    ij = ijk[:,:2].T
    k = ijk[:,2].T
    return csr_matrix((k,ij), shape=(N,M))

class bunch(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def nametuple(name, fields, dictionary):
    '''
    >>> nametuple('Point', 'x,y', dict(x=1,y=2,z=3))
    Point(x=1, y=2)
    '''
    d = {x:dictionary[x] for x in fields.split(',')}
    return namedtuple(name, fields)(**d)

def NamedTuple(attribs):
    '''
    >>> def Point():
    ...    This = NamedTuple("x,y")
    ...    x = 1
    ...    y = 2
    ...    return This()
    >>> Point()
    Point(x=1, y=2)
    '''
    import inspect
    outerframe = inspect.currentframe().f_back
    name = outerframe.f_code.co_name
    NamedTuple = namedtuple(name, attribs)
    def This():
        local = inspect.currentframe().f_back.f_locals
        def fields():
            for field in NamedTuple._fields:
                yield local.get(field)
        return NamedTuple._make(fields())
    return This

def set_attributes(obj, namedtuple):
    for f in namedtuple.__class__._fields:
        setattr(obj, f, getattr(namedtuple, f))

def set_dictionary(obj, namedtuple):
    for f in namedtuple.__class__._fields:
        obj[f] = getattr(namedtuple, f)

def this_function_name():
    '''
    >>> def f():
    ...    return this_function_name()
    >>> f()
    'f'
    >>> a = f
    >>> a()
    'f'
    '''
    import inspect
    outerframe = inspect.currentframe().f_back
    name = outerframe.f_code.co_name
    return name

def make_namedtuple(metaclass, dictionary):
    '''
    Create a named tuple from a dictionary which is a superset of the fields

    >>> Point = namedtuple('Point', 'x,y')
    >>> make_namedtuple(Point, {'x':1, 'y':2, 'z':3})
    Point(x=1, y=2)
    '''
    return metaclass._make((dictionary[f] for f in metaclass._fields))

def apply_map(faces, identify):
    '''
    >>> apply_map([0,1,2], {0:3, 1:4, 2:5})
    array([3, 4, 5])
    '''
    faces = np.asanyarray(faces)
    data = faces.ravel()
    mp = np.arange(0,max(data)+1)
    mp[list(identify.keys())] = list(identify.values())
    data = mp[data]
    return data.reshape(faces.shape)

def bndry_with_zeros(a, axis=-1):
    '''
    Tab `a` with zeros at both ends along `axis`.
    >>> x = [1, 2, 4, 7, 0]
    >>> bndry_with_zeros(x)
    array([0, 1, 2, 4, 7, 0, 0])
    >>> x = [[1, 3, 6, 10], [0, 5, 6, 8]]
    >>> bndry_with_zeros(x, axis=0)
    array([[ 0,  0,  0,  0],
           [ 1,  3,  6, 10],
           [ 0,  5,  6,  8],
           [ 0,  0,  0,  0]])
    >>> bndry_with_zeros(x, axis=-1)
    array([[ 0,  1,  3,  6, 10,  0],
           [ 0,  0,  5,  6,  8,  0]])
    '''
    a = np.asanyarray(a)
    f1d = lambda x: np.concatenate(([0], x, [0]))
    return np.apply_along_axis(f1d, axis, a)

def interweave(a, b, axis=-1):
    ''' Interweave two arrays.
    >>> interweave([0, 1, 2], [3, 4, 5])
    array([0, 3, 1, 4, 2, 5])
    >>> interweave([[0,1],[2,3]],[[4,5],[6,7]])
    array([[0, 4, 1, 5],
           [2, 6, 3, 7]])
    >>> interweave([[0,1],[2,3]],[[4,5],[6,7]], axis=0)
    array([[0, 1],
           [4, 5],
           [2, 3],
           [6, 7]])
    '''
    a = np.asarray(a)
    b = np.asarray(b)
    assert(a.shape == b.shape)

    a = np.rollaxis(a, axis, len(a.shape))
    b = np.rollaxis(b, axis, len(b.shape))

    shape = np.array(a.shape)
    shape[-1] = a.shape[-1] + b.shape[-1]

    c = np.empty(shape, dtype=b.dtype).reshape(-1)
    c[0::2] = a.reshape(-1)
    c[1::2] = b.reshape(-1)

    c = c.reshape(shape)
    c = np.rollaxis(c, len(c.shape) - 1, axis)

    return c

def is_equidistant(x):
    '''
    >>> is_equidistant((0,1,2))
    True
    >>> is_equidistant((0,1,2.5))
    False
    '''
    d = np.diff(x)
    return (np.round_(d,8)==np.round_(d[0],8)).all()

def integrate_simpson(a, b, f):
    '''
    Simpson's 3-point rule O(h**4)
    http://en.wikipedia.org/wiki/Simpson%27s_rule

    >>> integrate_simpson(0, 1, lambda x: x)
    0.5
    >>> integrate_simpson(-1, -0.5, lambda x: x)
    -0.375
    '''
    I = ((b-a)/6.0)*(f(a) + 4*f((a+b)/2.0) + f(b))
    return I

def integrate_boole(x1, x5, f):
    '''
    Boole's 5-point rule O(h**7)
    http://en.wikipedia.org/wiki/Boole%27s_rule

    >>> integrate_boole(0, 1, lambda x: x)
    0.5
    >>> integrate_boole(0, 1, lambda x: x**4)
    0.2
    '''
    h = (x5 - x1)/4.0
    x2 = x1 + h
    x3 = x1 + 2*h
    x4 = x1 + 3*h
    #assert(x5 == x1 + 4*h)
    I = (2*h/45.0)*(7*f(x1) + 32*f(x2) + 12*f(x3) + 32*f(x4) + 7*f(x5))
    return I

def integrate_boole1(x, f):
    '''
    >>> integrate_boole1([0, 1], lambda x: x)
    array([ 0.5])
    '''
    x = np.asanyarray(x)
    return integrate_boole(x[:-1], x[1:], f)

def integrate_boole2(x1, x5, f):
    '''
    Boole's 5-point rule O(h**7) in 2D
    '''
    h = (x5 - x1)/4.0
    x2 = x1 + h
    x3 = x1 + 2*h
    x4 = x1 + 3*h
    I = (2*np.sqrt(h[0]**2 + h[1]**2)/45.0)*\
        (7*f(*x1) + 32*f(*x2) + 12*f(*x3) + 32*f(*x4) + 7*f(*x5))
    return I

def flat(f):
    if isinstance(f, tuple):
        return np.concatenate([x.ravel() for x in f])
    else:
        return f.ravel()

def integrate_1form(edge, f):
    '''
    Integrate a continuous one-form **f** along an **edge** ((x0, y0), (x1, y1))
    >>> integrate_1form( ((0,0), (1,0)), lambda x, y: (1, 0) )[0]
    array(1.0)
    >>> integrate_1form( ((0,0), (1,0)), lambda x, y: (0, 1) )[0]
    array(0.0)
    '''
    def tmp(x0, y0, x1, y1):
        dx = x1 - x0
        dy = y1 - y0
        def _f(t):
            x = x0 + t*dx
            y = y0 + t*dy
            fx, fy = f(x, y)
            return fx*dx + fy*dy
        return scipy.integrate.quad(_f, 0, 1)

    ((x0, y0), (x1, y1)) = edge
    return np.vectorize(tmp)(x0, y0, x1, y1)

def _integrate_2form(face, f):
    g = lambda x: 0
    h = lambda x: 1-x

    def tmp(x0, y0, x1, y1, x2, y2):
        dx1 = x1 - x0
        dy1 = y1 - y0
        dx2 = x2 - x0
        dy2 = y2 - y0
        _f = lambda u, v: f(x0 + u*dx1 + v*dx2,
                            y0 + u*dy1 + v*dy2)*(dx1*dy2 - dy1*dx2)
        return scipy.integrate.dblquad(_f, 0, 1, g, h)

    ((x0, y0), (x1, y1), (x2, y2)) = face
    return np.vectorize(tmp)(x0, y0, x1, y1, x2, y2)

def integrate_2form(face, f):
    '''
    Integrate a continuous two-form **f** on a **face** ((x0, y0), (x1, y1), (x2, y2), ...)
    >>> integrate_2form( ((0,0), (2,0), (0,2)), lambda x, y: 1 )[0]
    2.0
    >>> integrate_2form( ((0,0), (1,0), (1,1), (0,1)), lambda x, y: 1 )[0]
    1.0
    '''
    integral = 0.0
    error = 0.0

    a = face[0]
    for b, c in zip(face[1:-1], face[2:]):
        i, e = _integrate_2form((a,b,c), f)
        integral += i
        error += e

    return integral, error

def check(g, interface):
    '''
    Check whether an object satisfies an interface.

    >>> from dec.grid1 import *
    >>> check(Grid_1D_Periodic(4), Grid_1D_Interface)
    True
    >>> check(Grid_1D_Chebyshev(4), Grid_1D_Interface)
    True
    >>> check(Grid_1D_Regular(4), Grid_1D_Interface)
    True

    '''
    rslt = True
    for i in interface:
        rslt = (rslt and hasattr(g, i))
    return rslt
