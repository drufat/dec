"""
Spectral DEC
=============
"""
from __future__ import division
from dec.helper import *
from numpy import *
from numpy.testing import assert_almost_equal
import itertools
import operator
from functools import reduce
try:
    from scipy.linalg.special_matrices import toeplitz
except ImportError:
    from scipy.linalg.basic import toeplitz

seterr(divide='ignore', invalid='ignore')

def alpha0(N, x):
    r"""

    .. math::
        \alpha_{N}(x)=\frac{1}{N}
        \begin{cases}
            \cot\frac{x}{2}\,\sin\frac{Nx}{2} & \text{if N even,}\\
            \csc\frac{x}{2}\,\sin\frac{Nx}{2} & \text{if N odd.}
        \end{cases}
        
        
    >>> def α0(N, i): return round(alpha0(N, i*2*pi/N), 15)
    >>> (α0(5, 0), α0(5, 1), α0(5, 2)) == (1.0, 0.0, 0.0) 
    True
    >>> (α0(6, 0), α0(6, 1), α0(6, 2)) == (1.0, 0.0, 0.0)
    True
    
    """
    if N % 2 == 0:
        y = (sin(N*x/2) / tan(x/2)) / N
    else:
        y = (sin(N*x/2) / sin(x/2)) / N

    if hasattr(y, '__setitem__'):
        y[x==0] = 1
    else:
        if x==0: y = 1

    return y

def beta0(N, x):
    r"""

    .. math::
        \beta_{N}(x)=
        \begin{cases}
            \frac{1}{2\pi} -\frac{1}{4}\cos\frac{Nx}{2} +
                \frac{1}{N}\sum\limits_{n=1}^{N/2}
                    \frac{n\cos nx}{\sin\frac{n\pi}{N}} & \text{if N even,}\\
            \frac{1}{2\pi} +
                \frac{1}{N}\sum\limits_{n=1}^{(N-1)/2}
                    \frac{n\cos nx}{\sin\frac{n\pi}{N}} & \text{if N odd.}
        \end{cases}
    """
    if N % 2 == 0:
        y = 1/(2*pi) - cos(N*x/2)/4
        for n in range(1, N//2 + 1):
            y += n*cos(n*x) / sin(n*pi/N) / N
    else:
        y = 1/(2*pi) + 0*x
        for n in range(1, (N-1)//2 + 1):
            y += n*cos(n*x) / sin(n*pi/N) / N
    return y

def alpha(N, n, x):
    r"""

    .. math::
        \alpha_{N, n} (x) = \alpha_N(x-h n)
    """
    return alpha0(N, x-2*pi/N*n)

def beta(N, n, x):
    r"""

    .. math::
        \beta_{N, n} (x) = \beta_N(x-h n)
    """
    return beta0(N, x-2*pi/N*n)

def gamma(N, k):
    r"""

    .. math::
        \gamma_{N, n} = \int_0^{\frac{h}{2}=\frac{\pi}{N}} \beta_{N,n}(x) \, dx =
        \begin{cases}
        \frac{1}{2N} -
        \frac{(-1)^k}{2N} +
        \frac{1}{N}\sum\limits_{n=1}^{N/2}
        \frac{\sin(2 k n \pi/N) - \sin((2 k-1) n \pi/N)}
        {\sin\frac{n\pi}{N}} & \text{if N even,}\\
        \frac{1}{2N} +
        \frac{1}{N}\sum\limits_{n=1}^{(N-1)/2}
        \frac{\sin(2 k n \pi/N) - \sin((2 k-1) n \pi/N)}{\sin\frac{n\pi}{N}} & \text{if N odd.}
        \end{cases}

    """
    if N % 2 == 0:
        y = 1/(2*N) - (-1)**k/(2*N)
        for n in range(1, N//2 + 1):
            y += (sin(2*n*k*pi/N) - sin((2*k-1)*n*pi/N)) / sin(n*pi/N) / N
    else:
        y = 1/(2*N)
        for n in range(1, (N-1)//2 + 1):
            y += (sin(2*n*k*pi/N) - sin((2*k-1)*n*pi/N)) / sin(n*pi/N) / N
    return y

########################################
# Periodic Basis Functions
########################################

def phi0(N, n, x):
    r"""
    Basis functions for primal 0-forms.

    .. math::
        \phi_{N,n}^{0}(x)=\alpha_{N,n}(x)
    """
    return alpha(N, n, x)

def phi1(N, n, x):
    r"""
    Basis functions for primal 1-forms.

    .. math::
        \phi_{N,n}^{1}(x)=\beta_{N,n+\frac{1}{2}}(x)
    """
    return beta(N, n + 0.5, x)

def phid0(N, n, x):
    r"""
    Basis functions for dual 0-forms.

    .. math::
        \widetilde{\phi}_{N,n}^{0}(x)=\alpha_{N,n+\frac{1}{2}}(x)
    """
    return alpha(N, n + 0.5, x)

def phid1(N, n, x):
    r"""
    Basis functions for dual 1-forms.

    .. math::
        \widetilde{\phi}_{N,n}^{1}(x)=\alpha_{N,n}(x)
    """
    return beta(N, n, x)

########################################
# Mapping between semi-circle and line #
########################################

def varphi(x):
    r"""
    .. math::
        \varphi:&& [-1,1]\to[0,\pi]\\
                && x\mapsto\arccos(-x)

    >>> varphi(-1) == 0
    True
    >>> varphi(+1) == pi
    True
    """
    return arccos(-x)

def varphi_inv(x):
    r"""
    .. math::
        \varphi^{-1}:&& [0,\pi]\to[-1,1]\\
                && \theta\mapsto-\cos(\theta)

    >>> varphi_inv(0) == -1
    True
    >>> varphi_inv(pi) == +1
    True
    """
    return -cos(x)

############################################################
# Regular Basis Functions
############################################################

def delta(N, x):
    r"""
    Basis function for dual half-edge at the boundary.

    .. math::
        \delta_{N}(x)=\left((N-1)^{2}\alpha_{2N-2,0}(x)+\frac{1}{2}\cos\left((N-1)x\right)\right)\sin(x)
    """

    return ((N-1)**2 * alpha(2*N-2, 0, x) + 0.5*cos((N-1)*x))*sin(x)

def rho(N, n, x):
    r"""

    .. math::
        \rho_{N, n}(x) = 2 \gamma_{2N-2, n}\delta_N(x) + 2 \gamma_{2N-2, N-n-1} \delta_N(\pi-x)
    """
    return 2*gamma(2*N-2, n)*delta(N, x) + 2*gamma(2*N-2, N-n-1)*delta(N, pi-x)


def kappa0(N, n, x):
    r"""
    Basis functions for primal 0-forms.

    .. math::
        \kappa_{N,n}^{0}(x) =
        \begin{cases}
            \alpha_{2N-2,n}(x), & n\in\{0,N-1\}\\
            \alpha_{2N-2,n}(x)+\alpha_{2N-2,2N-2-n}(x), & n\in\{1,\dots,N-2\}
        \end{cases}
   """
    if n in (0, N-1):
        return  alpha(2*N-2, n, x)
    else:
        return (alpha(2*N-2, n, x) +
                alpha(2*N-2, 2*N-2-n, x))

def kappa1(N, n, x):
    r"""

    .. math::
        \kappa_{N,n}^{1}(x) = \left(
        \beta_{2N-2,n+\frac{1}{2}}(x)-
        \beta_{2N-2,2N-3-n+\frac{1}{2}}(x)\right)
        \mathbf{d}x,n\in\{0,\dots,N-2\}
    """
    return (beta(2*N-2, n+0.5, x) -
            beta(2*N-2, 2*N-3-n+0.5, x))

def kappad0(N, n, x):
    r"""

    .. math::
        \widetilde{\kappa}_{N,n}^{0}(x)=
        \alpha_{2N-2,\, n+\frac{1}{2}}(x)+
        \alpha_{2N-2,\,2N-3-n+\frac{1}{2}}(x),\quad n\in\{0,\dots,N-2\}
    """
    return (alpha(2*N-2, n+0.5, x) +
            alpha(2*N-2, 2*N-3-n+0.5, x))

def kappad1(N, n, x):
    r"""

    .. math::
        \widetilde{\kappa}_{N,n}^{1}(x)=
        \begin{cases}
            \delta(x)\mathbf{d}x & \qquad n=0\\
            \left(\beta_{2N-2,n}(x)-\beta_{2N-2,2N-2-n}(x)-\rho_{N,n}(x)\right)\mathbf{d}x & \qquad n\in\{1,\dots,N-2\}\\
            \delta(\pi-x)\mathbf{d}x & \qquad n=N-1
        \end{cases}
    """
    if n == 0:
        return delta(N, x)
    if n == N-1:
        return delta(N, pi - x)

    y = (beta(2*N-2, n, x) -
         beta(2*N-2, 2*N-2-n, x)
         - rho(N, n, x) )
    return y

############################################################
# Chebyshev Basis Functions
# They seem to be all polynomials:
# >>> x = linspace(-1, +1, 10000)
# >>> round_(polyfit(x, psid1(5, 5, x), 10), 3)[::-1]
# array([ 0.125,  0.657, -2.177, -9.941, -5.121,  9.941,  7.877, -0.   ,
#        -0.   ,  0.   ,  0.   ])
############################################################

#TODO: This is a dirty hack. Compute the limits explicilty?
def __fix_singularity_at_boundary(x):
    # Avoid division by zero
    # Make sure function has no side effects
    if type(x) is ndarray:
        x += (x==-1)*1e-16 - (x==+1)*1e-16
    else:
        if x==-1: x += 1e-16
        if x==+1: x -= 1e-16
    return x

def psi0(N, n, x):
    r"""
    Basis functions for primal 0-forms.

    .. math::
        \psi_{N,n}^{0}(x)=\kappa_{N,n}^{0}(\arccos(-x))
    """
    return kappa0(N, n, arccos(-x))

def psi1(N, n, x):
    r"""
    Basis functions for primal 1-forms.

    .. math::
        \psi_{N,n}^{1}(x)\mathbf{d}x=
            \kappa_{N,n}^{1}(\arccos(-x))\frac{\mathbf{d}x}{\sqrt{1-x^{2}}}
    """
    #x = __fix_singularity_at_boundary(x)
    return kappa1(N, n, arccos(-x))/sqrt(1 - x**2)

def psid0(N, n, x):
    r"""
    Basis functions for dual 0-forms.

    .. math::
        \tilde{\psi}_{N,n}^{0}(x)=\tilde{\kappa}_{N,n}^{0}(\arccos(-x))
    """
    return kappad0(N, n, arccos(-x))

def psid1(N, n, x):
    r"""
    Basis functions for dual 1-forms.

    .. math::
        \tilde{\psi}_{N,n}^{1}(x)\mathbf{d}x=\tilde{\kappa}_{N,n}^{1}(\arccos(-x))\frac{\mathbf{d}x}{\sqrt{1-x^{2}}}
    """
    x = __fix_singularity_at_boundary(x)
    return kappad1(N, n, arccos(-x))/sqrt(1 - x**2)

###########################
# Lagrange polynomials
###########################

def lagrange_polynomials(x):
    r"""
    Lagrange Polynomials for the set of points defined by :math:`x_m`.
    The Lagrange Polynomials are such that they are 1 at the point, and 0
    everywhere else.

    .. math::
        \psi_{n}^{0}(x)=\prod_{m=0,m\neq n}^{N-1}\frac{x-x_{m}}{x_{n}-x_{m}}

    >>> L = lagrange_polynomials([0, 1, 2])
    >>> [l(0) for l in L]
    [1.0, 0.0, -0.0]
    >>> [l(1) for l in L]
    [-0.0, 1.0, 0.0]
    >>> [l(2) for l in L]
    [0.0, -0.0, 1.0]

    """
    N = len(x)
    L = [None]*N
    for i in range(N):
        def Li(y, i=i):
            gen = ((y-x[j])/(x[i]-x[j]) for j in set(range(N)).difference([i]))
            return reduce(operator.mul, gen)
        L[i] = (Li)
    return L

def freq(N):
    """
    >>> freq(5)
    array([-2., -1.,  0.,  1.,  2.])
    >>> freq(6)
    array([-3., -2., -1.,  0.,  1.,  2.])
    """
    return fft.fftshift(fft.fftfreq(N)*N)

F    = lambda x: fft.fftshift(fft.fft(x))
Finv = lambda x: fft.ifft(fft.ifftshift(x))

def method_in_fourier_space(g):
    def f(x, *args, **keywords):
        return Finv(g(F(x), *args, **keywords))
    f.__doc__ = g.__doc__ # make sure doctests are run
    return f

def H(a):
    r"""

    .. math::
        \mathbf{H}^0 = \widetilde{\mathbf{H}}^0 =
        \mathcal{F}^{-1} \mathbf{I}^{-\frac{h}{2}, \frac{h}{2}} \mathcal{F}
    """
    N = len(a)
    h = 2*pi/N
    return Finv( F(a) * I_diag(N, -h/2, h/2) )

def Hinv(a):
    r"""

    .. math::
        \mathbf{H}^1 = \widetilde{\mathbf{H}}^1 =
        \mathcal{F}^{-1}{\mathbf{I}^{-\frac{h}{2},\frac{h}{2}}}^{-1}\mathcal{F}
    """
    N = len(a)
    h = 2*pi/N
    return Finv( F(a) / I_diag(N, -h/2, h/2) )

def I(a, x0, x1):
    N = len(a)
    return Finv ( I_diag(N, x0, x1) * F(a) )

def Iinv(a, x0, x1):
    N = len(a)
    return Finv ( F(a) / I_diag(N, x0, x1) )

def S(a):
    N = len(a)
    h = 2*pi/N
    return Finv( F(a) * S_diag(N, h/2) )

def Sinv(a):
    N = len(a)
    h = 2*pi/N
    return Finv( F(a) * S_diag(N, -h/2) )

def I_diag(N, a, b):
    r"""

    A diagonal matrix that corresponds to integration in Fourier space.
    Corresponds to :math:`f(x) \mapsto \int_{x+a}^{x+b} f(\xi) d\xi`

    .. math::
        \mathbf{I}_{\phantom{a,b}nn}^{a,b}
            =\frac{e^{inb}-e^{ina}}{in}

    .. math::
        \mathbf{I}_{\phantom{a,b}00}^{a,b}=b-a
    """
    n = freq(N)
    y = (exp(1j*n*b) - exp(1j*n*a))/(1j*n)
    y[n==0] = b - a
    return y

def S_diag(N, a):
    r"""

    A diagonal matrix that corresponds to shifting in Fourier Space
    Corresponds to :math:`f(x) \mapsto f(x-h)`

    .. math::
        \mathbf{S}_{\phantom{a}nn}^{a}=e^{ina}
    """
    n = freq(N)
    return exp(1j*n*a)

def fourier_I(x, a, b):
    r"""
    Corresponds to :math:`f(x) \mapsto \int_{x+a}^{x+b} f(\xi) d\xi`
    """
    N = x.shape[0]
    n = freq(N)
    y = (exp(1j*n*b) - exp(1j*n*a))/(1j*n)
    y[n==0] = b - a
    return x*y

def fourier_I_inv(x, a, b):
    N = x.shape[0]
    n = freq(N)
    y = (exp(1j*n*b) - exp(1j*n*a))/(1j*n)
    y[n==0] = b - a
    return x/y

def fourier_T(x, h):
    r"""
    Corresponds to :math:`f(x) \mapsto f(x+a)`
    """
    N = x.shape[0]
    n = freq(N)
    return x*exp(1j*n*h)

def fourier_J(x, a, b, c):
    r"""
    Corresponds to :math:`f(x) \mapsto \int_{x+a}^{x+b} f(\xi) \sin(\xi+c) d\xi`
    """
    c = fourier_I(x, a, b)
    b = (roll(c[2:]*x,+1)*exp(1j*c) - roll(c[:-2]*x,-1)*exp(-1j*c))/2j
    return b

def refine(x):
    """
    Resample x at a twice refined grid.
    >>> N = 4
    >>> x = linspace(0, 2*pi, N+1)[:-1]
    >>> y = linspace(0, 2*pi, 2*N+1)[:-1]
    >>> approx(refine(cos(x)), cos(y))
    True
    """
    x = interweave(x, S(x))
    return x

def subdivide(x):
    """
    Subdivide x like below. Assume points are equidistant.
    *   *   *   *   *   *
    * * * * * * * * * * *
    """
    if len(x) < 2: return x
    assert is_equidistant(x)
    y = interweave(x[:-1], x[:-1] + 0.5*diff(x))
    return concatenate((y, [x[-1]]))

def integrate_spectral_coarse(x, f):
    """
    >>> integrate_spectral_coarse(linspace(-pi, pi, 3), sin)
    array([-2.,  2.])
    """
    assert is_equidistant(x)
    assert approx(2*pi, x[-1] - x[0]), x
    f0 = f(x[:-1] + 0.5*diff(x))
    f1 = real(H(f0))
    return f1

def integrate_spectral(x, f):
    '''
    >>> integrate_spectral(linspace(-pi, pi, 3), sin)
    array([-2.,  2.])
    '''
    assert is_equidistant(x)

    r = subdivide
    f1 = integrate_spectral_coarse(r(r(r(x))), f)

    return (f1[0::8] + f1[1::8] + f1[2::8] + f1[3::8] +
            f1[4::8] + f1[5::8] + f1[6::8] + f1[7::8])

def integrate_chebyshev(xi, f):
    """
    >>> integrate_chebyshev(array([cos(pi), cos(.5*pi), cos(0)]), lambda x: x)
    array([-0.5,  0.5])
    """
    assert approx(2, xi[-1] - xi[0]), xi

    F = lambda theta: f(-cos(theta))*sin(theta)
    x = varphi(xi)

    assert(is_equidistant(x))

    # complete the circle from the other side
    x = concatenate((x, x[1:]+pi))
    return integrate_spectral(x, F)[:len(xi)-1]

def integrate_chebyshev_dual(xi, f):
    """
    Integrate points that may include the two half-edges at the boundary.
    #>>> integrate_chebyshev_dual(array([cos(pi), cos(0.75*pi), cos(0.25*pi), cos(0)]), lambda x: x)
    #array([-0.25,  0.  ,  0.25])
    """
    x = varphi(xi)
    z = varphi_inv(concatenate(([x[0]], subdivide(x[1:-1]), [x[-1]])))
    i, j = concatenate(( [0], integrate_chebyshev(z, f), [0] )).reshape(-1, 2).T
    return i+j

def split_args(I):
    """
    Convert integration function from I(x, f) to I(x0, x1, f) form
    """
    def J(x0, x1, f, I=I):
        assert_almost_equal(x0[1:], x1[:-1])
        return I(concatenate((x0, [x1[-1]])), f)
    return J

##############################
# Discrete exterior calculus
##############################

def hodge_star_matrix(P, B):
    """
    Compute Hodge-Star matrix from basis functions.
    """
    P0, P1, P0d, P1d = P
    B0, B1, B0d, B1d = B
    H0 = vstack(P1d(b) for b in B0).T
    H1 = vstack(P0d(b) for b in B1).T
    H0d = vstack(P1(b) for b in B0d).T
    H1d = vstack(P0(b) for b in B1d).T
    return H0, H1, H0d, H1d

def reconstruction(basis_fn):
    """
    Give the reconstruction functions for the set of basis functions basis_fn.
    """
    def summation(a, B):
        return lambda *x: sum(a[i]*f(*x) for i, f in enumerate(B))
    return [(lambda a: summation(a, B)) for B in basis_fn]

Grid_1D_Interface = """
        pnts,
        xmin, xmax, lenx,
        verts, verts_dual,
        edges, edges_dual,
        basis_fn,
        B0, B1, B0d, B1d,
        projection,
        P0, P1, P0d, P1d,
        reconstruction,
        R0, R1, R0d, R1d,
        derivative,
        D0, D0d,
        hodge_star,
        H0, H1, H0d, H1d,
    """

def Grid_1D_Periodic(n, xmin=0, xmax=2*pi):

    assert xmax > xmin

    dimension = 1

    pnts = linspace(xmin, xmax, num=(n+1))
    lenx = abs(xmax - xmin)
    delta = diff(pnts)

    verts = pnts[:-1]
    edges = (pnts[:-1], pnts[1:])

    verts_dual = verts + 0.5*delta
    edges_dual = (roll(verts_dual,shift=1), verts_dual)
    edges_dual[0][0] -= lenx
    delta_dual = delta

    V = verts
    S0 = arange(len(V))
    S1 = (S0[:-1], S0[1:])

    Vd = verts_dual
    S0d = arange(len(Vd))
    S1d = (S0d[:-1], S0d[1:])

    P0 = lambda f: f(verts)
    P1 = lambda f: split_args(integrate_spectral)(
                    edges[0], edges[1], f)
    P0d = lambda f: f(verts_dual)
    P1d = lambda f: split_args(integrate_spectral)(
                    edges_dual[0], edges_dual[1], f)
    def projection():
        return P0, P1, P0d, P1d

    B0 = [lambda x, i=i: phi0(n, i, x) for i in range(n)]
    B1 = [lambda x, i=i: phi1(n, i, x) for i in range(n)]
    B0d = [lambda x, i=i: phid0(n, i, x) for i in range(n)]
    B1d = [lambda x, i=i: phid1(n, i, x) for i in range(n)]
    def basis_fn():
        return B0, B1, B0d, B1d

    R0, R1, R0d, R1d = reconstruction(basis_fn())

    D0  = lambda f: roll(f, shift=-1) - f
    D0d = lambda f: roll(D0(f), shift=+1)
    def derivative():
        return D0, D0d

    H0 = lambda x: real(H(x))
    H1 = lambda x: real(Hinv(x))
    H0d = H0
    H1d = H1
    def hodge_star():
        return H0, H1, H0d, H1d

    def derivative_matrix():
        rng = arange(n)
        ons = ones(n)
        d = row_stack((
                   column_stack((
                     rng,
                     roll(rng, shift= -1),
                     +ons)),
                   column_stack((
                     rng,
                     rng,
                     -ons))
                   ))
        D = sparse_matrix(d, n, n)
        return D, -D.T

    def differentiation_toeplitz():
        raise NotImplemented
        #TODO: fix this implementation
        h = 2*pi/n
        assert n % 2 == 0
        column = concatenate(( [ 0 ], (-1)**arange(1,n) / tan(arange(1,n)*h/2)  ))
        row = concatenate(( column[:1], column[1:][::-1] ))
        D = toeplitz(column, row)
        return D

    def hodge_star_toeplitz():
        """
        The Hodge-Star using a Toeplitz matrix.
        """
        column = P1d(lambda x: alpha0(n, x))
        row = concatenate((column[:1], column[1:][::-1]))
        return toeplitz(column, row)

    def wedge():
        """
        Return \alpha ^ \beta. Keep only for primal forms for now.
        """
        def w00(alpha, beta):
            return alpha * beta
        def _w01(alpha, beta):
            return S(H( alpha * Hinv(Sinv(beta)) ))
        def w01(alpha, beta):
#            a = interweave(alpha, T(alpha, [S]))
#            b = interweave(T(beta, [Hinv, Sinv]), T(beta, [Hinv]))
            a = refine(alpha)
            b = refine(Hinv(Sinv(beta)))
            c = S(H(a * b))
            return c[0::2] + c[1::2]
        return w00, w01, _w01

    def contraction(V):
        """
        Return i_V. Keep only for primal forms for now.
        """
        def C1(alpha):
            return Hinv(Sinv(V)) * Hinv(Sinv(alpha))
        return C1

    return bunch(**locals())

def _slow_integration(a, b, f):
    from scipy.integrate import quad
    return array([quad(f, _a, _b)[0] for _a, _b in zip(a, b)])

def A_diag(N):
    r"""

    .. math::
        \mathbf{A}=\text{diag}\left(\begin{array}{ccccccc}\frac{1}{2} & 1 & 1 & \dots & 1 & 1 & \frac{1}{2}\end{array}\right)

    """
    d = concatenate(([0.5], ones(N-2), [0.5]))
    return d

def H0_regular(f):
    r"""

    .. math::
        \mathbf{H}^{0}=
            \mathbf{A}
            \mathbf{M}_{0}^{\dagger}
            \mathbf{I}^{-\frac{h}{2},\frac{h}{2}}
            \mathbf{M}_{0}^{+}
    """
    f = mirror0(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = I_space(-h/2, h/2)(f)
    f = unmirror0(f)
    f = f*A_diag(f.shape[0])
    return  real(f)

def H1_regular(f):
    r"""

    .. math::
        \mathbf{H}^{1}=
            \mathbf{M}_{1}^{\dagger}
            {\mathbf{I}^{-\frac{h}{2},\frac{h}{2}}}^{-1}
            \mathbf{M}_{1}^{-}
    """
    f = mirror1(f, -1)
    N = f.shape[0]; h = 2*pi/N
    f = I_space_inv(-h/2, h/2)(f)
    f = unmirror1(f)
    return f

def H0d_regular(f):
    r"""

    .. math::
        \widetilde{\mathbf{H}}^{0}=
            \mathbf{M}_{1}^{\dagger}
            \mathbf{I}^{-\frac{h}{2},\frac{h}{2}}
            \mathbf{M}_{1}^{-}
    """
    f = mirror1(f, -1)
    N = f.shape[0]; h = 2*pi/N
    f = I_space(-h/2, h/2)(f)
    f = unmirror1(f)
    return f

def H1d_regular(f):
    r"""

    .. math::
        \widetilde{\mathbf{H}}^{1}=
            \mathbf{M}_{1}^{\dagger}
            {\mathbf{I}^{-\frac{h}{2},\frac{h}{2}}}^{-1}
            \mathbf{M}_{1}^{+}
            \mathbf{A}^{-1}
    """
    f = f/A_diag(f.shape[0])
    f = mirror0(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = I_space_inv(-h/2, +h/2)(f)
    f = unmirror0(f)
    return f

def Grid_1D_Regular(n, xmin=0, xmax=pi):

    assert xmax > xmin

    dimension = 1

    N = n
    pnts = linspace(xmin, xmax, num=n)
    lenx = abs(xmax - xmin)
    delta = diff(pnts)

    verts = pnts
    verts_dual = verts[:-1] + 0.5*delta

    edges = (pnts[:-1], pnts[1:])
    tmp = concatenate(([xmin], verts_dual, [xmax]))
    delta_dual = diff(tmp)
    edges_dual = (tmp[:-1], tmp[1:])

    P0 = lambda f: f(verts)
    P1 = lambda f: _slow_integration(edges[0], edges[1], f)
    P0d = lambda f: f(verts_dual)
    P1d = lambda f: _slow_integration(edges_dual[0], edges_dual[1], f)
    def projection():
        return P0, P1, P0d, P1d

    B0 = [lambda x, i=i: kappa0(n, i, x) for i in range(n)]
    B1 = [lambda x, i=i: kappa1(n, i, x) for i in range(n-1)]
    B0d = [lambda x, i=i: kappad0(n, i, x) for i in range(n-1)]
    B1d = [lambda x, i=i: kappad1(n, i, x) for i in range(n)]
    def basis_fn():
        return B0, B1, B0d, B1d

    R0, R1, R0d, R1d = reconstruction(basis_fn())

    D0 = lambda f: diff(f)
    D0d = lambda f: diff(concatenate(([0], f, [0])))
    def derivative():
        return D0, D0d

    H0 = H0_regular
    H1 = H1_regular
    H0d = H0d_regular
    H1d = H1d_regular
    def hodge_star():
        return H0, H1, H0d, H1d

    return bunch(**locals())

def extend(f, n):
    r"""

    .. math::
        \mathbf{E}^{n}:\quad
        \begin{bmatrix}x_{0}\\
        \vdots\\
        x_{N-1}
        \end{bmatrix}
        \mapsto
        \frac{N+2n}{N}
        \begin{bmatrix}\left.\begin{array}{c}
        0\\
        \vdots\\
        0
        \end{array}\right\} n\\
        \begin{array}{c}
        x_{0}\\
        \vdots\\
        x_{N-1}
        \end{array}\\
        \left.\begin{array}{c}
        0\\
        \vdots\\
        0
        \end{array}\right\} n
        \end{bmatrix}

    """
    N = f.shape[0]
    return (N+2.0*n)/N*_extend(f, n)

def unextend(f, n):
    r"""

    .. math::
        \mathbf{E}^{-n}:\quad\begin{bmatrix}\left.\begin{array}{c}
        x_{-n}\\
        \vdots\\
        x_{-1}
        \end{array}\right\} n\\
        \begin{array}{c}
        x_{0}\\
        \vdots\\
        x_{N-1}
        \end{array}\\
        \left.\begin{array}{c}
        x_{N}\\
        \vdots\\
        x_{N+n-1}
        \end{array}\right\} n
        \end{bmatrix}\mapsto\frac{N-2n}{N}\begin{bmatrix}\left.\begin{array}{c}
        x_{0}+x_{N}\\
        \vdots\\
        x_{n-1}+x_{N+n-1}
        \end{array}\right\} n\\
        \begin{array}{c}
        x_{n}\\
        \vdots\\
        x_{N-n-1}
        \end{array}\\
        \left.\begin{array}{c}
        x_{N-n}+x_{-n}\\
        \vdots\\
        x_{N-1}+x_{-1}
        \end{array}\right\} n
        \end{bmatrix}
    """
    N = f.shape[0]
    return (N-2.0*n)/N*_unextend(f, n)

def _extend(f, n):
    """
    >>> _extend([1, 2, 3, 4], 2)
    array([ 0.,  0.,  1.,  2.,  3.,  4.,  0.,  0.])
    """
    return concatenate(( zeros(n),
                         f,
                         zeros(n) ))

def _unextend(f, n):
    """
    >>> _unextend(array([0,0,1,2,3,4,0,0]), 2)
    array([1, 2, 3, 4])
    >>> _unextend(array([1,2,0,0,3,4]), 2)
    array([4, 6])
    >>> _unextend(array([0,1,2,3,4,5,6,7]), 2)
    array([ 8, 10,  4,  6])
    """
    assert 2*n <= f.shape[0]
    x = f[n:-n].copy()
    x[:n]  += f[-n:]
    x[-n:] += f[:n]
    return x

def mirror0(f, sign=+1):
    r"""

    .. math::
        \mathbf{M}_{0}^{\pm}:\quad\begin{bmatrix}x_{0}\\
                x_{1}\\
                \vdots\\
                x_{N-2}\\
                x_{N-1}
            \end{bmatrix}\mapsto
            \begin{bmatrix}x_{0}\\
                x_{1}\\
                \vdots\\
                x_{N-2}\\
                x_{N-1}\\
                \pm x_{N-2}\\
                \vdots\\
                \pm x_{1}
            \end{bmatrix}

    >>> mirror0(array([1, 2, 3]))
    array([1, 2, 3, 2])
    >>> mirror0(array([1, 2, 3]), -1)
    array([ 1,  2,  3, -2])
    >>> to_matrix(mirror0, 3)
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.]])
    >>> to_matrix(lambda x: mirror0(x, -1), 3)
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [-0., -1., -0.]])
   """
    return concatenate((f, sign*f[::-1][1:-1]))

def mirror1(f, sign=+1):
    r"""

    .. math::
        \mathbf{M}_{1}^{\pm}:\quad
        \begin{bmatrix}x_{0}\\
            x_{1}\\
            \vdots\\
            x_{N-2}\\
            x_{N-1}
        \end{bmatrix}\mapsto
        \begin{bmatrix}x_{0}\\
            x_{1}\\
            \vdots\\
            x_{N-2}\\
            x_{N-1}\\
            \pm x_{N-1}\\
            \pm x_{N-2}\\
            \vdots\\
            \pm x_{1}\\
            \pm x_{0}
        \end{bmatrix}

    >>> mirror1(array([1, 2, 3]))
    array([1, 2, 3, 3, 2, 1])
    >>> mirror1(array([1, 2, 3]), -1)
    array([ 1,  2,  3, -3, -2, -1])
    >>> to_matrix(mirror1, 2)
    array([[ 1.,  0.],
           [ 0.,  1.],
           [ 0.,  1.],
           [ 1.,  0.]])
    >>> to_matrix(lambda x: mirror1(x, -1), 2)
    array([[ 1.,  0.],
           [ 0.,  1.],
           [-0., -1.],
           [-1., -0.]])
    """
    return concatenate((f, sign*f[::-1]))

def unmirror0(f):
    r"""

    .. math::
        \mathbf{M}_{0}^{\dagger}:\quad\begin{bmatrix}x_{0}\\
        x_{1}\\
        \vdots\\
        x_{N-1}\\
        x_{N}\\
        \vdots\\
        x_{2N-1}
        \end{bmatrix}\mapsto\begin{bmatrix}x_{0}\\
        x_{1}\\
        \vdots\\
        x_{N-1}\\
        x_{N}\\
        x_{N+1}
        \end{bmatrix}

    >>> unmirror0(array([1, 2, 3, 2]))
    array([1, 2, 3])
    """
    return f[:(len(f)/2+1)]

def unmirror1(f):
    r"""

    .. math::
        \mathbf{M}_{1}^{\dagger}:\quad\begin{bmatrix}x_{0}\\
        x_{1}\\
        \vdots\\
        x_{N-1}\\
        x_{N}\\
        \vdots\\
        x_{2N-1}
        \end{bmatrix}\mapsto\begin{bmatrix}x_{0}\\
        x_{1}\\
        \vdots\\
        x_{N-1}
        \end{bmatrix}

    >>> unmirror1(array([1, 2, 3, -3, -2, -1]))
    array([1, 2, 3])
    """
    return f[:(len(f)/2)]

#########
# Methods using FFT
#########

def H_exp(a):
    r"""
    :math:`\int_{x-h/2}^{x+h/2}f(\xi) \exp(i \xi) d\xi`

    This transformation is singular non-invertible.
    """

    N = len(a)
    h = 2*pi/N
    c = I_diag(N+2, -h/2, h/2)

    a = F(a)
    b = roll(c[2:]*a,+1)
    return Finv(b)

def H_sin(a):
    r"""
    :math:`\int_{x-h/2}^{x+h/2}f(\xi) \sin(\xi) d\xi`

    This transformation is singular non-invertible.

    >>> x = linspace(0, 2*pi, 6)[:-1]
    >>> round_(real( H_sin(sin(2*x)) ), 3)
    array([ 0.271,  0.438, -0.573, -0.573,  0.438])
    """

    N = len(a)
    h = 2*pi/N

    a = F(a)
    c = I_diag(N+2, -h/2, h/2)
    b = (roll(c[2:]*a,+1) - roll(c[:-2]*a,-1))/2j

    return Finv(b)

def H_sin_dual(a):
    r"""
    :math:`\int_{x-h/2}^{x+h/2}f(\xi) \sin(\xi+\frac{h}{2}) d\xi`

    This transformation is singular non-invertible.

    >>> x = linspace(0, 2*pi, 6)[:-1]
    >>> round_(real( H_sin_dual(sin(2*x)) ), 3)
    array([ 0.219,  0.573, -0.084, -0.844,  0.135])
    """

    N = len(a)
    h = 2*pi/N

    a = F(a)
    c = I_diag(N+2, -h/2, h/2)
    s = exp(1j*h/2)
    b = (roll(c[2:]*a,+1)*s - roll(c[:-2]*a,-1)/s)/2j

    return Finv(b)

def Omega(N):
    r"""

    .. math::
        \mathbf{\Omega}_{nn}=\sin\left(nh\right)

    """

    h = 2*pi/N
    o = sin( arange(N)*h )
    return o

def Omega_d(N):
    r"""

    .. math::
        \mathbf{\widetilde{\Omega}}_{nn}
            =\sin\left(\left(n+\frac{1}{2}\right)h\right)

    If :math:`\widetilde{\omega}` is the length of each dual edge, then

    .. math::
        \mathbf{\widetilde{\Omega}}
            =\text{diag}(\mathbf{M}_{1}^{\dagger}
            \mathcal{F}^{-1}{\mathbf{I}^{-\frac{h}{2},\frac{h}{2}}}^{-1}
            \mathcal{F}\mathbf{M}_{1}^{-}\widetilde{\omega})
    """

    h = 2*pi/N
    o = sin( (arange(N)+0.5)*h )
    return o

def half_edge_base(N):
    r"""
    This is the discrete version of :math:`\delta(x)` - the basis functions
    for the half edge at -1

    .. math::
        \mathbf{B}=
            \text{diag}\left(\underset{N}{\underbrace{
            (N-1)^{2},-\frac{1}{2},\frac{1}{2},-\frac{1}{2},\cdots}
            }\right)

    .. math::
        \mathbf{B}^\dagger=
            \text{diag}\left(\underset{N}{\underbrace{
            \cdots,-\frac{1}{2},\frac{1}{2},-\frac{1}{2},(N-1)^{2}}
            }\right)


    >>> half_edge_base(3)
    array([ 4.5, -0.5,  0.5])
    """
    a = 0.5*ones(N)
    a[1::2] += -1
    a[0] += (N-1)**2
    return a

def half_edge_integrals(f):
    """
    >>> approx( half_edge_integrals([1,0,0]), gamma(3, 0))
    True
    >>> approx( half_edge_integrals([0,1,0]), gamma(3, 1))
    True
    """
    N = len(f)
    h = 2*pi/N
    f = Hinv(f)
    p = I(f, 0, h/2)
    return p[0]

def pick(f, n):
    r"""

    Pick the nth element in the array f.

    .. math::
        \mathcal{\mathbf{P}}^{n}:\quad\begin{bmatrix}x_{0}\\
        \vdots\\
        x_{n-1}\\
        x_{n}\\
        x_{n+1}\\
        \vdots\\
        x_{N-1}
        \end{bmatrix}\mapsto\begin{bmatrix}x_{n}\\
        \vdots\\
        x_{n}\\
        x_{n}\\
        x_{n}\\
        \vdots\\
        x_{n}
        \end{bmatrix}

    .. math::
       \mathcal{\mathbf{P}}^{0}=\begin{pmatrix}1 & 0 & 0 & 0 & 0\\
        1\\
        1\\
        1\\
        1
        \end{pmatrix},\quad\mathcal{\mathbf{P}}^{1}=\begin{pmatrix} 0 & 1 & 0 & 0 & 0\\
         & 1\\
         & 1\\
         & 1\\
         & 1
        \end{pmatrix},\dots

    """

    return f[n]*ones(f.shape[0])


def reverse(f):
    r"""

    Reverse array.

    .. math::
        \mathbf{R}:\quad\begin{bmatrix}x_{0}\\
        \vdots\\
        x_{n-1}\\
        x_{n}\\
        x_{n+1}\\
        \vdots\\
        x_{N-1}
        \end{bmatrix}\mapsto\begin{bmatrix}x_{N-1}\\
        \vdots\\
        x_{n+1}\\
        x_{n}\\
        x_{n-1}\\
        \vdots\\
        x_{0}
        \end{bmatrix}

    .. math::
        \mathbf{R} = \begin{pmatrix} &  &  &  & 1\\
         &  &  & 1\\
         &  & 1\\
         & 1\\
        1
        \end{pmatrix}
    """
    return f[::-1]


def I_space(a, b):
    return lambda f: Finv(F(f)*I_diag(f.shape[0], a, b))

def I_space_inv(a, b):
    return lambda f: Finv(F(f)/I_diag(f.shape[0], a, b))

def T_space(a):
    return lambda f: Finv(F(f)*S_diag(f.shape[0], a))

def T_space_inv(a):
    return lambda f: Finv(F(f)/S_diag(f.shape[0], a))

def E_space(n):
    return lambda f: Finv(extend(F(f), n))

def E_space_inv(n):
    return lambda f: Finv(unextend(F(f), n))

def matA(a):
    return concatenate(( [a[0]], a[1:-1:2]+a[2:-1:2] , [a[-1]] ))

def matB(f):
    b = half_edge_base(f.shape[0])
    return b*f[0]

def matB1(f):
    b = half_edge_base(f.shape[0])
    return b[::-1]*f[-1]

def matC(f):
    return concatenate(( [0], f[1:-1], [0] ))

def H0d_cheb(f):
    r"""

    .. math::
        \widetilde{\mathbf{H}}^0 = {\mathbf{M}_1}^{\dagger}
                \mathbf{I}^{-\frac{h}{2}, \frac{h}{2}}
                \widetilde{\mathbf{\Omega}} \mathbf{M}_1^+
    """

    f = mirror1(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = f*Omega_d(N)
    f = I_space(-h/2, h/2)(f)
    f = unmirror1(f)

    return f

def H1_cheb(f):
    r"""

    .. math::
        \mathbf{H}^1 = {\mathbf{M}_1}^{\dagger}
                \widetilde{\mathbf{\Omega}}^{-1}
                {\mathbf{I}^{-\frac{h}{2}, \frac{h}{2}}}^{-1}
                \mathbf{M}_1^-
    """

    f = mirror1(f, -1)
    N = f.shape[0]; h = 2*pi/N
    f = I_space_inv(-h/2, h/2)(f)
    f = f/Omega_d(N)
    f = unmirror1(f)

    return f

def H0_cheb(f):
    """
    Attempt to simplify the hodge-star.
    """
    N = f.shape[0]; h = pi/(N-1)
    f = mirror0(f, +1)
    f = E_space(N-1)(f)
    f = I_space(0, h/2)(f*Omega(f.shape[0]))
    f = unmirror1(f)
    f = matA(f)
    return real(f)

def H0_cheb_alternate(f):
    r"""

    .. math::
        \mathbf{H}^0 = \mathbf{M}_0^{\dagger}
            (\mathcal{A}^{0} \mathbf{E}^{-1} \mathbf{I}^{-\frac{h}{2}, 0} +
             \mathcal{A}^{N-1} \mathbf{E}^{-1} \mathbf{I}^{0, +\frac{h}{2}})
             \mathbf{\Omega} \mathbf{E}^{1}
             \mathbf{M}_0^{+}
    """
    N = f.shape[0]
    f = mirror0(f, +1)
    h = 2*pi/f.shape[0]
    f = E_space(1)(f)
    f = f*Omega(f.shape[0])
    l, r = I_space(-h/2, 0)(f), I_space(0, +h/2)(f)
    l, r = E_space_inv(1)(l), E_space_inv(1)(r)
    l[0], r[N-1] = 0, 0
    f = l+r
    f = unmirror0(f)
    #TODO: Is it possible to avoid discarding the imaginary part?
    f = real(f)
    return  f

def H1d_cheb(f):
    r"""

    .. math::

        \widetilde{\mathbf{H}}^{1} = \mathbf{M}_{0}^{\dagger}\left(\mathbf{T^{-\frac{h}{2}}}\mathbf{\Omega}^{-1}\mathbf{T}^{\frac{h}{2}}-\mathbf{B}\mathbf{I}^{0,\frac{h}{2}}-\mathbf{B}^{\dagger}\mathbf{I}^{-\frac{h}{2},0}\right)\mathbf{I}^{-\frac{h}{2},\frac{h}{2}}^{-1}\mathbf{M}_{0}^{-}\mathbf{C}+\mathbf{B}+\mathbf{B}^{\dagger}

    """
    N = f.shape[0]; h = pi/(N-1)
    B = half_edge_base(N)
    def endpoints(f):
        f0 = mirror0(matC(f), -1)
        aa = f - unmirror0(I_space(0, h/2)(I_space_inv(-h/2, h/2)(f0)))
        bb = f - unmirror0(I_space(-h/2, 0)(I_space_inv(-h/2, h/2)(f0)))
        return matB(aa) + matB1(bb)
    def midpoints(f):
        f = mirror0(matC(f), -1)
        # Shift function with S, Sinv to avoid division by zero at x=0, x=pi
        f = I_space_inv(-h/2, h/2)(f)
        f = T_space(+h/2)(f)
        f = f/Omega_d(f.shape[0])
        f = T_space(-h/2)(f)
        f = unmirror0(f)
        return f
    return midpoints(f) + endpoints(f)

def laplacian(g):
    """
        Laplacian Operator
    """
    D, DD = g.derivative()
    H0, H1, H0d, H1d = g.hodge_star()

    L = lambda x: H1d(DD(H1(D(x))))
    Ld = lambda x: H1(D(H1d(DD(x))))

    return L, Ld

def Grid_1D_Chebyshev(n, xmin=-1, xmax=+1):

    N = n
    dimension = 1

    # 2n-1 points: n primal, n-1 dual
    x = sin(linspace(-pi/2, pi/2, 2*n-1))
    p = 0.5*(xmin*(1-x) + xmax*(1+x))

    verts = p[::2]
    delta = diff(verts)
    edges = (verts[:-1], verts[1:])

    verts_dual = p[1::2]
    tmp = concatenate(([p[0]], verts_dual, [p[-1]]))
    delta_dual = diff(tmp)
    edges_dual = (tmp[:-1], tmp[1:])

    V = verts
    S0 = arange(len(V))
    S1 = (S0[:-1], S0[1:])

    Vd = verts_dual
    S0d = arange(len(Vd))
    S1d = (S0d[:-1], S0d[1:])

    P0 = lambda f: f(verts)
    P1 = lambda f: integrate_chebyshev(verts, f)
    P0d = lambda f: f(verts_dual)
    P1d = lambda f: integrate_chebyshev_dual(
            concatenate(([-1], verts_dual, [+1])), f)
    def projection():
        return P0, P1, P0d, P1d

    B0  = [lambda x, i=i:  psi0(n, i, x) for i in range(n)]
    B1  = [lambda x, i=i:  psi1(n, i, x) for i in range(n-1)]
    B0d = [lambda x, i=i: psid0(n, i, x) for i in range(n-1)]
    B1d = [lambda x, i=i: psid1(n, i, x) for i in range(n)]
    def basis_fn():
        return B0, B1, B0d, B1d

    R0, R1, R0d, R1d = reconstruction(basis_fn())

    def boundary_condition(f):
        bc = zeros(array(B1d).shape)
        bc[ 0] = -f(xmin)
        bc[-1] = +f(xmax)
        return bc

    D0 = lambda f: diff(f)
    D0d = lambda f: diff(concatenate(([0], f, [0])))
    def derivative():
        return D0, D0d

    H0 = H0_cheb
    H1 = H1_cheb
    H0d = H0d_cheb
    H1d = H1d_cheb
    def hodge_star():
        return H0, H1, H0d, H1d
    H0, H1, H0d, H1d = hodge_star()

    return bunch(**locals())
