'''
Spectral DEC
=============
'''

from numpy import *
from numpy.testing import assert_almost_equal
import operator
from functools import reduce
try:
    from scipy.linalg.special_matrices import toeplitz
except ImportError:
    from scipy.linalg.basic import toeplitz
from dec.helper import *
from dec.forms import *

np.seterr(divide='ignore', invalid='ignore')

def alpha0_alt(N, x):
    '''
    >>> def a0(N, i): return round(alpha0_alt(N, i*2*pi/N), 15)
    >>> assert (a0(5, 0), a0(5, 1), a0(5, 2)) == (1.0, 0.0, 0.0)
    >>> assert (a0(6, 0), a0(6, 1), a0(6, 2)) == (1.0, 0.0, 0.0)
    '''
    y = sin(N*x/2)*(cos(x/2)*(1-(-1)**N) + 
                             (1+(-1)**N))/sin(x/2)/N/2

    if hasattr(y, '__setitem__'):
        y[x==0] = 1
    elif x==0:
        y = 1

    return y

def alpha0(N, x):
    r'''

    .. math::
        \alpha_{N}(x)=\frac{1}{N}
        \begin{cases}
            \cot\frac{x}{2}\,\sin\frac{Nx}{2} & \text{if N even,}\\
            \csc\frac{x}{2}\,\sin\frac{Nx}{2} & \text{if N odd.}
        \end{cases}


    >>> def a0(N, i): return round(alpha0(N, i*2*pi/N), 15)
    >>> assert (a0(5, 0), a0(5, 1), a0(5, 2)) == (1.0, 0.0, 0.0)
    >>> assert (a0(6, 0), a0(6, 1), a0(6, 2)) == (1.0, 0.0, 0.0)

    '''
    if N % 2 == 0:
        y = (sin(N*x/2) / tan(x/2)) / N
    else:
        y = (sin(N*x/2) / sin(x/2)) / N

    if hasattr(y, '__setitem__'):
        y[x==0] = 1
    elif x==0:
        y = 1

    return y

def beta0(N, x):
    r'''

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
    '''
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
    r'''

    .. math::
        \alpha_{N, n} (x) = \alpha_N(x-h n)
    '''
    return alpha0(N, x-2*pi/N*n)

def beta(N, n, x):
    r'''

    .. math::
        \beta_{N, n} (x) = \beta_N(x-h n)
    '''
    return beta0(N, x-2*pi/N*n)

def gamma(N, k):
    r'''

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

    '''
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
    r'''
    Basis functions for primal 0-forms.

    .. math::
        \phi_{N,n}^{0}(x)=\alpha_{N,n}(x)
    '''
    return alpha(N, n, x)

def phi1(N, n, x):
    r'''
    Basis functions for primal 1-forms.

    .. math::
        \phi_{N,n}^{1}(x)=\beta_{N,n+\frac{1}{2}}(x)
    '''
    return beta(N, n + 0.5, x)

def phid0(N, n, x):
    r'''
    Basis functions for dual 0-forms.

    .. math::
        \widetilde{\phi}_{N,n}^{0}(x)=\alpha_{N,n+\frac{1}{2}}(x)
    '''
    return alpha(N, n + 0.5, x)

def phid1(N, n, x):
    r'''
    Basis functions for dual 1-forms.

    .. math::
        \widetilde{\phi}_{N,n}^{1}(x)=\alpha_{N,n}(x)
    '''
    return beta(N, n, x)

########################################
# Mapping between semi-circle and line #
########################################

def varphi(x):
    r'''
    .. math::
        \varphi:&& [-1,1]\to[0,\pi]\\
                && x\mapsto\arccos(-x)

    >>> varphi(-1) == 0
    True
    >>> varphi(+1) == pi
    True
    '''
    return arccos(-x)

def varphi_inv(x):
    r'''
    .. math::
        \varphi^{-1}:&& [0,\pi]\to[-1,1]\\
                && \theta\mapsto-\cos(\theta)

    >>> varphi_inv(0) == -1
    True
    >>> varphi_inv(pi) == +1
    True
    '''
    return -cos(x)

############################################################
# Regular Basis Functions
############################################################

def delta(N, x):
    r'''
    Basis function for dual half-edge at the boundary.

    .. math::
        \delta_{N}(x)=\left((N-1)^{2}\alpha_{2N-2,0}(x)+\frac{1}{2}\cos\left((N-1)x\right)\right)\sin(x)
    '''

    return ((N-1)**2 * alpha(2*N-2, 0, x) + 0.5*cos((N-1)*x))*sin(x)

def rho(N, n, x):
    r'''

    .. math::
        \rho_{N, n}(x) = 2 \gamma_{2N-2, n}\delta_N(x) + 2 \gamma_{2N-2, N-n-1} \delta_N(\pi-x)
    '''
    return 2*gamma(2*N-2, n)*delta(N, x) + 2*gamma(2*N-2, N-n-1)*delta(N, pi-x)

def kappa0(N, n, x):
    r'''
    Basis functions for primal 0-forms.

    .. math::
        \kappa_{N,n}^{0}(x) =
        \begin{cases}
            \alpha_{2N-2,n}(x), & n\in\{0,N-1\}\\
            \alpha_{2N-2,n}(x)+\alpha_{2N-2,n}(2\pi-x), & n\in\{1,\dots,N-2\}
        \end{cases}
   '''
    if n in (0, N-1):
        return  alpha(2*N-2, n, x)
    else:
        return (alpha(2*N-2, n, x) +
                alpha(2*N-2, n, 2*pi - x))

def kappad0(N, n, x):
    r'''

    .. math::
        \widetilde{\kappa}_{N,n}^{0}(x)=
        \alpha_{2N-2,\, n+\frac{1}{2}}(x) +
        \alpha_{2N-2,\, n+\frac{1}{2}}(2\pi-x),\quad n\in\{0,\dots,N-2\}
    '''
    return (alpha(2*N-2, n+0.5, x) +
            alpha(2*N-2, n+0.5, 2*pi - x))

def kappa1(N, n, x):
    r'''

    .. math::
        \kappa_{N,n}^{1}(x) = \left(
        \beta_{2N-2,n+\frac{1}{2}}(x)-
        \beta_{2N-2,n+\frac{1}{2}}(2\pi-x)\right)
        \mathbf{d}x,n\in\{0,\dots,N-2\}
    '''
    return (beta(2*N-2, n+0.5, x) -
            beta(2*N-2, n+0.5, 2*pi - x))

def kappa1_symm(N, n, x):
    r'''

    .. math::
        \kappa_{N,n}^{S1}(x) = \left(
        \beta_{2N-2,n+\frac{1}{2}}(x)+
        \beta_{2N-2,n+\frac{1}{2}}(2\pi-x)\right)
        \mathbf{d}x,n\in\{0,\dots,N-2\}
    '''
    return (beta(2*N-2, n+0.5, x) + 
            beta(2*N-2, n+0.5, 2*pi - x))

def kappad1(N, n, x):
    r'''

    .. math::
        \widetilde{\kappa}_{N,n}^{1}(x)=
        \begin{cases}
            \delta(x)\mathbf{d}x & \qquad n=0\\
            \left(\beta_{2N-2,n}(x)-
                  \beta_{2N-2,n}(2\pi-x)-\rho_{N,n}(x)\right)\mathbf{d}x & \qquad n\in\{1,\dots,N-2\}\\
            \delta(\pi-x)\mathbf{d}x & \qquad n=N-1
        \end{cases}
    '''
    if n == 0:
        return delta(N, x)
    if n == N-1:
        return delta(N, pi - x)

    y = (beta(2*N-2, n, x) -
         beta(2*N-2, n, 2*pi - x)
         - rho(N, n, x) )
    return y

def kappad1_symm(N, n, x):
    r'''

    .. math::
        \widetilde{\kappa}_{N,n}^{S1}(x) = \left(
            \beta_{2N-2,n}(x)+
            \beta_{2N-2,n}(2\pi-x) \right)
        \mathbf{d}x,n\in\{0,\dots,N-2\}
    '''
    return (beta(2*N-2, n, x) + beta(2*N-2, n, 2*pi - x))

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
    r'''
    Basis functions for primal 0-forms.

    .. math::
        \psi_{N,n}^{0}(x)=\kappa_{N,n}^{0}(\arccos(-x))
    '''
    return kappa0(N, n, arccos(-x))

def psi1(N, n, x):
    r'''
    Basis functions for primal 1-forms.

    .. math::
        \psi_{N,n}^{1}(x)\mathbf{d}x=
            \kappa_{N,n}^{1}(\arccos(-x))\frac{\mathbf{d}x}{\sqrt{1-x^{2}}}
    '''
    #x = __fix_singularity_at_boundary(x)
    return kappa1(N, n, arccos(-x))/sqrt(1 - x**2)

def psid0(N, n, x):
    r'''
    Basis functions for dual 0-forms.

    .. math::
        \tilde{\psi}_{N,n}^{0}(x)=\tilde{\kappa}_{N,n}^{0}(\arccos(-x))
    '''
    return kappad0(N, n, arccos(-x))

def psid1(N, n, x):
    r'''
    Basis functions for dual 1-forms.

    .. math::
        \tilde{\psi}_{N,n}^{1}(x)\mathbf{d}x=\tilde{\kappa}_{N,n}^{1}(\arccos(-x))\frac{\mathbf{d}x}{\sqrt{1-x^{2}}}
    '''
    x = __fix_singularity_at_boundary(x)
    return kappad1(N, n, arccos(-x))/sqrt(1 - x**2)

#########################
#  Lagrange Polynomials
#########################

def lagrange_polynomials(x):
    r'''
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

    '''
    N = len(x)
    L = [None]*N
    for i in range(N):
        def Li(y, i=i):
            gen = ((y-x[j])/(x[i]-x[j]) for j in set(range(N)).difference([i]))
            return reduce(operator.mul, gen)
        L[i] = (Li)
    return L

def freq(N):
    '''
    >>> freq(5)
    array([-2., -1.,  0.,  1.,  2.])
    >>> freq(6)
    array([-3., -2., -1.,  0.,  1.,  2.])
    '''
    return fft.fftshift(fft.fftfreq(N)*N)

F    = lambda x: fft.fftshift(fft.fft(x))
Finv = lambda x: fft.ifft(fft.ifftshift(x))

def method_in_fourier_space(g):
    def f(x, *args, **keywords):
        return Finv(g(F(x), *args, **keywords))
    f.__doc__ = g.__doc__ # make sure doctests are run
    return f

def H(a):
    r'''

    .. math::
        \mathbf{H}^0 = \widetilde{\mathbf{H}}^0 =
        \mathcal{F}^{-1} \mathbf{I}^{-\frac{h}{2}, \frac{h}{2}} \mathcal{F}
    '''
    N = len(a)
    h = 2*pi/N
    return Finv( F(a) * I_diag(N, -h/2, h/2) ).real

def Hinv(a):
    r'''

    .. math::
        \mathbf{H}^1 = \widetilde{\mathbf{H}}^1 =
        \mathcal{F}^{-1}{\mathbf{I}^{-\frac{h}{2},\frac{h}{2}}}^{-1}\mathcal{F}
    '''
    N = len(a)
    h = 2*pi/N
    return Finv( F(a) / I_diag(N, -h/2, h/2) ).real

def I(a, x0, x1):
    N = len(a)
    return Finv ( I_diag(N, x0, x1) * F(a) )
 
def Iinv(a, x0, x1):
    N = len(a)
    return Finv ( F(a) / I_diag(N, x0, x1) )
 
def S(a):
    N = len(a)
    h = 2*pi/N
    return Finv( F(a) * S_diag(N, h/2) ).real
 
def Sinv(a):
    N = len(a)
    h = 2*pi/N
    return Finv( F(a) * S_diag(N, -h/2) ).real

def I_diag(N, a, b):
    r'''

    The diagonal that corresponds to integration in Fourier space.
    Corresponds to :math:`f(x) \mapsto \int_{x+a}^{x+b} f(\xi) d\xi`

    .. math::
        \mathbf{I}_{\phantom{a,b}nn}^{a,b}
            =\frac{e^{inb}-e^{ina}}{in}

    .. math::
        \mathbf{I}_{\phantom{a,b}00}^{a,b}=b-a
    '''
    n = freq(N)
    y = (exp(1j*n*b) - exp(1j*n*a))/(1j*n)
    y[n==0] = b - a
    return y

def S_diag(N, a):
    r'''
    The diagonal that corresponds to shifting in Fourier Space
    Corresponds to :math:`f(x) \mapsto f(x-h)`

    .. math::
        \mathbf{S}_{\phantom{a}nn}^{a}=e^{ina}
    '''
    n = freq(N)
    return exp(1j*n*a)

def fourier_I(x, a, b):
    r'''
    Corresponds to :math:`f(x) \mapsto \int_{x+a}^{x+b} f(\xi) d\xi`
    '''
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

def fourier_S(x, a):
    n = freq(x.shape[0])
    return x*exp(1j*n*a)

def fourier_S_inv(x, a):
    n = freq(x.shape[0])
    return x*exp(-1j*n*a)

def fourier_T(x, h):
    r'''
    Corresponds to :math:`f(x) \mapsto f(x+a)`
    '''
    N = x.shape[0]
    n = freq(N)
    return x*exp(1j*n*h)

def fourier_K(x, a, b):
    r'''
    Corresponds to :math:`f(x) \mapsto \int_{x+a}^{x+b} f(\xi) \sin(\xi) d\xi`
    '''
    x = array(x, dtype=complex)
    N = x.shape[0]
    
    x = hstack([[0], x, [0]])
    x = (roll(x,+1) - roll(x,-1))
    x *= I_diag(N+2, a, b)/2j
    rslt = x[1:-1]

    rslt[ 0] += x[-1]
    rslt[-1] += x[0]
    return rslt

def fourier_K_inv(x, a, b):
    # Make sure type is coerced to complex, otherwise numpy ignores the complex parts
    # and reverts to reals.
    x = array(x.copy(), dtype=complex)
    N = x.shape[0]
    I = I_diag(N+2, a, b)
    x /= I[1:-1]
    
    if (isclose(I[0], I[N]) or 
        isclose(I[1], I[N+1]) or 
        isclose(I[0]*I[1], I[N]*I[N+1])):
        raise ValueError("Singular operator.")

    y = zeros(N, dtype=complex)
    # The computations below are essentially Schur's complement?
    E = sum(x[::2]); O = sum(x[1::2])
    if N % 2 == 0:
        y[0]  = O/(1-I[0]/I[N])
        y[-1] = E/(I[N+1]/I[1]-1)
    else:
        y[0]  = (I[1]/I[N+1]*E+O)/(1-I[1]*I[0]/I[N]/I[N+1])
        y[-1] = (I[N]/I[0]*E+O)/(I[N]*I[N+1]/I[0]/I[1]-1)
    x[0]  -= y[-1]*I[N+1]/I[1]
    x[-1] -= -y[0]*I[0]/I[N]
    
    x = hstack([[-y[0]], x , [y[-1]]])
    y[::2] = -cumsum(x[::2])[:-1]
    y[1::2] = cumsum(x[1::2][::-1])[:-1][::-1]
    
    y *= 2j
    
    return y

def refine(x):
    '''
    Resample x at a twice refined grid.
    >>> N = 4
    >>> x = linspace(0, 2*pi, N+1)[:-1]
    >>> y = linspace(0, 2*pi, 2*N+1)[:-1]
    >>> approx(refine(cos(x)), cos(y))
    True
    '''
    x = interweave(x, S(x))
    return x

def subdivide(x):
    '''
    Subdivide x like below. Assume points are equidistant.
    *   *   *   *   *   *
    * * * * * * * * * * *
    '''
    if len(x) < 2: return x
    assert is_equidistant(x)
    y = interweave(x[:-1], x[:-1] + 0.5*diff(x))
    return concatenate((y, [x[-1]]))

def integrate_spectral_coarse(x, f):
    '''
    >>> integrate_spectral_coarse(linspace(-pi, pi, 3), sin)
    array([-2.,  2.])
    '''
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
    '''
    >>> integrate_chebyshev(array([cos(pi), cos(.5*pi), cos(0)]), lambda x: x)
    array([-0.5,  0.5])
    '''
    assert approx(2, xi[-1] - xi[0]), xi

    F = lambda theta: f(-cos(theta))*sin(theta)
    x = varphi(xi)

    assert(is_equidistant(x))

    # complete the circle from the other side
    x = concatenate((x, x[1:]+pi))
    return integrate_spectral(x, F)[:len(xi)-1]

def integrate_chebyshev_dual(xi, f):
    '''
    Integrate points that may include the two half-edges at the boundary.
    #>>> integrate_chebyshev_dual(array([cos(pi), cos(0.75*pi), cos(0.25*pi), cos(0)]), lambda x: x)
    #array([-0.25,  0.  ,  0.25])
    '''
    x = varphi(xi)
    z = varphi_inv(concatenate(([x[0]], subdivide(x[1:-1]), [x[-1]])))
    i, j = concatenate(( [0], integrate_chebyshev(z, f), [0] )).reshape(-1, 2).T
    return i+j

def split_args(I):
    '''
    Convert integration function from I(x, f) to I(x0, x1, f) form
    '''
    def J(x0, x1, f, I=I):
        assert_almost_equal(x0[1:], x1[:-1])
        return I(concatenate((x0, [x1[-1]])), f)
    return J

##############################
# Discrete exterior calculus
##############################

def hodge_star_matrix(P, B):
    '''
    Compute Hodge-Star matrix from basis functions.
    '''
    P0, P1, P0d, P1d = P
    B0, B1, B0d, B1d = B
    H0 = vstack(P1d(b) for b in B0).T
    H1 = vstack(P0d(b) for b in B1).T
    H0d = vstack(P1(b) for b in B0d).T
    H1d = vstack(P0(b) for b in B1d).T
    return H0, H1, H0d, H1d

def reconstruction(basis_fn):
    '''
    Give the reconstruction functions for the set of basis functions basis_fn.
    '''
    def R(a, B):
        def r(*x):
            return sum(a[i]*f(*x) for i, f in enumerate(B))
        return r
    return [(lambda a, B=B: R(a, B)) for B in basis_fn]

def A_diag(N):
    r'''
    
    .. math::
        \mathbf{A}=\text{diag}\left(\begin{array}{ccccccc}\frac{1}{2} & 1 & 1 & \dots & 1 & 1 & \frac{1}{2}\end{array}\right)
        
    >>> A_diag(2)
    array([ 0.5,  0.5])
    >>> A_diag(3)
    array([ 0.5,  1. ,  0.5])

    '''
    assert N > 1
    d = concatenate(([0.5], ones(N-2), [0.5]))
    return d

def extend(f, n):
    r'''

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
        
    >>> extend(array([ 1.,  2.,  3.,  4.]), 2)
    array([ 0.,  0.,  2.,  4.,  6.,  8.,  0.,  0.])
    
    '''
    N = f.shape[0]
    return (N+2.0*n)/N*_extend(f, n)

def unextend(f, n):
    r'''

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

    >>> unextend(array([ 0.,  0.,  2.,  4.,  6.,  8.,  0.,  0.]), 2)
    array([ 1.,  2.,  3.,  4.])

    '''
    N = f.shape[0]
    return (N-2.0*n)/N*_unextend(f, n)

def _extend(f, n):
    '''
    >>> _extend([1, 2, 3, 4], 2)
    array([ 0.,  0.,  1.,  2.,  3.,  4.,  0.,  0.])
    '''
    return concatenate(( zeros(n),
                         f,
                         zeros(n) ))

def _unextend(f, n):
    '''
    >>> _unextend(array([0,0,1,2,3,4,0,0]), 2)
    array([1, 2, 3, 4])
    >>> _unextend(array([1,2,0,0,3,4]), 2)
    array([4, 6])
    >>> _unextend(array([0,1,2,3,4,5,6,7]), 2)
    array([ 8, 10,  4,  6])
    '''
    assert 2*n <= f.shape[0]
    x = f[n:-n].copy()
    x[:n]  += f[-n:]
    x[-n:] += f[:n]
    return x

def mirror0(f, sign=+1):
    r'''

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
   '''
    return concatenate((f, sign*f[::-1][1:-1]))

def mirror1(f, sign=+1):
    r'''

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
    '''
    return concatenate((f, sign*f[::-1]))

def unmirror0(f):
    r'''

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
    '''
    return f[:(len(f)/2+1)]

def unmirror1(f):
    r'''

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
    '''
    return f[:(len(f)/2)]

def Omega(N):
    r'''

    .. math::
        \mathbf{\Omega}_{nn}=\sin\left(nh\right)

    '''

    h = 2*pi/N
    o = sin( arange(N)*h )
    return o

def Omega_d(N):
    r'''

    .. math::
        \mathbf{\widetilde{\Omega}}_{nn}
            =\sin\left(\left(n+\frac{1}{2}\right)h\right)

    If :math:`\widetilde{\omega}` is the length of each dual edge, then

    .. math::
        \mathbf{\widetilde{\Omega}}
            =\text{diag}(\mathbf{M}_{1}^{\dagger}
            \mathcal{F}^{-1}{\mathbf{I}^{-\frac{h}{2},\frac{h}{2}}}^{-1}
            \mathcal{F}\mathbf{M}_{1}^{-}\widetilde{\omega})
    '''

    h = 2*pi/N
    o = sin( (arange(N)+0.5)*h )
    return o

def half_edge_base(N):
    r'''
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
    '''
    a = 0.5*ones(N)
    a[1::2] += -1
    a[0] += (N-1)**2
    return a

def half_edge_integrals(f):
    '''
    >>> approx( half_edge_integrals([1,0,0]), gamma(3, 0))
    True
    >>> approx( half_edge_integrals([0,1,0]), gamma(3, 1))
    True
    '''
    N = len(f)
    h = 2*pi/N
    f = Hinv(f)
    p = I(f, 0, h/2)
    return p[0]

def pick(f, n):
    r'''

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

    '''

    return f[n]*ones(f.shape[0])

def reverse(f):
    r'''

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
    '''
    return f[::-1]

def S_space(a):
    return lambda f: Finv(F(f)*S_diag(f.shape[0], a))

def S_space_inv(a):
    return lambda f: Finv(F(f)/S_diag(f.shape[0], a))

def I_space(a, b):
    return lambda f: Finv(F(f)*I_diag(f.shape[0], a, b))

def I_space_inv(a, b):
    return lambda f: Finv(F(f)/I_diag(f.shape[0], a, b))

def K_space(a, b):
    return lambda f: Finv(fourier_K(F(f), a, b))

def K_space_inv(a, b):
    return lambda f: Finv(fourier_K_inv(F(f), a, b))

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

def fold1(f, sgn=+1):
    '''
    
    >>> fold1(array([0, 1, 2, 3]), +1)
    array([3, 3])
    >>> fold1(array([0, 1, 2, 3]), -1)
    array([-3, -1])

    '''    
    return f[:f.shape[0]/2] + sgn*f[::-1][:f.shape[0]/2]


def fold0(f, sgn=+1):
    '''
    
    >>> fold0(array([0, 1, 2, 3]), +1)
    array([0, 4, 2])
    >>> fold0(array([0, 1, 2, 3]), -1)
    array([ 0, -2, -2])

    '''    
    return (    hstack([f[:f.shape[0]/2], [0]]) + 
            sgn*hstack([[0], f[::-1][:f.shape[0]/2]]) )

def unfold0(f):
    '''
    
    >>> unfold0(array([ 1, 1, 1]))
    array([ 1. ,  0.5, -1. , -0.5])
    '''
    return hstack([ [f[0]], .5*f[1:-1], [-f[-1]], -.5*f[1:-1][::-1] ])

def laplacian(g):
    '''
        Laplacian Operator
    '''
    D, DD = g.derivative()
    H0, H1, H0d, H1d = g.hodge_star()

    L = lambda x: H1d(DD(H1(D(x))))
    Ld = lambda x: H1(D(H1d(DD(x))))

    return L, Ld
