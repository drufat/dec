from dec.spectral import *

#########
# Methods using FFT
#########

def H_exp(a):
    r'''
    :math:`\int_{x-h/2}^{x+h/2}f(\xi) \exp(i \xi) d\xi`

    This transformation is singular non-invertible.
    '''

    N = len(a)
    h = 2*pi/N
    c = I_diag(N+2, -h/2, h/2)

    a = F(a)
    b = roll(c[2:]*a,+1)
    return Finv(b)

def H_sin(a):
    r'''
    :math:`\int_{x-h/2}^{x+h/2}f(\xi) \sin(\xi) d\xi`

    This transformation is singular non-invertible.

    >>> x = linspace(0, 2*pi, 6)[:-1]
    >>> round_(real( H_sin(sin(2*x)) ), 3)
    array([ 0.271,  0.438, -0.573, -0.573,  0.438])
    '''

    N = len(a)
    h = 2*pi/N

    a = F(a)
    c = I_diag(N+2, -h/2, h/2)
    b = (roll(c[2:]*a,+1) - roll(c[:-2]*a,-1))/2j

    return Finv(b)

def H_sin_dual(a):
    r'''
    :math:`\int_{x-h/2}^{x+h/2}f(\xi) \sin(\xi+\frac{h}{2}) d\xi`

    This transformation is singular non-invertible.

    >>> x = linspace(0, 2*pi, 6)[:-1]
    >>> round_(real( H_sin_dual(sin(2*x)) ), 3)
    array([ 0.219,  0.573, -0.084, -0.844,  0.135])
    '''

    N = len(a)
    h = 2*pi/N

    a = F(a)
    c = I_diag(N+2, -h/2, h/2)
    s = exp(1j*h/2)
    b = (roll(c[2:]*a,+1)*s - roll(c[:-2]*a,-1)/s)/2j

    return Finv(b)

def I_sin(a, b, v):

    v = F(v)
    c = I_diag(v.shape[0], a, b)
    v = (roll(c*v,+1) - roll(c*v,-1))/2j

    return Finv(v)

### Chebyshev hodge-stars

def H0d_cheb(f):
    r'''

    .. math::
        \widetilde{\mathbf{H}}^0 = 
            {\mathbf{M}_1}^{\dagger}
             \mathbf{I}^{-\frac{h}{2}, \frac{h}{2}}
             \widetilde{\mathbf{\Omega}} 
             \mathbf{M}_1^+
    '''

    f = mirror1(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = f*Omega_d(N)
    f = I_space(-h/2, h/2)(f)
    f = unmirror1(f)

    return real(f)

def H1_cheb(f):
    r'''

    .. math::
        \mathbf{H}^1 =
            {\mathbf{M}_1}^{\dagger}
            \widetilde{\mathbf{\Omega}}^{-1}
            {\mathbf{I}^{-\frac{h}{2}, \frac{h}{2}}}^{-1}
            \mathbf{M}_1^-
    '''

    f = mirror1(f, -1)
    N = f.shape[0]; h = 2*pi/N
    f = I_space_inv(-h/2, h/2)(f)
    f = f/Omega_d(N)
    f = unmirror1(f)

    return real(f)

def H0_cheb_alternate(f):
    r'''

    .. math::
        \mathbf{H}^0 = \mathbf{M}_0^{\dagger}
            (\mathcal{A}^{0} \mathbf{E}^{-1} \mathbf{I}^{-\frac{h}{2}, 0} +
             \mathcal{A}^{N-1} \mathbf{E}^{-1} \mathbf{I}^{0, +\frac{h}{2}})
             \mathbf{\Omega} \mathbf{E}^{1}
             \mathbf{M}_0^{+}
    '''
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

def H0_cheb(f):
    r'''
    .. math::

        \mathbf{H}^0 &=& 
                \mathbf{A}
                \mathbf{M}_0^{\dagger}
                \mathbf{I}^{0, +\frac{h}{2}}
                \mathbf{\Omega} 
                \mathbf{E}^{N-1}
                \mathbf{M}_0^{+}
    '''
    N = f.shape[0]; h = pi/(N-1)
    f = mirror0(f, +1)
    f = E_space(N-1)(f)
    f = f*Omega(f.shape[0])
    f = I_space(0, h/2)(f)
    f = unmirror1(f)
    f = matA(f)

    return real(f)

def H1d_cheb(f):
    r'''

    .. math::
        
        \widetilde{\mathbf{H}}^{1} = \mathbf{M}_{0}^{\dagger}
                                     \left(\mathbf{T^{-\frac{h}{2}}}\mathbf{\Omega}^{-1}\mathbf{T}^{\frac{h}{2}}-
                                                                    \mathbf{B}\mathbf{I}^{0,\frac{h}{2}}-
                                                                    \mathbf{B}^{\dagger}\mathbf{I}^{-\frac{h}{2},0}\right)
                                     \mathbf{I}^{-\frac{h}{2},\frac{h}{2}}{}^{-1}\mathbf{M}_{0}^{-}\mathbf{C}+
                                     \mathbf{B}+\mathbf{B}^{\dagger}

    '''
    N = f.shape[0]; h = pi/(N-1)
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
