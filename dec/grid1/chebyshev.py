from dec.spectral import *

class Grid_1D_Chebyshev:
    
    def __init__(self, n, xmin=-1, xmax=+1):

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

        self.dimension = dimension
        self.n = n
        self.xmin = xmin
        self.xmax = xmax
        self.delta = delta
        self.delta_dual = delta_dual
        self.pnts = p
        self.verts = verts
        self.verts_dual = verts_dual
        self.edges = edges
        self.edges_dual = edges_dual

    def projection(self):
        P0 = lambda f: f(self.verts)
        #P1 = lambda f: slow_integration(self.edges[0], self.edges[1], f)
        P1 = lambda f: integrate_chebyshev(self.verts, f)
        P0d = lambda f: f(self.verts_dual)
        #P1d = lambda f: slow_integration(self.edges_dual[0], self.edges_dual[1], f)
        P1d = lambda f: integrate_chebyshev_dual(
                concatenate(([-1], self.verts_dual, [+1])), f)
        return P0, P1, P0d, P1d

    def basis_fn(self):
        n = self.n
        B0  = [lambda x, i=i:  psi0(n, i, x) for i in range(n)]
        B1  = [lambda x, i=i:  psi1(n, i, x) for i in range(n-1)]
        B0d = [lambda x, i=i: psid0(n, i, x) for i in range(n-1)]
        B1d = [lambda x, i=i: psid1(n, i, x) for i in range(n)]
        return B0, B1, B0d, B1d

    def reconstruction(self):
        R0, R1, R0d, R1d = reconstruction(self.basis_fn())
        return R0, R1, R0d, R1d

    def boundary_condition(self, f):
        bc = zeros((self.n, ))
        bc[ 0] = -f(self.xmin)
        bc[-1] = +f(self.xmax)
        return bc

    def derivative(self):
        D0 = lambda f: diff(f)
        D0d = lambda f: diff(concatenate(([0], f, [0])))
        return D0, D0d

    def hodge_star(self):
        H0 = H0_cheb
        H1 = H1_cheb
        H0d = H0d_cheb
        H1d = H1d_cheb
        return H0, H1, H0d, H1d

    def switch(self):
        return S_cheb, S_cheb_pinv

    def wedge(self):
        
        def w00(a, b):
            return a*b
        
        def w01(a, b):
            raise NotImplemented
        
        return w00, w01

    def contraction(self, V):
        return contraction1(self, V)

def H0d_cheb(f):
    f = mirror1(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = F(f)
    f = fourier_S(f, -h/2)
    f = fourier_K(f, 0, h)
    f = Finv(f)
    f = unmirror1(f)
    return real(f)

def H1_cheb(f):
    f = mirror1(f, -1)
    N = f.shape[0]; h = 2*pi/N
    f = F(f)
    f = fourier_K_inv(f, 0, h)
    f = fourier_S(f, +h/2)
    f = Finv(f)
    f = unmirror1(f)
    return real(f)

def H0_cheb(f):
    '''
    
    >>> to_matrix(H0_cheb, 2)
    array([[ 0.75,  0.25],
           [ 0.25,  0.75]])
       
    '''
    f = mirror0(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = F(f)
    f = fourier_K(f, 0, h/2)
    f = Finv(f)
    f = fold0(f, -1)
    return real(f)

def H1d_cheb_new(f):
     
#     f=f.copy()
#     aa, bb = f[0], f[-1]
#     f[0], f[-1] = 0, 0
#     f = mirror0(f, -1)
#     N = f.shape[0]; h = 2*pi/N
#     f = F(f)
#     f = fourier_K_inv(f, -h/2, h/2)
#     f = Finv(f)
#     f = unmirror0(f)
#     return real(f)
    
    f = unfold0(f)
    N = f.shape[0]; h = 2*pi/N
    f = F(f)
    f = fourier_K_inv(f, 0, h/2)
    f = Finv(f)
    f = unmirror0(f)
    return real(f)

def H1d_cheb(f):
    '''
    
    >>> to_matrix(H1d_cheb, 2)
    array([[ 1.5, -0.5],
           [-0.5,  1.5]])
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
    
def S_cheb(f):
    '''
    Shift from primal to dual 0-forms.
    >>> S_cheb(array([-1, 0, 1]))
    array([-0.70710678,  0.70710678])
    >>> S_cheb(array([1, 0, -1]))
    array([ 0.70710678, -0.70710678])
    '''
    f = mirror0(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = Finv(F(f)*S_diag(N, h/2))
    f = unmirror1(f)
    return real(f)

def S_cheb_pinv(f):
    '''
    This is a pseudo inverse of S_cheb.     
    >>> allclose(  S_cheb_pinv(array([ -1/sqrt(2),  1/sqrt(2)])), 
    ...            array([ -1, 0, 1]))
    True
    >>> allclose(  S_cheb_pinv(array([ 1/sqrt(2),  -1/sqrt(2)])), 
    ...            array([ 1, 0, -1]))
    True
    '''
    f = mirror1(f, +1)
    N = f.shape[0]; h = 2*pi/N
    f = Finv(F(f)*S_diag(N, -h/2))
    f = unmirror0(f)
    return real(f)
