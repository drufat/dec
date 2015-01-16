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
        P1d = lambda f: integrate_chebyshev_dual(
                concatenate(([-1], self.verts_dual, [+1])), f)
        #P1d = lambda f: slow_integration(self.edges_dual[0], self.edges_dual[1], f)
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

    def contraction(self, V):
        '''
        Implement contraction where V is the one-form corresponding to the vector field.
        '''
        S, Sinv = self.switch()
        H0, H1, H0d, H1d = self.hodge_star()

        def C1(f):                
            return Sinv(H1(V)) * Sinv(H1(f))
        
        return C1
    
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
