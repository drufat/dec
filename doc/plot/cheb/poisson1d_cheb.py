import dec
from dec.spectral import *

def run():
    size = [int(x) for x in exp(linspace(log(3), log(50), 30)) ]

    #f = lambda x: x**3
    #q = lambda x: 6*x

    f = lambda x: exp(x)
    f_prime = lambda x: exp(x)
    q = lambda x: exp(x)

    #f = lambda x: x**2
    #f_prime = lambda x: 2*x
    #q = lambda x: 2 + 0*x

    L = [[], []]
    for N in size:

        g = Grid_1D_Chebyshev(N, -1, 1)
        laplace, laplace_d = laplacian(g)

        # Dirichlet boundary conditions
        # Solve: H1(D(H1d(DD(f) + bc))) == q

        bc = zeros(N); bc[0] = -f(-1); bc[-1] = f(+1)

        pinv = linalg.inv( to_matrix(laplace_d, N-1) )
        z = dot(pinv, g.P0d(q) - g.H1(g.D0(g.H1d(bc))) )

        err = linalg.norm(g.P0d(f) - z, inf)
        L[0].append( err )

        # Neumann boundary conditions
        # Solve: H1d(DD(  H1(D(x))) + bc) == q

        bc = zeros(N); bc[0] = -f_prime(-1); bc[-1] = f_prime(+1)

        # The matrix is not invertible, because it is defined up to a constant,
        # first derivative. Use linalg.pinv instead of linalg.inv
        p = to_matrix(laplace, N)
        one = zeros(N); one[0] = 1
        p = vstack((one, p))
        pinv = linalg.pinv(p)

        z = real( dot(pinv, concatenate( ([f(g.verts[0])], g.P0(q) - g.H1d(bc) ) ) ) )

        err = linalg.norm(g.P0(f) - z, inf)

        L[1].append( err )

    return size, L

dataname = "converge/poisson1d_cheb.json"
#dec.store_data(dataname, run())

import matplotlib
matplotlib.rcParams['legend.fancybox'] = True
import matplotlib.pyplot as plt

fontsize = 20
labelsize = 13.5

def fix_axis_labels(plt):
    plt.tick_params(labelsize=labelsize)
    plt.xticks((1, 10, 100), ('1', '10', '100'))
    plt.yticks((1, 1e-4, 1e-8, 1e-12),
               ('1e0', '1e-4', '1e-8', '1e-12'))
    plt.axis([.9, 111, 1e-15, 10])

ax = plt.subplot(1, 1, 1)
size, L = dec.get_data(dataname)
ax.loglog(size, L[0], '-o', color='r', label='Primal, Dirichlet bc')
ax.loglog(size, L[1], '-s', color='b', label='Dual, Neumann bc')
ax.grid(True)
ax.legend(shadow=True, fancybox=True)

#plt.title(r'Non-Periodic, Chebyshev')
plt.xlabel(r'$N$', fontsize=fontsize)
plt.ylabel(r'$\parallel f - \Delta^{-1} q \parallel_\infty$', fontsize=fontsize)
fix_axis_labels(plt)

#plt.savefig("Documents/Research/Desbrun/latex/SpectralDEC/plot/cheb/poisson1d_cheb.pdf")
plt.draw()
plt.show()
