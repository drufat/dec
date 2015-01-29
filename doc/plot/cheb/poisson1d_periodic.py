import dec
from dec.grid1 import *
from dec.spectral import *

def run():
    size = [int(x) for x in exp(linspace(log(3), log(50), 30))]

    f = lambda x: exp(sin(x))
    q = lambda x: f(x)*(cos(x)**2 - sin(x))

    L = [[], []]
    for N in size:

        g = Grid_1D_Periodic(N)
        laplace, laplace_d = laplacian(g)

        #################
        p = to_matrix(laplace, N)
        one = zeros(N); one[0] = 1
        p = vstack((one, p))

        z = dot(linalg.pinv(p), concatenate(( [f(g.verts[0])], g.P0(q) )) )

        err = linalg.norm(g.P0(f) - z, inf)
        L[0].append( err )

        #######################

        p = to_matrix(laplace_d, N)
        one = zeros(N); one[0] = 1
        p = vstack((one, p))

        z = dot(linalg.pinv(p), concatenate(( [f(g.verts_dual[0])], g.P0d(q) )) )

        err = linalg.norm(g.P0d(f) - z, inf)
        L[1].append( err )

    return size, L
dataname = "converge/poisson1d_periodic.json"
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

size, L = dec.get_data(dataname)

ax = plt.subplot(1, 1, 1)
ax.loglog(size, L[0], '-o', color='r', label='Primal')
ax.loglog(size, L[1], '-s', color='b', label='Dual')

ax.grid(True)
ax.legend(shadow=True, fancybox=True)

#plt.title(r'Periodic')
plt.xlabel(r'$N$', fontsize=fontsize)
plt.ylabel(r'$\parallel f - \Delta^{-1} q \parallel_\infty$', fontsize=fontsize)
fix_axis_labels(plt)

plt.draw()
#plt.savefig("Documents/Research/Desbrun/latex/SpectralDEC/plot/cheb/poisson1d_periodic.pdf")
plt.show()
