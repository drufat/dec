import dec
from dec.grid2 import *

def run():
    size = [int(x) for x in exp(linspace(log(3), log(30), 30)) ]

    f = lambda x, y: exp(x) + exp(y)
    f_grad = lambda x, y: (exp(x), exp(y))
    q = lambda x, y: exp(x) + exp(y)

    f1 = lambda x, y: (exp(x), exp(y))
    f1_curl = lambda x, y: 0*x + 0*y
    f1_div = lambda x, y: exp(x)+exp(y)
    q1 = lambda x, y: (exp(x), exp(y))

    hs = lambda f: lambda x, y: (-f(x,y)[1], f(x,y)[0])

    L = [[], [], [], []]
    for N in size:
        g = Grid_2D_Chebyshev(N, N)
        L0, L1, L0d, L1d = laplacian2(g)
        D0, D1, D0d, D1d = g.derivative()
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        H0, H1, H2, H0d, H1d, H2d = g.hodge_star()
        add = lambda x, y: (x[0]+y[0], x[1]+y[1])
        BC0, BC1 = g.boundary_condition()

        z = L0(P0(f)) + H2d(BC1(hs(f_grad)))
        err0 = linalg.norm(P0(q) - z, inf)
        L[0].append(err0)

        z = L0d(P0d(f)) + H2(D1(H1d(BC0(f))))
        err1 = linalg.norm(P0d(q) - z, inf)
        L[1].append(err1)

        # curl, and normal
        z = add( L1(P1(f1)), add(H1d(BC0(f1_curl)), D0(H2d(BC1(hs(f1))))) )
        err2 = linalg.norm(flat(g.P1(q1)) - flat(z), inf)
        L[2].append(err2)

        # tangent, and divergence
        z = add( L1d(P1d(f1)), add(H1(D0(H2d(BC1(f1)))), BC0(f1_div)) )
        err3 = linalg.norm(flat(g.P1d(q1)) - flat(z), inf)
        L[3].append(err3)

    #    print N, err0, err1, err2, err3
    return size, L

dataname = "converge/poisson2d_cheb.json"
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
    plt.axis([.9, 111, 1e-13, 10])

plt.figure()
ax = plt.subplot(1, 1, 1)
size, L = dec.get_data(dataname)
ax.loglog(size, L[0], '-o', color='r', label='Primal')
ax.loglog(size, L[1], '-s', color='b', label='Dual')
ax.grid(True)
ax.legend(shadow=True, fancybox=True)
#plt.title(r'Chebyshev 0-forms')
plt.xlabel(r'$N$', fontsize=fontsize)
plt.ylabel(r'$\parallel \Delta f - q \parallel_\infty$', fontsize=fontsize)
fix_axis_labels(plt)
#plt.savefig("Documents/Research/Desbrun/latex/SpectralDEC/plot/cheb/poisson2d_cheb_00.pdf")

plt.figure()
ax = plt.subplot(1, 1, 1)
ax.loglog(size, L[2], '-o', color='r', label='Primal')
ax.loglog(size, L[3], '-s', color='b', label='Dual')
ax.grid(True)
ax.legend(shadow=True, fancybox=True)
#plt.title(r'Chebyshev 1-forms')
plt.xlabel(r'$N$', fontsize=fontsize)
plt.ylabel(r'$\parallel \Delta f - q \parallel_\infty$', fontsize=fontsize)
fix_axis_labels(plt)
#plt.savefig("Documents/Research/Desbrun/latex/SpectralDEC/plot/cheb/poisson2d_cheb_01.pdf")

plt.draw()
plt.show()
