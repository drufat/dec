import dec
from dec.grid2 import *

def run():
    size = [int(x) for x in exp(linspace(log(3), log(50), 30))]

    a = lambda x: exp(sin(x))
    b = lambda x: cos(x)*exp(sin(x))
    c = lambda x: exp(sin(x))*(cos(x)**2 - sin(x))

    f = lambda x, y: a(x) + a(y)
    q = lambda x, y: c(x) + c(y)

    f1 = lambda x, y: (a(x), a(y))
    q1 = lambda x, y: (c(x), c(y))

    def add(x, y):
        return (x[0]+y[0], x[1]+y[1])

    L = [[], [], [], []]
    for N in size:
        g = Grid_2D_Periodic(N, N)
        L0, L1, L0d, L1d = laplacian2(g)

        z =  L0(g.P0(f))
        err0 = linalg.norm(g.P0(q) - z, inf)
        L[0].append(err0)

        z =  L0d(g.P0d(f))
        err1 = linalg.norm(g.P0d(q) - z, inf)
        L[1].append(err1)

        z =  L1(g.P1(f1))
        err2 = linalg.norm(flat(g.P1(q1)) - flat(z), inf)
        L[2].append(err2)

        z =  L1d(g.P1d(f1))
        err3 = linalg.norm(flat(g.P1d(q1)) - flat(z), inf)
        L[3].append(err3)

    #    print N, err0, err1, err2, err3
    return size, L

dataname = "converge/poisson2d_periodic.json"
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

plt.figure()
ax = plt.subplot(1, 1, 1)
size, L = dec.get_data(dataname)
ax.loglog(size, L[0], '-o', color='r', label='Primal')
ax.loglog(size, L[1], '-s', color='b', label='Dual')
ax.grid(True)
ax.legend(shadow=True, fancybox=True)

#plt.title(r'Periodic 0-forms')
plt.xlabel(r'$N$', fontsize=fontsize)
plt.ylabel(r'$\parallel \Delta f - q \parallel_\infty$', fontsize=fontsize)
fix_axis_labels(plt)
#plt.savefig("Documents/Research/Desbrun/latex/SpectralDEC/plot/cheb/poisson2d_periodic_00.pdf")

plt.figure()
ax = plt.subplot(1, 1, 1)

ax.loglog(size, L[2], '-s', color='r', label='Primal')
ax.loglog(size, L[3], '-s', color='b', label='Dual')
ax.grid(True)
ax.legend(shadow=True, fancybox=True)

#plt.title(r'Periodic 1-forms')
plt.xlabel(r'$N$', fontsize=fontsize)
plt.ylabel(r'$\parallel \Delta f - q \parallel_\infty$', fontsize=fontsize)
fix_axis_labels(plt)
#plt.savefig("Documents/Research/Desbrun/latex/SpectralDEC/plot/cheb/poisson2d_periodic_01.pdf")

plt.draw()
plt.show()
