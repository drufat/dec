from dec.spectral import *
from dec.grid1 import *
import matplotlib.pyplot as plt

def fix_axis_labels():
    plt.tick_params(labelsize=45)
    plt.xticks((-1, 0, 1), ('$-1$', '$0$', '$+1$'))
    plt.yticks((0, 1), ('$0$', '$1$'))
    plt.axis([-1.5, 1.5, -0.5, 1.5])

def draw_stemlines(x, y):
    markerline, stemlines, baseline = plt.stem(x, y, '--')
    plt.setp(markerline, 'markerfacecolor', 'k')
    plt.setp(baseline, 'color', 'k', 'linewidth', 0.5)
    plt.setp(stemlines, 'color', 'k')

def _savefig(name):
    plt.savefig("Documents/Research/Desbrun/latex/SpectralDEC/figs/"+name)

x = linspace(-1, +1, 1000)

M = 7
h = 2 * pi / (2 * M - 2)
n = arange(M)

pnts = cos(n * h)[::-1]

for m in range(4):
    plt.figure()
    plt.plot(x, psi0(M, m, x), 'k', linewidth=2)
    draw_stemlines(pnts, psi0(M, m, pnts))
    fix_axis_labels()
    #_savefig("cheb_a%d.pdf" % m)

for m in range(4):
    plt.figure()
    f = lambda x: psi1(M, m, x) * abs(pnts[m + 1] - pnts[m])
    rect = 0 * x; rect[logical_and(pnts[m] < x, x < pnts[m + 1])] = 1;
    plt.plot(x, f(x), 'k', linewidth=2)
    plt.plot(x, rect, 'k--')
    fix_axis_labels();
    for n in range(M - 1):
        color = 'green' if n % 2 else 'red'
        cut = logical_and(pnts[n] < x, x < pnts[n + 1])
        plt.fill_between(x[cut], f(x)[cut], rect[cut], color=color)
    #_savefig("cheb_b%d.pdf" % m)

plt.show()
