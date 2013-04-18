from pylab import *
from dec.spectral import *

def fix_axis_labels():
    tick_params(labelsize=45)
    xticks((-1, 0, 1), ('$-1$', '$0$', '$+1$'))
    yticks((0, 1), ('$0$', '$1$'))
    axis([-1.5, 1.5, -0.5, 1.5])

def draw_stemlines(x, y):
    markerline, stemlines, baseline = stem(x, y, '--')
    setp(markerline, 'markerfacecolor', 'k')
    setp(baseline, 'color', 'k', 'linewidth', 0.5)
    setp(stemlines, 'color', 'k')

def _savefig(name):
    savefig("Documents/Research/Desbrun/latex/SpectralDEC/figs/"+name)

x = linspace(-1, +1, 1000)

M = 7
h = 2 * pi / (2 * M - 2)
n = arange(M)

pnts = cos(n * h)[::-1]

for m in range(4):
    figure()
    plot(x, psi0(M, m, x), 'k', linewidth=2)
    draw_stemlines(pnts, psi0(M, m, pnts))
    fix_axis_labels()
    #_savefig("cheb_a%d.pdf" % m)

for m in range(4):
    figure()
    f = lambda x: psi1(M, m, x) * abs(pnts[m + 1] - pnts[m])
    rect = 0 * x; rect[logical_and(pnts[m] < x, x < pnts[m + 1])] = 1;
    plot(x, f(x), 'k', linewidth=2)
    plot(x, rect, 'k--')
    fix_axis_labels();
    for n in range(M - 1):
        color = 'green' if n % 2 else 'red'
        cut = logical_and(pnts[n] < x, x < pnts[n + 1])
        fill_between(x[cut], f(x)[cut], rect[cut], color=color)
    #_savefig("cheb_b%d.pdf" % m)

show()
