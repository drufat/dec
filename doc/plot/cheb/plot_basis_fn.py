from pylab import *
from dec.grid1 import *

def fix_axis_labels():
    axis([-pi, 3 * pi, -0.5, 1.5])
    tick_params(labelsize=45)
    xticks((0, 2 * pi), ('$0$', '$2\pi$'))
    yticks((0, 1), ('$0$', '$1$'))

def draw_stemlines(x, y):
    markerline, stemlines, baseline = stem(x, y, '--')
    setp(markerline, 'markerfacecolor', 'k')
    setp(baseline, 'color', 'k', 'linewidth', 0.5)
    setp(stemlines, 'color', 'k')

def _savefig(name):
    savefig(name)

x = linspace(-pi, 3 * pi, 1000)

figure()
N = 6
h = 2 * pi / N
n = arange(-N, 2 * N)
plot(x, alpha0(6, x), 'k', linewidth=2)
draw_stemlines(n * h, alpha0(6, n * h))
fix_axis_labels()
#_savefig("6a.pdf")

figure()
plot(x, beta0(N, x) * h, 'k', linewidth=2)
rect = 0 * x; rect[logical_and(-h / 2 < x, x < h / 2)] = 1; rect[logical_and(2 * pi - h / 2 < x, x < 2 * pi + h / 2)] = 1
plot(x, rect, 'k--')
for n in arange(-N, 2 * N):
    color = 'green' if n % 2 else 'red'
    cut = logical_and(-h / 2 + n * h < x, x < h / 2 + n * h)
    fill_between(x[cut], beta0(N, x)[cut] * h, rect[cut], color=color)
fix_axis_labels()
#_savefig("6b.pdf")

figure()
N = 7
h = 2 * pi / N
n = arange(-N, 2 * N)
plot(x, alpha0(N, x), 'k', linewidth=2)
draw_stemlines(n * h, alpha0(N, n * h))
fix_axis_labels()
#_savefig("7a.pdf")

figure()
plot(x, beta0(7, x) * h, 'k', linewidth=2)
rect = 0 * x; rect[logical_and(-h / 2 < x, x < h / 2)] = 1; rect[logical_and(2 * pi - h / 2 < x, x < 2 * pi + h / 2)] = 1
plot(x, rect, 'k--')
for n in arange(-N, 2 * N):
    color = 'green' if n % 2 else 'red'
    cut = logical_and(-h / 2 + n * h < x, x < h / 2 + n * h)
    fill_between(x[cut], beta0(N, x)[cut] * h, rect[cut], color=color)
fix_axis_labels()
#_savefig("7b.pdf")

show()
