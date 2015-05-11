from numpy import *
from matplotlib.collections import LineCollection    

def plot_bases_1d(plt, g, xmin, xmax, name):

    N = len(g.verts)	
    z = linspace(xmin, xmax, 100)
    B0, B1, B0d, B1d = g.basis_fn()
    
    plt.subplot(221)
    plt.title("${%s}^0_{%d,n}$" % (name,N))
    for i, b in enumerate(B0):
        plt.plot(z, b(z), label=str(i))
        plt.scatter(g.verts, 0*g.verts)
    
    plt.subplot(222)
    plt.title("${%s}^1_{%d,n}$" % (name,N))
    for i, b in enumerate(B1):
        plt.plot(z, b(z)*g.delta[i], label=str(i))
        plt.scatter(g.verts, 0*g.verts)
    
    plt.subplot(223)
    plt.title("$\widetilde{%s}^0_{%d,n}$" % (name,N))
    for i, b in enumerate(B0d):
        plt.plot(z, b(z), label=str(i))
        plt.scatter(g.verts, 0*g.verts)
        plt.scatter(g.dual.verts, 0*g.dual.verts, color='red', marker='x')
    
    plt.subplot(224)
    plt.title("$\widetilde{{%s}}^1_{%d,n}$" % (name,N))
    for i, b in enumerate(B1d):
        plt.plot(z, b(z)*g.dual.delta[i], label=str(i))
        plt.scatter(g.verts, 0*g.verts)
        plt.scatter(g.dual.verts, 0*g.dual.verts, color='red', marker='x')

def grid_1d(ax, g):
    
    vertices_1d(ax, g)
    edges_1d(ax, g)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')

def vertices_1d(ax, g):    

    verts = g.verts
    ax.scatter(verts, 0*verts, color='k')
    
def edges_1d(ax, g):
    
    for x0, x1 in zip(*g.edges):
        ax.plot((x0, x1), (0, 0),
                color='r', linewidth=3, zorder=0)

if __name__ == '__main__':

    from dec.grid1 import *
    import matplotlib.pyplot as plt
    
    ax = plt.subplot(311)
    g = Grid_1D_Chebyshev(3)
    grid_1d(ax, g)

    ax = plt.subplot(312)
    g = Grid_1D_Chebyshev(5)
    grid_1d(ax, g)

    ax = plt.subplot(313)
    g = Grid_1D_Chebyshev(15)
    grid_1d(ax, g)

    plt.show()
