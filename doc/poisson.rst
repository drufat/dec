Poisson Operator
================

1D
---

.. autofunction:: dec.spectral.laplacian

.. math::
    \mathbf{\nabla}^2 f = q

.. math::
    \star \mathbf{d} \star \mathbf{d} f=q
    
Discrete form:

Primal:

.. math::
	\mathbf{\widetilde{H}}^{1}
	(\mathbf{\widetilde{D}^{1}}
	\mathbf{H}^{1}
	\mathbf{D^{0}}
	\bar{f}^{0} + a)=\bar{q}^{0}

where `a` is Neumann b.c.

Dual:

.. math::
	\mathbf{H}^{1}
	\mathbf{D^{0}}
	\mathbf{\widetilde{H}}^{1}
	(\mathbf{\widetilde{D}^{1}}
	\tilde{f}^{0} + b)=\tilde{q}^{0}

where `b` is Dirichlet b.c.

Periodic grid 1D
----------------

.. math::
    \begin{eqnarray}
    f(x) &=& e^{\sin x} \\
    q(x) &=& e^{\sin x} (\cos^2 x - \sin x)
    \end{eqnarray}

.. plot:: plot/cheb/poisson1d_periodic.py

Chebyshev grid 1D
-----------------

.. math::
    \begin{eqnarray}
    f(x) &=&  e^x \\
    q(x) &=&  e^x
    \end{eqnarray}

Boundary Conditions:


Dirichlet boundary conditions:

.. math::

    f(-1) = e^{-1} \quad f(+1) = e
    
Neumann boundary condition:

.. math::

    f^\prime(-1)= e^{-1} \quad f^\prime(+1)= e
    
.. plot:: plot/cheb/poisson1d_cheb.py

2D
---

.. math::
    (\star \mathbf{d} \star \mathbf{d} + \mathbf{d} \star \mathbf{d} \star ) f=q

Periodic grid 2D
----------------

.. math::
	f = e^{\sin(x)} \mathbf{d}x + e^{\sin(y)} \mathbf{d}y

Convergence for operator.

.. plot:: plot/cheb/poisson2d_periodic.py

Chebyshev grid 2D
----------------- 

.. math::
	f = e^{x} \mathbf{d}x + e^{y} \mathbf{d}y

.. math::
	q = e^{x} \mathbf{d}x + e^{y} \mathbf{d}y

Convergence for operator.

.. plot:: plot/cheb/poisson2d_cheb.py

Vector field
------------

.. plot:: plot/cheb/poisson_2d_example.py
