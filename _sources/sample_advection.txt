Sample Advection
================

We will obtain explicit analytical solutions for the Navier-Stokes equation in the two dimensional periodic domain :math:`[-\pi, \pi] \times [-\pi, \pi]`.

.. math::

    \dot{\mathbf{v}}+\left(\mathbf{v}\cdot\nabla\right)\mathbf{v}	& = &	-\nabla p \\
    \nabla\cdot\mathbf{v}	& = &	0

Applying the divergence operator on both sides of the first equation,

.. math::

    \nabla^2 p  & = & - \nabla \cdot \left(\left(\mathbf{v}\cdot\nabla\right)\mathbf{v}\right) \\

or in different notation

.. math::
    \Delta p & = & - \operatorname{div}(\operatorname{adv}(\mathbf{v}))

which can be solved for the pressure by inverting the Laplacian operator. On a periodic
domain this can be done by applying the Fourier transform on both sides.

.. math::

    -(u^2 + v^2) \mathcal{F}(p)  = - \mathcal{F} \left( \operatorname{div}(\operatorname{adv}(\mathbf{v})) \right)\\

And the solution for the pressure is

.. math::
    p =  \mathcal{F}^{-1} \left( \frac{-1}{u^2+v^2} \mathcal{F} (\operatorname{div}(\operatorname{adv}(\mathbf{v}))) \right)

from which it follows that the evolution in time is given by

.. math::

    \mathbf{\dot{v}} = -\nabla p - \left(\mathbf{v}\cdot\nabla\right)\mathbf{v}

Examples
---------

Let us compute the vorticity :math:`\omega`, pressure :math:`p`, and the time derivative of the velocity
:math:`\mathbf{\dot{v}}` for a prescribed initial velocity field :math:`\mathbf{v}(x,y)`.

Example 1
..........

.. math::

    \mathbf{v}(x, y) & = & \begin{pmatrix}- 2 \sin{\left (y \right )} \cos^{2}{\left (\frac{x}{2} \right )}, & 2 \sin{\left (x \right )} \cos^{2}{\left (\frac{y}{2} \right )}\end{pmatrix}  \\
    \omega (x,y) & = & 2 \cos{\left (x \right )} \cos{\left (y \right )} + \cos{\left (x \right )} + \cos{\left (y \right )} \\

.. plot::

    from dec.symbolic import *
    import matplotlib.pyplot as plt
    plot(plt, V1, p1)
    plt.show()

Example 2
..........

.. math::

    \mathbf{v}(x, y) & = & \begin{pmatrix}- \sin{\left (y \right )}, & \sin{\left (x \right )}\end{pmatrix} \\
    \omega (x,y) & = & \cos{\left (x \right )} + \cos{\left (y \right )}

.. plot::

    from dec.symbolic import *
    import matplotlib.pyplot as plt
    plot(plt, V2, p2)
    plt.show()

Example 3
..........

.. math::

    \mathbf{v}(x, y) & = & \begin{pmatrix}- \sin{\left (2 y \right )}, & \sin{\left (x \right )}\end{pmatrix} \\
    \omega (x,y) & = & \cos{\left (x \right )} + 2 \cos{\left (2 y \right )}

.. plot::

    from dec.symbolic import *
    import matplotlib.pyplot as plt
    plot(plt, V3, p3)
    plt.show()

Example 4
..........

.. math::
    \mathbf{v}(x, y) & = & \begin{pmatrix}1, & 0\end{pmatrix} \\
     \omega (x,y) & = & 0

.. plot::

    from dec.symbolic import *
    import matplotlib.pyplot as plt
    plot(plt, V4, p4)
    plt.show()

.. autofunction:: dec.symbolic.div
.. autofunction:: dec.symbolic.vort
.. autofunction:: dec.symbolic.grad
