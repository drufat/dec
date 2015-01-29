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

    \mathbf{v}(x, y) & = & \left\{-2 \cos ^2\left(\frac{x}{2}\right) \sin (y),2 \sin (x) \cos ^2\left(\frac{y}{2}\right)\right\}  \\
    \omega (x,y) & = & 2 \cos (x) \cos (y)+\cos (x)+\cos (y) \\
    p(x,y) & = & \frac{1}{20} (\cos (2 x) (4 \cos (y)+5)+4 \cos (x) (5 \cos (y)+\cos (2 y)+5)+5 (4 \cos (y)+\cos (2 y))) \\
    \dot{\mathbf{v}}(x,y) & = & \left\{\frac{1}{5} \sin (x) (\cos (x) \cos (y)-\cos (2 y)),-\frac{1}{5} \sin (y) (\cos (2 x)-\cos (x) \cos (y))\right\}

.. plot::

    from dec.symbolic import *
    import matplotlib.pyplot as plt
    plot(plt, V[0], p[0])
    plt.show()

Example 2
..........

Is this the Rudman vortex?

.. math::

    \mathbf{v}(x, y) & = & \left \{ - \sin{\left (\frac{y}{2} \right )} \cos{\left (\frac{x}{2} \right )}, \quad \sin{\left (\frac{x}{2} \right )} \cos{\left (\frac{y}{2} \right )}\right \}  \\
    \omega (x,y) & = & \cos{\left (\frac{x}{2} \right )} \cos{\left (\frac{y}{2} \right )} \\
    p(x,y) & = & - \frac{1}{4} \cos{\left (x \right )} - \frac{1}{4} \cos{\left (y \right )} \\
    \dot{\mathbf{v}}(x,y) & = & \left\{0 , 0\right\}

.. plot::

    from dec.symbolic import *
    import matplotlib.pyplot as plt
    plot(plt, V[1], p[1])
    plt.show()

Example 3
..........

.. math::

    \mathbf{v}(x, y) & = & \{-\sin (y),\sin (x)\} \\
    \omega (x,y) & = & \cos (x)+\cos (y) \\
    p(x,y) & = & \cos (x) \cos (y) \\
    \dot{\mathbf{v}}(x,y) & = & \{0,0\}

.. plot::

    from dec.symbolic import *
    import matplotlib.pyplot as plt
    plot(plt, V[2], p[2])
    plt.show()

Example 4
..........

.. math::

    \mathbf{v}(x, y) & = & \{-\sin (2y),\sin (x)\} \\
    \omega (x,y) & = & \cos (x)+2 \cos (2 y)\\
    p(x,y) & = & \frac{4}{5} \cos (x) \cos (2 y) \\
    \dot{\mathbf{v}}(x,y) & = & \left\{\frac{6}{5} \sin (x) \cos (2 y),\frac{1}{5} (-3) \cos (x) \sin (2 y)\right\}

.. plot::

    from dec.symbolic import *
    import matplotlib.pyplot as plt
    plot(plt, V[3], p[3])
    plt.show()

Example 5
..........

.. math::
    \mathbf{v}(x, y) & = & \{1,0\} \\
    \omega (x,y) & = & 0\\
    p(x,y) & = & 0\\
    \dot{\mathbf{v}}(x,y) & = & \{0,0\}

.. plot::

    from dec.symbolic import *
    import matplotlib.pyplot as plt
    plot(plt, V[4], p[4])
    plt.show()

.. autofunction:: dec.symbolic.div
.. autofunction:: dec.symbolic.vort
.. autofunction:: dec.symbolic.grad
