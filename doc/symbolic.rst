Symbolic Computations
----------------------

Assume you have a triangle in the plane defined by its three vertices at
:math:`(x_0,y_0)`, :math:`(x_1,y_1)`, and :math:`(x_2,y_2)`. Here we will
attempt to find a general expression for the moments of the triangle, where the
moments are defined as averages by the following expression:

.. math:: 

   \langle x^n y^m \rangle = \frac{1}{A}\int_\triangle x^n y^m \, dA 
   
For example, :math:`\langle x \rangle` and :math:`\langle y \rangle` are the
coordinates of the center of mass of the triangle.

Integrating manually to compute the first few moments, one can obtain:

.. math:: 

   \langle x \rangle = \frac{1}{3} \left(x_0+x_1+x_2\right) 

.. math:: 

   \langle x^2 \rangle = \frac{1}{6} \left(x_0^2+x_1^2+x_2^2+x_0 x_1+x_0 x_2+x_1 x_2\right) 

.. math::
   
   \langle x y \rangle = \frac{1}{12}\left(2 x_0 y_0+x_1 y_0+x_2 y_0+x_0
   y_1+2 x_1 y_1+x_2 y_1+x_0 y_2+x_1 y_2+2 x_2 y_2 \right)

Is it possible to obtain a closed form expression for general :math:`n` and
:math:`m` using combinations of the coordinates?

Can the same be done for line segments defined by their endpoints
:math:`(x_0,y_0)` and :math:`(x_1,y_1)`? Again the first few moments in this
case are:

.. math:: 

   \langle x \rangle = \frac{1}{2} \left(x_0+x_1\right) 
   
.. math::

   \langle x^2 \rangle = \frac{1}{3} \left(x_0^2+x_0 x_1+x_1^2\right) 

.. math::

   \langle x^3 \rangle = \frac{1}{4} \left(x_0^3+x_0^2 x_1+x_0 x_1^2+x_1^3\right) 

.. math::

   \langle x y \rangle = \frac{1}{6} \left(2 x_0 y_0+x_1 y_0+x_0 y_1+2 x_1 y_1\right) 

.. math:: 

   \langle x^2 y \rangle  = \frac{1}{12} \left( 3 x_0^2 y_0+2 x_0 x_1 y_0+x_1^2 y_0+x_0^2 y_1+2 x_0 x_1 y_1+3 x_1^2 y_1\right)

.. autofunction:: dec.symbolic.coords_symbolic 
.. autofunction:: dec.symbolic.moments_symbolic
.. autofunction:: dec.symbolic.moments_eval 
.. autofunction:: dec.symbolic.measure_eval
.. autofunction:: dec.symbolic.coord_eval

