Wedge Product in Coordinates
============================

Given a manifold :math:`M`, the wedge product is a map that constructs
higher order forms

.. math:: \wedge:\quad\Lambda^{k}\times\Lambda^{l}\to\Lambda^{k+l}

The wedge product has the following properties:

-  :math:`\alpha\wedge\beta` is **associative**:
   :math:`\alpha\wedge(\beta\wedge\gamma)=(\alpha\wedge\beta)\wedge\gamma`

-  :math:`\alpha\wedge\beta` is **bilinear** in :math:`\alpha` and
   :math:`\beta`:

   .. math:: (a\alpha_{1}+b\alpha_{2})\wedge\beta

   .. math:: \alpha\wedge(c\beta_{1}+d\beta_{2})

-  :math:`\alpha\wedge\beta` is **anticommutative**:
   :math:`\alpha\wedge\beta=(-1)^{kl}\beta\wedge\alpha`, where
   :math:`\alpha` is a :math:`k`-form and :math:`\beta` is an
   :math:`l`-form.

In the discrete setting we will only be able to preserve some of these
continuous properties. Namely, the bilinearity and anticommutativity
will be preserved exactly, whereas the associativity will be satisfied
only in the limit where the mesh size tends to zero (:math:`h\to0`) and
will not be exact.

The wedge product is a an operator that is independent of the metric,
i.e. it is a homomorphism under a pull-back:

.. math:: \varphi^{*}(\alpha\wedge\beta)=(\varphi^{*}\alpha)\wedge(\varphi^{*}\beta)

Examples
--------

Consider the **2D** case

-  :math:`\wedge:\quad\Lambda^{0}\times\Lambda^{1}\to\Lambda^{1}`

.. math::

   \begin{aligned}
   \alpha & =\alpha\\
   \beta & =\beta_{x}dx+\beta_{y}dy\\
   \alpha\wedge\beta & =\alpha\beta_{x}dx+\alpha\beta_{y}dy\end{aligned}

-  :math:`\wedge:\quad\Lambda^{1}\times\Lambda^{1}\to\Lambda^{2}`

.. math::

   \begin{aligned}
   \alpha & =\alpha_{x}dx+\alpha_{y}dy\\
   \beta & =\beta_{x}dx+\beta_{y}dy\\
   \alpha\wedge\beta & =\left(\alpha_{x}\beta_{y}-\beta_{x}\alpha_{y}\right)dx\wedge dy\end{aligned}

The **3D** case, on the other hand, will be

-  :math:`\wedge:\quad\Lambda^{0}\times\Lambda^{1}\to\Lambda^{1}`

.. math::

   \begin{aligned}
   \alpha & =\alpha\\
   \beta & =\beta_{x}dx+\beta_{y}dy+\beta_{z}dz\\
   \alpha\wedge\beta & =\alpha\beta_{x}dx+\alpha\beta_{y}dy+\alpha\beta_{z}dz\end{aligned}

-  :math:`\wedge:\quad\Lambda^{1}\times\Lambda^{1}\to\Lambda^{2}`

.. math::

   \begin{aligned}
   \alpha= & \alpha_{x}dx+\alpha_{y}dy+\alpha_{z}dz\\
   \beta= & \beta_{x}dx+\beta_{y}dy+\beta_{z}dz\\
   \alpha\wedge\beta= & (\alpha_{x}\beta_{y}-\alpha_{y}\beta_{x})dx\wedge dy+\\
   + & (\alpha_{y}\beta_{z}-\alpha_{z}\beta_{y})dy\wedge dz+\\
   + & (\alpha_{z}\beta_{x}-\alpha_{x}\beta_{z})dz\wedge dx\end{aligned}

-  :math:`\wedge:\quad\Lambda^{1}\times\Lambda^{2}\to\Lambda^{3}`

.. math::

   \begin{aligned}
   \alpha & =\alpha_{x}dx+\alpha_{y}dy+\alpha_{z}dz\\
   \beta & =\beta_{xy}dx\wedge dy+\beta_{yz}dy\wedge dz+\beta_{zx}dz\wedge dx\\
   \alpha\wedge\beta & =(\alpha_{x}\beta_{yz}+\alpha_{y}\beta_{zx}+\alpha_{z}\beta_{xy})dx\wedge dy\wedge dz\end{aligned}


