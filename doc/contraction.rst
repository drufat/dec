Contraction in Coordinates
==========================

Given a manifold :math:`M`, the interior product is defined as the
contraction of a differential form with a vector field.

.. math:: \mathbf{i}_{X}:\quad\Lambda^{k}(M)\rightarrow\Lambda^{k-1}(M)

The contraction of a one-form :math:`\alpha=\alpha_{i}e^{i}` with a
vector field :math:`X=X^{i}e_{i}` is given by

.. math::

   \begin{aligned}
   \mathbf{i}_{X}\alpha & =\langle X,\alpha\rangle\\
    & =\alpha_{i}X^{j}\langle e^{i},e_{j}\rangle\\
    & =\alpha_{i}X^{j}\delta_{j}^{i}\\
    & =\alpha_{i}X^{i}\end{aligned}

The contraction of a two-form :math:`\beta=\beta_{ij}e^{i}\wedge e^{j}`
with the same vector field is given by

.. math::

   \begin{aligned}
   \mathbf{i}_{X}\beta & =\beta_{ij}\mathbf{i}_{X}(e^{i}\wedge e^{j})\\
    & =\beta_{ij}\left((\mathbf{i}_{X}e^{i})\wedge e^{j}-e^{i}\wedge(\mathbf{i}_{X}e^{j})\right)\\
    & =\beta_{ij}\left(X^{i}e^{j}-X^{j}e^{i}\right)\end{aligned}

The contraction of a three-form
:math:`\gamma=\gamma{}_{ijk}e^{i}\wedge e^{j}\wedge e^{k}` is going to
be

.. math::

   \begin{aligned}
   \mathbf{i}_{X}\gamma & =\gamma_{ijk}\mathbf{i}_{X}(e^{i}\wedge e^{j}\wedge e^{k})\\
    & =\gamma_{ijk}\left(\mathbf{i}_{X}e^{i}\wedge e^{j}\wedge e^{k}-e^{i}\wedge\mathbf{i}_{X}e^{j}\wedge e^{k}+e^{i}\wedge e^{j}\wedge\mathbf{i}_{X}e^{k}\right)\\
    & =\gamma_{ijk}\left(X^{i}e^{j}\wedge e^{k}-X^{j}e^{i}\wedge e^{k}+X^{k}e^{i}\wedge e^{j}\right)\end{aligned}

Examples
--------

In **2D** the explicit formula for the contractions are

:math:`\mathbf{i}:\quad\mathfrak{X}\wedge\Lambda^{1}\to\Lambda^{0}`

.. math::

   \begin{aligned}
   \alpha & =\alpha_{x}dx+\alpha_{y}dy\\
   \mathbf{i}_{X}\alpha & =X^{x}\alpha_{x}+X^{y}\alpha_{y}\end{aligned}

:math:`\mathbf{i}:\quad\mathfrak{X}\wedge\Lambda^{2}\to\Lambda^{1}`

.. math::

   \begin{aligned}
   \beta & =\beta_{xy}\,dx\wedge dy\\
   \mathbf{i}_{X}\beta & =-\beta_{xy}X^{y}dx+\beta_{xy}X^{x}dy\end{aligned}

If **3D**, on the other hand, the contractions become

:math:`\mathbf{i}:\quad\mathfrak{X}\wedge\Lambda^{1}\to\Lambda^{0}`

.. math::

   \begin{aligned}
   \alpha & =\alpha_{x}dx+\alpha_{y}dy+\alpha_{z}dz\\
   \mathbf{i}_{X}\alpha & =X^{x}\alpha_{x}+X^{y}\alpha_{y}+X^{z}\alpha_{z}\end{aligned}

:math:`\mathbf{i}:\quad\mathfrak{X}\wedge\Lambda^{2}\to\Lambda^{1}`

.. math::

   \begin{aligned}
   \beta & =\beta_{xy}dx\wedge dy+\beta_{yz}dy\wedge dz+\beta_{zx}dz\wedge dx\\
   \mathbf{i}_{X}\beta & =\left(\beta_{zx}X^{z}-\beta_{xy}X^{y}\right)dx+\\
    & +\left(\beta_{xy}X^{x}-\beta_{yz}X^{z}\right)dy+\\
    & +\left(\beta_{yz}X^{y}-\beta_{zx}X^{x}\right)dz\end{aligned}

:math:`\mathbf{i}:\quad\mathfrak{X}\wedge\Lambda^{3}\to\Lambda^{2}`

.. math::

   \begin{aligned}
   \gamma & =\gamma_{xyz}dx\wedge dy\wedge dz\\
   \mathbf{i}_{X}\gamma & =\gamma_{xyz}\left(X^{z}dx\wedge dy+X^{x}dy\wedge dz+X^{y}dz\wedge dx\right)\end{aligned}


