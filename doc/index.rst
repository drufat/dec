**************************
Discrete Exterior Calculus
**************************

Contents
==========

.. toctree::
   :maxdepth: 3

   symbolic
   contraction
   wedge
   sample_advection
   paper
   bases
   hodge-star
   poisson

Installation
=============

To install simply run::

   pip install dec

or build from source::

   git clone https://github.com/drufat/dec.git
   cd dec
   python setup.py install

Operators
==========

.. autofunction:: dec.forms.operators

Grids
======

.. autofunction:: dec.grid1.Grid_1D_Periodic

.. plot:: plot/grid/periodic.py
	:width: 500

.. autofunction:: dec.grid1.Grid_1D_Chebyshev

.. plot:: plot/grid/cheb.py
	:width: 500

.. autofunction:: dec.grid1.Grid_1D_Regular

.. plot:: plot/grid/regular.py
	:width: 500

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

