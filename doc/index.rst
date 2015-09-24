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
    examples/schur
    examples/contraction
    examples/grid1d
    examples/grids
    examples/hodge
    examples/whitney

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

.. automethod:: dec.grid1.Grid_1D.periodic

.. plot:: plot/grid/periodic.py
	:width: 500

.. automethod:: dec.grid1.Grid_1D.chebyshev

.. plot:: plot/grid/cheb.py
	:width: 500

.. automethod:: dec.grid1.Grid_1D.regular

.. plot:: plot/grid/regular.py
	:width: 500

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

