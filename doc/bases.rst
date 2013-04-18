Mapping
----------

.. autofunction:: dec.spectral.varphi
.. autofunction:: dec.spectral.varphi_inv

Helper Functions
------------------

.. autofunction:: dec.spectral.alpha0
.. autofunction:: dec.spectral.beta0
.. autofunction:: dec.spectral.alpha
.. autofunction:: dec.spectral.beta

.. autofunction:: dec.spectral.rho
.. autofunction:: dec.spectral.gamma
.. autofunction:: dec.spectral.delta

Periodic Basis Functions
----------------------------

.. autofunction:: dec.spectral.phi0
.. autofunction:: dec.spectral.phi1
.. autofunction:: dec.spectral.phid0
.. autofunction:: dec.spectral.phid1

.. plot:: 

	from dec.spectral import *
	from matplotlib.pyplot import *
	from dec.plot import plot_bases_1d
	N = 5
	g = Grid_1D_Periodic(N)
	plot_bases_1d(g, g.xmin, g.xmax, "\phi")

Regular Basis Functions
----------------------------

.. autofunction:: dec.spectral.kappa0
.. autofunction:: dec.spectral.kappa1
.. autofunction:: dec.spectral.kappad0
.. autofunction:: dec.spectral.kappad1

Even 0-forms, Odd 1-forms

.. plot:: 

    from dec.spectral import *
    from matplotlib.pyplot import *
    from dec.plot import plot_bases_1d
    
    N = 2
    g = Grid_1D_Regular(N)
    figure()
    plot_bases_1d(g, g.xmin, g.xmax, "\kappa")
    
    N = 5
    g = Grid_1D_Regular(N)
    figure()
    plot_bases_1d(g, g.xmin, g.xmax, "\kappa")
    
    figure()
    plot_bases_1d(g, -g.xmax, g.xmax, "\kappa")
    
    figure()
    N = 9
    g = Grid_1D_Regular(N)
    plot_bases_1d(g, g.xmin, g.xmax, "\kappa")
    
    show()
	
Chebyshev Basis Functions
----------------------------

.. autofunction:: dec.spectral.psi0
.. autofunction:: dec.spectral.psi1
.. autofunction:: dec.spectral.psid0
.. autofunction:: dec.spectral.psid1

.. plot:: 

	from dec.spectral import *
	from matplotlib.pyplot import *
	from dec.plot import plot_bases_1d
	N = 5
	g = Grid_1D_Chebyshev(N)
	plot_bases_1d(g, g.xmin, g.xmax, "\kappa")
	show()

:math:`\psi^0` and :math:`\widetilde{\psi}^0` match the Lagrange polynomials

.. autofunction:: dec.spectral.lagrange_polynomials


2D Periodic Basis Function
-----------------------------

.. plot:: plot/cheb/basis_forms_2d.py

.. plot:: plot/cheb/vector_field_plot.py

