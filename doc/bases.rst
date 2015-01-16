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
.. autofunction:: dec.spectral.phid0
.. autofunction:: dec.spectral.phi1
.. autofunction:: dec.spectral.phid1

.. plot:: 

    from dec.grid1 import *
    from matplotlib.pyplot import *
    from dec.plot import plot_bases_1d
    N = 5
    g = Grid_1D_Periodic(N)
    plot_bases_1d(g, g.xmin, g.xmax, "\phi")

Regular Basis Functions
----------------------------

Even(Symmetric) 0-forms

.. autofunction:: dec.spectral.kappa0
.. autofunction:: dec.spectral.kappad0

Odd(Anti-symmetric) 1-forms

.. autofunction:: dec.spectral.kappa1
.. autofunction:: dec.spectral.kappad1

Even(Symmetric) 1-forms

.. autofunction:: dec.spectral.kappa1_symm
.. autofunction:: dec.spectral.kappad1_symm

In general basis functions for one-forms must be anti-symmetric because they
represent derivatives of the 0-forms bases, which have mirror symmetry 
between :math:`x` and :math:`2\pi-x`. Therefore, we should pick :py:func:`kappa1`
and :py:func:`kappa1d`, over :py:func:`kappa1_symm`
and :py:func:`kappa1d_symm`. However, when we do that, the hodge-stars are no longer 
exact inverses of each other (up to a sign) as they are in the continuous case. 
Is that because the anti-symmetric bases functions 
for 1-forms are no longer expressible as linear combinations of the 
symmetric bases functions for 0-forms?


.. plot:: 

    from dec.grid1 import *
    from matplotlib.pyplot import *
    from dec.plot import plot_bases_1d
    
    N = 2
    g = Grid_1D_Regular(N)
    figure()
    plot_bases_1d(g, g.xmin, g.xmax, "\kappa")
    
    N = 3
    g = Grid_1D_Regular(N)
    figure()
    plot_bases_1d(g, g.xmin, g.xmax, "\kappa")
    
    figure()
    plot_bases_1d(g, -g.xmax, g.xmax, "\kappa")
        
    show()
    
Chebyshev Basis Functions
----------------------------

.. autofunction:: dec.spectral.psi0
.. autofunction:: dec.spectral.psi1
.. autofunction:: dec.spectral.psid0
.. autofunction:: dec.spectral.psid1

.. plot:: 

    from dec.grid1 import *
    from matplotlib.pyplot import *
    from dec.plot import plot_bases_1d
    N = 5
    g = Grid_1D_Chebyshev(N)
    plot_bases_1d(g, g.xmin, g.xmax, "\kappa")
    show()

:math:`\psi^0` and :math:`\widetilde{\psi}^0` match the Lagrange polynomials

.. autofunction:: dec.spectral.lagrange_polynomials

:math:`\psi^1` and :math:`\widetilde{\psi}^1` match the Gerritsma edge functions as linear 
combinations of the derivatives of :math:`\psi^0` and are therefore also polynomials.

.. math::

    \psi_{N,n}^{1}(x)=-\sum_{k=0}^{n}\frac{d}{dx}\psi_{N,k}^{0}(x)


2D Periodic Basis Function
-----------------------------

.. plot:: plot/cheb/basis_forms_2d.py

.. plot:: plot/cheb/vector_field_plot.py

