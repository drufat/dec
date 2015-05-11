Periodic Hodge-Star
----------------------------

.. autofunction:: dec.spectral.H
.. autofunction:: dec.spectral.Hinv

where 

.. autofunction:: dec.spectral.I_diag
.. autofunction:: dec.spectral.S_diag

Regular Hodge-Star
----------------------------

.. autofunction:: dec.regular.H0d_regular
.. autofunction:: dec.regular.H1_regular
.. autofunction:: dec.regular.H0_regular
.. autofunction:: dec.regular.H1d_regular

where 

.. autofunction:: dec.spectral.A_diag

Note that these correspond to the symmetric basis functions for 1-forms.

Chebyshev Hodge-Star
----------------------------

.. autofunction:: dec.chebyshev.H0d_cheb
.. autofunction:: dec.chebyshev.H1_cheb
.. autofunction:: dec.chebyshev.H0_cheb
.. autofunction:: dec.chebyshev.H1d_cheb

where 

.. autofunction:: dec.spectral.Omega
.. autofunction:: dec.spectral.Omega_d
.. autofunction:: dec.spectral.extend

.. plot:: plot/cheb/resample.py

.. autofunction:: dec.spectral.unextend
.. autofunction:: dec.spectral.mirror0
.. autofunction:: dec.spectral.mirror1
.. autofunction:: dec.spectral.unmirror0
.. autofunction:: dec.spectral.unmirror1

.. autofunction:: dec.spectral.half_edge_base
.. autofunction:: dec.spectral.pick
