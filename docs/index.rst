
.. toctree::
   :caption: Classes
   :name: Sclass

   sherpafitter

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :name: Sexample
   
   examples_basic
   examples_complex
   examples_2d
   examples_mcmc



Welcome to AstroSherpaBridge's documentation!
=============================================

I would like to introduce you to the Google Summer of Code project, a brigde between `sherpa` and `astropy.modelling` this project is to allow astropy users to take advantage of the `sherpa` package, giving them acess to the fitting routines and uncertainties estimation.

Installation
------------

.. note::
	If you wish to import sherpa's fit routine through astropy you must use astropy's master github (not yet haven't PR'd) as it requires the patch which inserts the entry_points created by `astrosherpa_bridge` into `astropy.modeling.fitting`.



