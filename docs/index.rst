
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


Welcome to Saba's documentation!
=============================================

I would like to introduce you to the Google Summer of Code project saba. Saba the sherpa workd for bridge (Dislamer: Accoring to an online dictonary I found) but also an Acronym for Sherpa Astropy Bridge API.
Saba is an interface between `astropy.modeling` and `sherpa` to allow astropy users to take advantage of the `sherpa` package, giving them access to the statistics, fitting routines and uncertainties estimation.

Installation
------------

.. note::
	If you wish to import sherpa's fit routine through astropy you must use astropy's master github (not yet haven't PR'd) as it requires the patch which inserts the entry_points created by `saba` into `astropy.modeling.fitting`.

Conda:
------



Source:
-------
