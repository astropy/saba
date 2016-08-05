
.. toctree::
   :maxdepth: 2
   :name: TreeOFWoe

   sherpafitter
   examples_basic   
   examples_complex
   examples_2d
   examples_mcmc


Welcome to Saba's documentation
===============================

I would like to introduce you to the Google Summer of Code project saba. Saba the sherpa word for bridge - accoring to an online dictonary I found.
Saba is an interface between `astropy.modeling` and sherpa to allow astropy users to take advantage of the sherpa package, giving them access to the statistics, fitting routines and uncertainties estimation.

Installation
------------

.. note::
	If you wish to import sherpa's fit routine through astropy you must use astropy's master github (not yet haven't PR'd) as it requires the patch which inserts the entry_points created by `saba` into `astropy.modeling.fitting`.



Prerequisites
**************
 * numpy 
 * cython 
 * jinja2
 * astropy
 * sherpa


.. code-block:: bash

   conda install numpy cython jinja2

To import though astropy (Pre PR):

.. code-block:: bash

   git clone https://github.com/nocturnalastro/astropy.git astropy
   cd astropy 
   git checkout sherpa_bridge_v2
   python setup.py develop
   cd ../

Sherpa currently needs to be built after astropy on Mac OSX. 

.. code-block:: bash

   conda install -c sherpa sherpa 

.. code-block:: bash

   git clone https://github.com/nocturnalastro/saba.git saba
   cd saba
   python setup.py develop
   cd ../


The astropy interface
---------------------

If you are using the dev or astropy>=1.3 you can access the `~saba.SherpaFitter` from `astropy.modeling.fitting`. `saba` creates entry points which are inserted into the  `~astropy.modeling.fitting` namespace.

