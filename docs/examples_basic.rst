.. |saba| replace:: saba
.. |sherpa| replace:: `~sherpa`

Getting started
===============

I'll show you our API for the bridge. 
Firstly lets import the `~saba.SherpaFitter` class which is the interface with sherpa's fitting routines. 
`~saba.SherpaFitter` is available through `astropy.modeling.fitting` so it can be imported by:

.. code-block:: ipython

	from saba import SherpaFitter

or 

.. code-block:: ipython

	from astropy.modeling.fitting import SherpaFitter


Initialization
--------------

To initialize a fitter we simply provide names for ``statistic``, ``optimizer`` and ``estmethod`` this available value for those can be found in the docstring of  `~saba.SherpaFitter` these relate to objects withing `sherpa.stats`, `sherpa.optmethods` and `sherpa.estmethods`. 

.. code-block:: ipython

	sfit = SherpaFitter(statistic='chi2', optimizer='levmar', estmethod='confidence')

Now we have a fitter instance we need something to fit so lets import an astropy model specifically `~astropy.modeling.functional_models.Gaussian1D`. A full discription astropy's model and capabilities can be found `here <http://docs.astropy.org/en/stable/modeling/index.html>`_

.. code-block:: ipython

	from astropy.modeling.models import Gaussian1D

We also need some data so lets make some data with some added noise. 

.. code-block:: ipython

	import numpy as np

	np.random.seed(0x1337)
	true = Gaussian1D(amplitude=3, mean=0.9, stddev=0.5)
	err = 0.8
	step = 0.2
	x = np.arange(-3, 3, step)
	y = true(x) + err * np.random.uniform(-1, 1, size=len(x))

	yerrs = err * np.random.uniform(0.2, 1, size=len(x))
	binsize = step * np.ones(x.shape)
	# please note that binsize is the width of the bin!
	fit_model = true.copy() # ofset fit model from true

	fit_model.amplitude = 2
	fit_model.mean = 0
	fit_model.stddev = 0.2

For good measure lets plot it and take a look

.. image:: _generated/example_plot_data.png

Now we have some data let's fit it and get hopefully we get something similar to "True" back. 
As ``sfit`` has already been initialized as with other `astropy.modeling.fitting` fitters we just call it with some data and an astropy model and we get the fitted model returned. 

Fitting
-------

.. code-block:: ipython

	fitted_model = sfit(fit_model, x, y, xbinsize=binsize, err=yerrs)

Once again plotting the data.

.. image:: _generated/example_plot_fitted.png

Now we have a fit we can look at the outputs by doing:
	
.. code-block:: ipython
	
	print(sfit.fit_info)

.. code-block:: ipython
	
		datasets       = None
		itermethodname = none
		methodname     = levmar
		statname       = chi2
		succeeded      = True
		parnames       = ('wrap_.amplitude', 'wrap_.mean', 'wrap_.stddev')
		parvals        = (3.0646789274093185, 0.77853851419777986, 0.50721937454701504)
		statval        = 82.7366242121
		istatval       = 553.030876852
		dstatval       = 470.29425264
		numpoints      = 30
		dof            = 27
		qval           = 1.44381192266e-07
		rstat          = 3.06431941526
		message        = successful termination
		nfev           = 84


Uncertainty estimation and config
---------------------------------

One of the main driving forces behind this that using `sherpa` gives access to the uncertainty estimation methods, they are accessed through  `~saba.SherpaFitter.est_errors` method which uses the sherpa's  `~sherpa.fit.Fit.est_errors` method. 
As with the `~sherpa.optmethods` before we are able to adjust the configuration of the `~sherpa.estmethods`. Some of the properties can be passed through `~saba.SherpaFitter.est_errors` as keyword arguments such as the `sigma` however for access to all options we have the `est_config` property.

.. code-block:: ipython
	
	print(sfitter.est_config)
	sfitter.est_config['numcores'] = 5
	sfitter.est_config['max_rstat'] = 4
	print(sfitter.est_config)
	
.. code-block:: ipython
	
	{'eps': 0.01,
	 'fast': False,
	 'max_rstat': 3,
	 'maxfits': 5,
	 'maxiters': 200,
	 'numcores': 8,
	 'openinterval': False,
	 'parallel': True,
	 'remin': 0.01,
	 'sigma': 1,
	 'soft_limits': False,
	 'tol': 0.2,
	 'verbose': False}

	{'eps': 0.01,
	 'fast': False,
	 'max_rstat': 3,
	 'maxfits': 5,
	 'maxiters': 200,
	 'numcores': 5,
	 'openinterval': False,
	 'parallel': True,
	 'remin': 0.01,
	 'sigma': 1,
	 'soft_limits': False,
	 'tol': 0.2,
	 'verbose': False}


Then to use get the errors we can simply (N.B we can pass `sigma` in as a keyword). 

.. code-block:: ipython
	
	print(sfitter.est_config)
	sfitter.est_config['numcores'] = 5
	sfitter.est_config['max_rstat'] = 4
	print(sfitter.est_config)
	
.. code-block:: ipython
	
	{'eps': 0.01,
	 'fast': False,
	 'max_rstat': 3,
	 'maxfits': 5,
	 'maxiters': 200,
	 'numcores': 8,
	 'openinterval': False,
	 'parallel': True,
	 'remin': 0.01,
	 'sigma': 1,
	 'soft_limits': False,
	 'tol': 0.2,
	 'verbose': False}

	{'eps': 0.01,
	 'fast': False,
	 'max_rstat': 3,
	 'maxfits': 5,
	 'maxiters': 200,
	 'numcores': 5,
	 'openinterval': False,
	 'parallel': True,
	 'remin': 0.01,
	 'sigma': 1,
	 'soft_limits': False,
	 'tol': 0.2,
	 'verbose': False}


Then to use get the errors we can simply (N.B we can pass `sigma` in as a keyword). 
One of the main driving forces behind this that using sherpa gives access to the uncertainty estimation methods, they are accessed through  `~saba.SherpaFitter.est_errors` method which uses the sherpa's  `~sherpa.fit.Fit.est_errors` method. Then to use get the errors we can simply (N.B we can pass `sigma` in as a keyword). 

.. code-block:: ipython

	param_errors = sfit.est_errors(sigma=3)

In return we get a tuple of (prameter_name, best_fit_value, lower_value, upper_value) for the sake of plotting them we make models for the upper and lower values, lets output the values while we're at it. 

.. code-block:: ipython

	min_model = fitted_model.copy()
	max_model = fitted_model.copy()

	for pname, pval, pmin, pmax in zip(*param_errors):
		print(pname, pval, pmin, pmax)
		getattr(min_model, pname).value = pval + pmin
		getattr(max_model, pname).value = pval + pmax

.. code-block:: ipython

	('amplitude', 3.0646789274093185, -0.50152026852144349, 0.56964617033348119)
	('mean', 0.77853851419777986, -0.096264447380365548, 0.10293940565584792)
	('stddev', 0.50721937454701504, -0.098092469817728456, 0.11585973498734969)

.. image:: _generated/example_plot_error.png
