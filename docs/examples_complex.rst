More complex useage
===================

Now that you have the basics lets move on to some more complex useage of the fitter interface. 
Some quick Preamble

.. code-block::ipython
	from astropy.modeling.fitting import SherpaFitter
	sfitter = SherpaFitter(statistic='chi2', optimizer='levmar', estmethod='covariance')
	
	from astropy.modeling.models import Gaussian1D
	import numpy as np
	np.random.seed(0x1337)

Parameter constraints
---------------------

If you place any of the parameter constaraints on the astropy models then they will be respected by the fitter. Lets take a quick look at that. Firstly lets make a compound model by adding two `~astropy.modeling.functional_models.Gaussian1D` instances. 

.. code-block:: ipython

	double_gaussian = Gaussian1D(amplitude=10, mean=-1.5, stddev=0.5) + Gaussian1D(amplitude=3, mean=0.9, stddev=0.5)

Now we have the compound model lets add tie `amplitude_1` (the amplitude of the right hand side `~astropy.modeling.functional_models.Gaussian1D`) to `1.2*amplitude_0` and while we're at it let generate some data. 
To do this we must first define the `tiedfunc`

.. code-block:: ipython

	def tiedfunc(self): # a function used for tying amplitude_1
	return 1.2*self.amplitude_0

	double_gaussian.amplitude_1.tied = tiedfunc

	err = 0.8
	step = 0.2
	x = np.arange(-3, 3, step)
	y = double_gaussian(x) + err * np.random.uniform(-1, 1, size=len(x))
	yerrs = err * np.random.uniform(0.2, 1, size=len(x))
	binsize=(step/2) * np.ones(x.shape)  # please note these are binsize/2 not true errors! 


.. note :: without astropy PR #5129 we need to do this! 
	double_gaussian.amplitude_1.value = \
	double_gaussian.amplitude_1.tied(double_gaussian)

.. image:: _generated/example_plot_data2.png

Lets add some more parameter constraints to the model and fit the data. 
We can print the sherpa models to check things are doing what they should. 
 
.. code-block:: ipython

	fit_gg = double_gaussian.copy()
	fit_gg.mean_0.value = -0.5
	# sets the lower bound so we can force the parameter against it
	fit_gg.mean_0.min = -1.25
	fit_gg.mean_1.value = 0.8
	fit_gg.stddev_0.value = 0.9
	fit_gg.stddev_0.fixed = True

fitting this model is the same as earlier, we can also fit an unconstrained model for comparison. 

.. code-block:: ipython

	fitted_gg = sfitter(fit_gg,x, y, xbinsize=binsize, err=yerrs)

	sfitter2 = SherpaFitter(statistic='chi2', optimizer='levmar', estmethod='covariance')
	
	free_gg = sfitter2(double_gaussian.copy(), x, y, xbinsize=binsize, err=yerrs)


.. image:: _generated/example_plot_fitted2.png

The fitter keeps a copy of the converted model we can use it to compare the constrained and unconstrained model setups. 

.. note ::
	wrap\_.amplitude_1  should be `linked`, sherpa notation of astropy's `tied`
	wrap\_.stddev_0 should be `frozen`, sherpa notation for `fixed`
	and finally wrap\_.mean_0 should's value should have moved to its minimum while fitting!
	
	"wrap\_" is just prepended to the model name (we didn't set one so it's blank) on conversion to the sherpa model.

.. code-block:: ipython

	print("##Fit with constraints")
	print(sfitter._fitmodel.sherpa_model)
	print("##Fit without constraints")
	print(sfitter2._fitmodel.sherpa_model)

.. code-block:: ipython

	##Fit with constraints

	   Param        Type          Value          Min          Max      Units
	   -----        ----          -----          ---          ---      -----
	   wrap_.amplitude_0 thawed      5.58947 -3.40282e+38  3.40282e+38
	   wrap_.mean_0 thawed        -1.25        -1.25  3.40282e+38
	   wrap_.stddev_0 frozen          0.9 -3.40282e+38  3.40282e+38
	   wrap_.amplitude_1 linked      6.70736 expr: (1.2 * wrap_.amplitude_0)
	   wrap_.mean_1 thawed     0.869273 -3.40282e+38  3.40282e+38
	   wrap_.stddev_1 thawed     0.447021 -3.40282e+38  3.40282e+38

	##Fit without constraints

	   Param        Type          Value          Min          Max      Units
	   -----        ----          -----          ---          ---      -----
	   wrap_.amplitude_0 thawed      6.95483 -3.40282e+38  3.40282e+38
	   wrap_.mean_0 thawed     -1.59091 -3.40282e+38  3.40282e+38
	   wrap_.stddev_0 thawed     0.545582 -3.40282e+38  3.40282e+38
	   wrap_.amplitude_1 linked      8.34579 expr: (1.2 * wrap_.amplitude_0)
	   wrap_.mean_1 thawed     0.785016 -3.40282e+38  3.40282e+38
	   wrap_.stddev_1 thawed      0.46393 -3.40282e+38  3.40282e+38

Multiple models or multiple datasets
------------------------------------

We have three scenarios we can handle:
- fitting n datasets with n models
- fitting a single dataset with n models 
- or fitting n datasets with a single model

If n>1 for any of the scenarios we return a list of models. Firstly well look at a single dataset with the two models as above. 
We quickly copy the two models above and supply them to the fitter as a list - hopefully we get the same result. 

.. code-block:: ipython
	
	fit_gg = double_gaussian.copy()
	fit_gg.mean_0.value = -0.5
	fit_gg.mean_0.min = -1.25
	fit_gg.mean_1.value = 0.8
	fit_gg.stddev_0.value = 0.9
	fit_gg.stddev_0.fixed = True

	fm1,fm2 = sfitter([fit_gg, double_gaussian.copy()], x, y, xbinsize=binsize, err=yerrs)

.. image:: _generated/example_plot_simul.png

We also can fit multiple datasets with a single model so lets make a second datset. Lets generate a second dataset. 

.. code-block:: ipython

	second_gg = double_gaussian.copy()
	second_gg.mean_0 = -2
	second_gg.mean_1 = 0.5
	second_gg.amplitude_0 = 8
	second_gg.amplitude_1 = 5
	second_gg.stddev_0 = 0.4
	second_gg.stddev_1 = 0.8

	y2 = second_gg(x) + err * np.random.uniform(-1, 1, size=len(x))
	y2errs = err * np.random.uniform(0.2, 1, size=len(x))
	
We simply supply lists for each of the data parameters. You can also use `None` for when you don't have something like a missing binsizes - a lack of binsizes is a contrived example but a lack of y errors is not suitable for a chi:sup:2 fit and I don't want to make a new fitter. 

.. code-block:: ipython
	
	fit_gg=double_gaussian.copy()
	fit_gg.mean_0 = -2.3
	fit_gg.mean_1 = 0.7
	fit_gg.amplitude_0 = 2
	fit_gg.amplitude_1 = 3
	fit_gg.stddev_0 = 0.3
	fit_gg.stddev_1 = 0.5

	fm1,fm2 = sfitter(fit_gg, x=[x, x], y=[y, y2], xbinsize=[binsize, None], err=[yerrs, y2errs])

.. image:: _generated/example_plot_simul2.png

Background Data
---------------

We have error estimation and simultaneous fits but wait there's more you can also use background data!
This is required for many of the fit statistics as they are defined using the background data. 

All we have to do is supply a background array using the `bkg` keyword if there is a scaling of the background relative to the source spectra then you can use the `bkg_scale` keyword. 

.. code-block:: ipython

	y[y<0]=0
	cfitter = SherpaFitter(statistic='cstat', optimizer='levmar', estmethod='covariance')
	cfitter(fit_gg, x=x, y=y, xbinsize=binsize, err=yerrs, bkg=y, bkg_scale=0.3)

.. image:: _generated/example_plot_bkg.png
