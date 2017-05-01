.. include:: references.txt

Usage details
==============

Now that you have the basics let's move on to some more complex usage of the fitter interface.
First a quick preamble to do some imports and create our |SherpaFitter| object.

.. doctest-skip::

    from saba import SherpaFitter
    sfit = SherpaFitter(statistic='chi2', optimizer='levmar', estmethod='confidence')

    from astropy.modeling.models import Gaussian1D
    import numpy as np
    np.random.seed(0x1337)

Parameter constraints
---------------------

Parameter constraints can be set in astropy models, and those constraints are
taken into account during the fitting procedure. Let's take a quick look at
that. Firstly, let's make a compound model by adding two `~astropy.modeling.functional_models.Gaussian1D` instances.

After that, let's set the following constraint on the parameter `amplitude_1`
(the amplitude of the right hand side `~astropy.modeling.functional_models.Gaussian1D`):
`1.2*amplitude_0`.

In addition, let's create some synthetic data:

.. plot::
    :include-source:

    from astropy.modeling.models import Gaussian1D
    import numpy as np
    np.random.seed(0x1337)

    double_gaussian = Gaussian1D(
        amplitude=10, mean=-1.5, stddev=0.5) + Gaussian1D(amplitude=1, mean=0.9,
                                                         stddev=0.5)

    def tiedfunc(obj):  # a function used for tying amplitude_1
        return 1.2 * obj.amplitude_0

    double_gaussian.amplitude_1.tied = tiedfunc
    double_gaussian.amplitude_1.value = double_gaussian.amplitude_1.tied(
        double_gaussian)

    err = 0.8
    step = 0.2
    x = np.arange(-3, 3, step)
    y = double_gaussian(x) + err * np.random.uniform(-1, 1, size=len(x))
    yerrs = err * np.random.uniform(0.2, 1, size=len(x))
    # please note these are binsize/2 not true errors!
    binsize = (step / 2) * np.ones(x.shape)

    plt.errorbar(x, y, xerr=binsize, yerr=yerrs, ls="", label="data")
    # once again xerrs are binsize/2 not true errors!
    plt.plot(x, double_gaussian(x), label="True")
    plt.legend(loc=(0.78, 0.8), frameon=False)
    plt.xlim((-3, 3))

.. note :: without astropy PR #5129 we need to do this
    ``double_gaussian.amplitude_1.value = double_gaussian.amplitude_1.tied(double_gaussian)``

Let's set some more parameter constraints to the model and fit the data.
We can print the sherpa models to check things are doing what they should.

.. doctest-skip::

    fit_gg = double_gaussian.copy()
    fit_gg.mean_0.value = -0.5
    # sets the lower bound so we can force the parameter against it
    fit_gg.mean_0.min = -1.25
    fit_gg.mean_1.value = 0.8
    fit_gg.stddev_0.value = 0.9
    fit_gg.stddev_0.fixed = True

Fitting Config
--------------

A `~saba.SherpaFitter` object has the `opt_config` property which holds the
configuration details for the optimization routine. Its docstring contains
information about the the properties of the optimizer.

We can see those configuration details by using ``print(sfit.opt_config)``
which outputs:

.. doctest-skip::

    {'epsfcn': 1.1920928955078125e-07,
    'factor': 100.0,
    'ftol': 1.1920928955078125e-07,
    'gtol': 1.1920928955078125e-07,
    'maxfev': None,
    'verbose': 0,
    'xtol': 1.1920928955078125e-07}

Similarly for ``print(sfit.opt_config.__doc__)``:

.. doctest-skip::

    Levenberg-Marquardt optimization method.

    The Levenberg-Marquardt method is an interface to the MINPACK
    subroutine lmdif to find the local minimum of nonlinear least
    squares functions of several variables by a modification of the
    Levenberg-Marquardt algorithm [1]_.

    Attributes
    ----------
    ftol : number
       The function tolerance to terminate the search for the minimum;
       the default is sqrt(DBL_EPSILON) ~ 1.19209289551e-07, where
       DBL_EPSILON is the smallest number x such that `1.0 != 1.0 +
       x`. The conditions are satisfied when both the actual and
       predicted relative reductions in the sum of squares are, at
       most, ftol.

    xtol : number
       The relative error desired in the approximate solution; default
       is sqrt( DBL_EPSILON ) ~ 1.19209289551e-07, where DBL_EPSILON
       is the smallest number x such that `1.0 != 1.0 + x`. The
       conditions are satisfied when the relative error between two
       consecutive iterates is, at most, `xtol`.
    ...

The parameters can be changed as

.. doctest-skip::

    sfit.opt_config['ftol'] = 1e-5
    print(sfit.opt_config)

.. doctest-skip::

    {'epsfcn': 1.1920928955078125e-07,
     'factor': 100.0,
     'ftol': 1e-05,
     'gtol': 1.1920928955078125e-07,
     'maxfev': None,
     'verbose': 0,
     'xtol': 1.1920928955078125e-07}


Fitting this model is similar as showing previously. For the sake of
comparison let's also fit and unconstrained model:

.. doctest-skip::

    fitted_gg = sfit(fit_gg, x, y, xbinsize=binsize, err=yerrs)

    sfit_unconstrained = SherpaFitter(statistic='chi2', optimizer='levmar',
                                      estmethod='covariance')
    free_gg = sfit_unconstrained(double_gaussian.copy(), x, y,
                                 xbinsize=binsize, err=yerrs)

    plt.figure(figsize=(10, 5))
    plt.plot(x, double_gaussian(x), label="True")
    plt.errorbar(x, y, xerr=binsize, yerr=yerrs, ls="", label="data")
    plt.plot(x, fit_gg(x), label="Pre fit")
    plt.plot(x, fitted_gg(x), label="Fitted")
    plt.plot(x, free_gg(x), label="Free")
    plt.subplots_adjust(right=0.8)
    plt.legend(loc=(1.01, 0.55), frameon=False)
    plt.xlim((-3, 3))

.. plot::

    from saba import SherpaFitter
    from astropy.modeling.models import Gaussian1D
    import numpy as np
    import matplotlib.pyplot as plt

    sfit = SherpaFitter(statistic='chi2', optimizer='levmar', estmethod='confidence')

    double_gaussian = Gaussian1D(
        amplitude=10, mean=-1.5, stddev=0.5) + Gaussian1D(amplitude=1, mean=0.9,
                                                         stddev=0.5)

    def tiedfunc(self):  # a function used for tying amplitude_1
        return 1.2 * self.amplitude_0

    double_gaussian.amplitude_1.tied = tiedfunc
    double_gaussian.amplitude_1.value = double_gaussian.amplitude_1.tied(
        double_gaussian)

    err = 0.8
    step = 0.2
    x = np.arange(-3, 3, step)
    y = double_gaussian(x) + err * np.random.uniform(-1, 1, size=len(x))
    yerrs = err * np.random.uniform(0.2, 1, size=len(x))
    # please note these are binsize/2 not true errors!
    binsize = (step / 2) * np.ones(x.shape)

    fit_gg = double_gaussian.copy()
    fit_gg.mean_0.value = -0.5
    # sets the lower bound so we can force the parameter against it
    fit_gg.mean_0.min = -1.25
    fit_gg.mean_1.value = 0.8
    fit_gg.stddev_0.value = 0.9
    fit_gg.stddev_0.fixed = True

    fitted_gg = sfit(fit_gg, x, y, xbinsize=binsize, err=yerrs)
    sfit_unconstrained = SherpaFitter(statistic='chi2', optimizer='levmar',
                                      estmethod='confidence')
    free_gg = sfit_unconstrained(double_gaussian.copy(), x, y,
                                 xbinsize=binsize, err=yerrs)

    plt.figure(figsize=(10, 5))
    plt.plot(x, double_gaussian(x), label="True")
    plt.errorbar(x, y, xerr=binsize, yerr=yerrs, ls="", label="data")
    plt.plot(x, fit_gg(x), label="Pre fit")
    plt.plot(x, fitted_gg(x), label="Fitted")
    plt.plot(x, free_gg(x), label="Free")
    plt.subplots_adjust(right=0.8)
    plt.legend(loc=(1.01, 0.55), frameon=False)
    plt.xlim((-3, 3))



The fitter keeps a copy of the converted model so we can use it to compare the constrained and unconstrained model setups:

.. note ::
    ``wrap\_.amplitude_1``  should be `linked`, sherpa notation of astropy's `tied`
    ``wrap\_.stddev_0`` should be `frozen`, sherpa notation for `fixed`
    and finally ``wrap\_.mean_0``'s value should have moved to its minimum while fitting

    "wrap\_" is just perpended to the model name (we didn't set one so it's blank) on conversion to the sherpa `~sherpa.models.model.Model`.

.. doctest-skip::

    print("##Fit with constraints")
    print(sfit._fitmodel.sherpa_model)
    print("##Fit without constraints")
    print(sfit_unconstrained._fitmodel.sherpa_model)

.. doctest-skip::

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

Error Estimation Configuration
------------------------------

As with the `~sherpa.optmethods` before we are able to adjust the configuration of the `~sherpa.estmethods`. Some of the properties can be passed through `~saba.SherpaFitter.est_errors` as keyword arguments such as the `sigma` however for access to all options we have the `est_config` property.


.. doctest-skip::

    print(sfit.est_config)
    sfit.est_config['numcores'] = 5
    sfit.est_config['max_rstat'] = 4
    print(sfit.est_config)

.. doctest-skip::

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


Multiple models or multiple datasets
------------------------------------

We have three scenarios we can handle:

- Fitting ``N`` datasets with ``N`` models
- Fitting a single dataset with ``N`` models
- Fitting ``N`` datasets with a single model

If ``N > 1`` for any of the scenarios then calling the fitter will return a list of models. Firstly we look at a single dataset with the two models as above.
We quickly copy the two models above and supply them to the fitter as a list - hopefully we get the same result.

.. doctest-skip::

    fit_gg = double_gaussian.copy()
    fit_gg.mean_0.value = -0.5
    fit_gg.mean_0.min = -1.25
    fit_gg.mean_1.value = 0.8
    fit_gg.stddev_0.value = 0.9
    fit_gg.stddev_0.fixed = True

    fm1, fm2 = sfit([fit_gg, double_gaussian.copy()], x, y, xbinsize=binsize, err=yerrs)

.. plot::

    from saba import SherpaFitter
    from astropy.modeling.models import Gaussian1D, Gaussian2D
    import numpy as np
    import matplotlib.pyplot as plt

    sfitter = SherpaFitter(statistic='chi2', optimizer='levmar', estmethod='confidence')

    double_gaussian = Gaussian1D(
        amplitude=10, mean=-1.5, stddev=0.5) + Gaussian1D(amplitude=1, mean=0.9,
                                                         stddev=0.5)

    def tiedfunc(self):  # a function used for tying amplitude_1
        return 1.2 * self.amplitude_0

    double_gaussian.amplitude_1.tied = tiedfunc
    double_gaussian.amplitude_1.value = double_gaussian.amplitude_1.tied(
        double_gaussian)

    err = 0.8
    step = 0.2
    x = np.arange(-3, 3, step)
    y = double_gaussian(x) + err * np.random.uniform(-1, 1, size=len(x))
    yerrs = err * np.random.uniform(0.2, 1, size=len(x))
    # please note these are binsize/2 not true errors!
    binsize = (step / 2) * np.ones(x.shape)

    fit_gg = double_gaussian.copy()
    fit_gg.mean_0.value = -0.5
    fit_gg.mean_0.min = -1.25
    fit_gg.mean_1.value = 0.8
    fit_gg.stddev_0.value = 0.9
    fit_gg.stddev_0.fixed = True

    fm1, fm2 = sfitter([fit_gg, double_gaussian.copy()],
                       x, y, xbinsize=binsize, err=yerrs)

    plt.figure(figsize=(10, 5))
    plt.plot(x, double_gaussian(x), label="True")
    plt.errorbar(x, y, xerr=binsize, yerr=yerrs, ls="", label="data")
    plt.plot(x, fit_gg(x), label="Pre fit")
    plt.plot(x, fm1(x), label="Constrained")
    plt.plot(x, fm2(x), label="Free")
    plt.subplots_adjust(right=0.8)
    plt.legend(loc=(1.01, 0.55), frameon=False)
    plt.xlim((-3, 3))


We also can fit multiple datasets with a single model so let's make a second dataset:

.. doctest-skip::

    second_gg = double_gaussian.copy()
    second_gg.mean_0 = -2
    second_gg.mean_1 = 0.5
    second_gg.amplitude_0 = 8
    second_gg.amplitude_1 = 5
    second_gg.stddev_0 = 0.4
    second_gg.stddev_1 = 0.8

    y2 = second_gg(x) + err * np.random.uniform(-1, 1, size=len(x))
    y2errs = err * np.random.uniform(0.2, 1, size=len(x))

Here we supply lists for each of the data parameters. You can also use ``None`` for when you don't have something like a missing binsizes - a lack of binsizes is a contrived example but a lack of ``y`` errors is not suitable for a chi:sup:2 fit and you don't want to make a new fitter.

.. doctest-skip::

    fit_gg = double_gaussian.copy()
    fit_gg.mean_0 = -2.3
    fit_gg.mean_1 = 0.7
    fit_gg.amplitude_0 = 2
    fit_gg.amplitude_1 = 3
    fit_gg.stddev_0 = 0.3
    fit_gg.stddev_1 = 0.5

    fm1, fm2 = sfit(fit_gg, x=[x, x], y=[y, y2], xbinsize=[binsize, None], err=[yerrs, y2errs])

.. plot::

    from saba import SherpaFitter
    from astropy.modeling.models import Gaussian1D, Gaussian2D
    import numpy as np
    import matplotlib.pyplot as plt

    sfitter = SherpaFitter(statistic='chi2', optimizer='levmar', estmethod='confidence')

    double_gaussian = Gaussian1D(
        amplitude=10, mean=-1.5, stddev=0.5) + Gaussian1D(amplitude=1, mean=0.9,
                                                         stddev=0.5)

    def tiedfunc(self):  # a function used for tying amplitude_1
        return 1.2 * self.amplitude_0

    double_gaussian.amplitude_1.tied = tiedfunc
    double_gaussian.amplitude_1.value = double_gaussian.amplitude_1.tied(
        double_gaussian)

    err = 0.8
    step = 0.2
    x = np.arange(-3, 3, step)
    y = double_gaussian(x) + err * np.random.uniform(-1, 1, size=len(x))
    yerrs = err * np.random.uniform(0.2, 1, size=len(x))
    # please note these are binsize/2 not true errors!
    binsize = (step / 2) * np.ones(x.shape)

    fit_gg = double_gaussian.copy()
    fit_gg.mean_0 = -2.3
    fit_gg.mean_1 = 0.7
    fit_gg.amplitude_0 = 2
    fit_gg.amplitude_1 = 3
    fit_gg.stddev_0 = 0.3
    fit_gg.stddev_1 = 0.5

    second_gg = double_gaussian.copy()
    second_gg.mean_0 = -2
    second_gg.mean_1 = 0.5
    second_gg.amplitude_0 = 8
    second_gg.amplitude_1 = 5
    second_gg.stddev_0 = 0.4
    second_gg.stddev_1 = 0.8
    second_gg.amplitude_1.value = second_gg.amplitude_1.tied(second_gg)

    yy2 = second_gg(x) + err * np.random.uniform(-1, 1, size=len(x))
    yy2errs = err * np.random.uniform(0.2, 1, size=len(x))

    plt.errorbar(x, y, xerr=binsize, yerr=yerrs, ls="", label="data1")
    plt.errorbar(x, yy2, yerr=yy2errs, ls="", label="data2")
    plt.plot(x, fit_gg(x), label="Prefit")

    fitted_model = sfitter(fit_gg, x=[x, x], y=[y, yy2], xbinsize=[
                           binsize, None], err=[yerrs, yy2errs])

    plt.plot(x, fitted_model[0](x), label="Fitted")
    plt.plot(x, fitted_model[1](x), label="Fitted")
    plt.subplots_adjust(right=0.8)

    plt.legend(loc=(1.01, 0.55), frameon=False)
    plt.xlim((-3, 3))

Background Data
---------------

It is also possible specify background data which is required for several of the fit statistics.

This is done by supplying a background array using the `bkg` keyword.  If there is a scaling of the background relative to the source data then you can use the `bkg_scale` keyword

.. doctest-skip::

    y[y<0]=0
    cfit = SherpaFitter(statistic='cstat', optimizer='levmar', estmethod='covariance')
    cfit(fit_gg, x=x, y=y, xbinsize=binsize, err=yerrs, bkg=y, bkg_scale=0.3)

.. plot::

    from saba import SherpaFitter
    from astropy.modeling.models import Gaussian1D, Gaussian2D
    import numpy as np
    import matplotlib.pyplot as plt

    double_gaussian = Gaussian1D(
        amplitude=10, mean=-1.5, stddev=0.5) + Gaussian1D(amplitude=1, mean=0.9,
                                                         stddev=0.5)

    def tiedfunc(self):  # a function used for tying amplitude_1
        return 1.2 * self.amplitude_0

    double_gaussian.amplitude_1.tied = tiedfunc
    double_gaussian.amplitude_1.value = double_gaussian.amplitude_1.tied(
        double_gaussian)

    err = 0.8
    step = 0.2
    x = np.arange(-3, 3, step)
    y = double_gaussian(x) + err * np.random.uniform(-1, 1, size=len(x))
    yerrs = err * np.random.uniform(0.2, 1, size=len(x))
    # please note these are binsize/2 not true errors!
    binsize = (step / 2) * np.ones(x.shape)

    y[y<0]=0
    cfitter = SherpaFitter(statistic='cstat', optimizer='levmar', estmethod='covariance')

    fit_gg = double_gaussian.copy()
    fit_gg.mean_0 = -2.3
    fit_gg.mean_1 = 0.7
    fit_gg.amplitude_0 = 2
    fit_gg.amplitude_1 = 3
    fit_gg.stddev_0 = 0.3
    fit_gg.stddev_1 = 0.5

    cmo=cfitter(fit_gg, x=x, y=y, xbinsize=binsize, err=yerrs, bkg=y, bkg_scale=0.3)

    plt.errorbar(x, y, yerr=yerrs, xerr=binsize)
    plt.plot(x, cmo(x))
