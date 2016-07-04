# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an Astropy affiliated package.
"""
from main import Dataset, SherpaFitter, ConvertedModel, OptMethod, Stat, EstMethod
# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    pass

def make_dataset(n_dim, x, y, z=None, xerr=None, yerr=None, zerr=None, bkg=None, bkg_scale=1):  

    """
    Parameters
        ----------
        n_dim: int
            Used to veirfy required number of dimentions.
        x : array (or list of arrays)
            input coordinates
        y : array (or list of arrays)
            input coordinates
        z : array (or list of arrays) (optional)
            input coordinates
        xerr : array (or list of arrays) (optional)
            an array of errors in x
        yerr : array (or list of arrays) (optional)
            an array of errors in y
        zerr : array (or list of arrays) (optional)
            an array of errors in z

    returns:
        _data: a sherpa dataset
    """

    return Dataset(n_dim, x, y, z, xerr, yerr, zerr, bkg, bkg_scale)


def make_fitter(optimizer="levmar", statistic="leastsq", estmethod="covariance"):

    """
    Sherpa Fitter for astropy models. Yay :)

    Parameters
        ----------
        optimizer : string
            the name of a sherpa optimizer.
        statistic : string
            the name of a sherpa statistic.
        estmethod : string
            the name of a sherpa estmethod.
    """
    return SherpaFitter(optimizer, statistic, estmethod)


def make_stat(value):

    """
    A wrapper for the fit statistics of sherpa

    Parameter:
        value: String
            the name of a sherpa statistics.
    """
    return Stat(value)


def make_opt(value):

    """
    A wrapper for the optimization methods of sherpa

    Parameter:
        value: String
            the name of a sherpa optimization method.
    """
    return OptMethod(value)


def make_est(value):

    """
    A wrapper for the error estimation methods of sherpa

    Parameter:
        value: String
            the name of a sherpa statistics.
    """
    return EstMethod(value)


def make_converted_model(models, tie_list=None):
    """
    This  wraps the model convertion to sherpa models and from astropy models and back!

    Parameters:
        models: model : `~astropy.modeling.FittableModel` (or list of)

        tie_list: list (optional)
            a list of parameter pairs which will be tied accross models
            e.g. [(modelB.y, modelA.x)] will mean that y in modelB will be tied to x of modelA
    """

    return ConvertedModel(models, tie_list)
