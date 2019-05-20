from sherpa.estmethods import Confidence, Covariance, Projection
from sherpa.optmethods import GridSearch, LevMar, MonCar, NelderMead
from sherpa.stats import (Cash, Chi2, Chi2ConstVar, Chi2DataVar, Chi2Gehrels,
                          Chi2ModVar, Chi2XspecVar, CStat, LeastSq, WStat)

from .util import SherpaWrapper

class Stat(SherpaWrapper):
    """
    A wrapper for the fit statistics of sherpa

    Parameters
    ----------

        value: String
            the name of a sherpa statistics.
    """

    _sherpa_values = {
        'cash': Cash,
        'wstat': WStat,
        'cstat': CStat,
        'chi2': Chi2,
        'chi2constvar': Chi2ConstVar,
        'chi2datavar': Chi2DataVar,
        'chi2gehrels': Chi2Gehrels,
        'chi2modvar': Chi2ModVar,
        'chi2xspecvar': Chi2XspecVar,
        'leastsq': LeastSq
    }


class OptMethod(SherpaWrapper):
    """
    A wrapper for the optimization methods of sherpa

    Parameters
    ----------

        value: String
            the name of a sherpa optimization method.
    """
    _sherpa_values = {
        'simplex': GridSearch,
        'levmar': LevMar,
        'moncar': MonCar,
        'neldermead': NelderMead
    }


class EstMethod(SherpaWrapper):
    """
    A wrapper for the error estimation methods of sherpa

    Parameters
    ----------
        value: String
            the name of a sherpa statistics.
    """

    _sherpa_values = {
        'confidence': Confidence,
        'covariance': Covariance,
        'projection': Projection
    }
