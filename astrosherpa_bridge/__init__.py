# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an Astropy affiliated package.
"""

from main import Dataset, SherpaFitter, ConvertedModel, OptMethod, Stat
from main import EstMethod, SherpaMCMC
# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    pass



#from astropy.modeling.plugin import ModelingEntryPoint
#I'll move this later
class ModelingEntryPoint(object):

    
    def __init__(self):
        self.__doc__=self.entry.__doc__

    def get_info(self):
        return self.module, self.name, self.doc
    
    def __call__(self,*args,**kwargs):

        return self.entry(*args,**kwargs)



class MCMC_Entry(ModelingEntryPoint):

    doc = "An interface which makes use of sherpa's MCMC(pyBLoCXS) functionality."
    name = "SherpaMCMC"
    module = "astrosherpa_bridge"
    entry = SherpaMCMC
    



class Fitter_Entry(ModelingEntryPoint):
   
    doc = "An interface to allow astropy models to use sherpa's fit routines"
    name = "SherpaFitter"
    module = "astrosherpa_bridge"
    entry = SherpaFitter

