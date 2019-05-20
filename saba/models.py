from collections import OrderedDict
from sherpa.models import UserModel, Parameter, SimulFitModel

__all__ = ('ConvertedModel',)

class ConvertedModel(object):
    """
    This  wraps the model convertion to sherpa models and from astropy models
    and back!

    Parameters
    ----------
    models: `astropy.modeling.FittableModel` (or list of)
    tie_list: list (optional)
        a list of parameter pairs which will be tied accross models
        e.g. [(modelB.y, modelA.x)] will mean that y in modelB will be tied to
        x of modelA
    """

    def __init__(self, models, tie_list=None):
        self.model_dict = OrderedDict()
        try:
            models.parameters  # does it quack
            self.sherpa_model = self._astropy_to_sherpa_model(models)
            self.model_dict[models] = self.sherpa_model
        except AttributeError:
            for mod in models:
                self.model_dict[mod] = self._astropy_to_sherpa_model(mod)

                if tie_list is not None:
                    for par1, par2 in tie_list:
                        getattr(self.model_dict[par1._model],
                                par1.name).link = getattr(
                                    self.model_dict[par2._model], par2.name)

            self.sherpa_model = SimulFitModel("wrapped_fit_model",
                                              self.model_dict.values())

    @staticmethod
    def _astropy_to_sherpa_model(model):
        """
        Converts the model using sherpa's usermodel suppling the parameter
        detail to sherpa then using a decorator to allow the call method to
        act like the calc method
        """

        def _calc2call(func):
            """This decorator makes call and calc work together."""

            def _converter(inp, *x):
                if func.n_inputs == 1:
                    retvals = func.evaluate(x[0], *inp)
                else:
                    retvals = func.evaluate(x[0], x[1], *inp)
                return retvals

            return _converter

        if len(model.ineqcons) > 0 or len(model.eqcons) > 0:
            raise AstropyUserWarning(
                'In/eqcons are not supported by sherpa these will be ignored!')

        pars = []
        linkedpars = []
        for pname in model.param_names:
            param = getattr(model, pname)
            attr_names_and_vals = [
                ("name", param.name),
                ("val", param.value),
                ("min", param.min),
                ("max", param.max),
                ("hard_min", param.min),
                ("hard_max", param.max),
                ("units", None),
                ("frozen", param.fixed),
                ("alwaysfrozen" False),
            ]

            if model.name is None:
                model._name = ""

            pars.append(
                Parameter(modelname="wrap_%s" % (model.name or ''),
                          **dict([(atr, val)
                                  for atr, val in attr_names_and_vals
                                  if val is not None])))
            if param.tied is not False:
                linkedpars.append(pname)

        smodel = UserModel(model.name, pars)
        smodel.calc = _calc2call(model)

        for pname in linkedpars:
            param = getattr(model, pname)
            sparam = getattr(smodel, pname)
            sparam.link = param.tied(smodel)

        return smodel

    def get_astropy_model(self):
        """Returns an astropy model based on the sherpa model"""
        return_models = []

        for apymod, shmod in self.model_dict.items():
            return_models.append(apymod.copy())
            for pname, pval in map(lambda p: (p.name, p.val), shmod.pars):
                getattr(return_models[-1], pname.split(".")[-1]).value = pval

        if len(return_models) > 1:
            return return_models
        else:
            return return_models[0]
