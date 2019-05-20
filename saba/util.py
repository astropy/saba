from astropy.utils.exceptions import AstropyUserWarning

__all__ = ('SherpaWrapper', )

class SherpaWrapper(object):
    value = None

    def __init__(self, value=None):
        if value is not None:
            self.set(value)

    def set(self, value):
        try:
            self.value = self._sherpa_values[value.lower()]
        except KeyError:
            raise AstropyUserWarning("Value {} not found".format(
                value.lower()))  # todo handle
