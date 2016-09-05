from astropy.modeling.core import FittableModel


class Split(FittableModel):
    """
    Allows inputs to be split into multiple outputs based on indexs

    Parameters
    ----------
    split_indexes : tuple
        A tuple of integers representing indices on which the input 
        array will be split
    name : str, optional
        A human-friendly name associated with this model instance
        (particularly useful for identifying the individual components of a
        compound model).
    meta : dict-like
        Free-form metadata to associate with this model.

    """
    def __init__(self, split_indexes, name=None, meta=None):
        self._inputs = tuple('x')
        self._outputs = tuple('x' + str(idx) for idx in \
                              range(len(split_indexes)+1))
        self._split_indexes = split_indexes
        super(Split, self).__init__(name=name, meta=meta)

    def _format_expression(self):
        first = "x[:{0}]".format(self._split_indexes[0])
        last = "x[{0}:]".format(self._split_indexes[-1])
        if self.n_outputs > 1:
            middle = ["x[{0}:{1}]".format(*indexes) for indexes in zip(self._split_indexes[:-1], self._split_indexes[1:])]
            return ", ".join([first] + middle + [last])
        else:
            return ", ".join([first, last])

    @property
    def split_indexes(self):
        """Integers representing indices of the inputs."""
        return self._split_indexes

    @property
    def inputs(self):
        """
        The name(s) of the input variable(s) on which a model is evaluated.
        """
        return self._inputs

    @property
    def outputs(self):
        """The name(s) of the output(s) of the model."""
        return self._outputs


    def __repr__(self):
        if self.name is None:
            return '<SplitInput1D {0})>'.format(self._format_expression())
        else:
            return '<SplitInput1D({0}, name={1})>'.format(self._format_expression(), self.name)

    def evaluate(self, x):
        first = x[:self._split_indexes[0]]
        last = x[self._split_indexes[-1]:]
        if self.n_outputs>2:
            middle=[x[indexes[0]:indexes[1]] for indexes in zip(self._split_indexes[:-1], self._split_indexes[1:])]
            return [first] + middle + [last]
        else:
            return [first, last]

    def __call__(self, x):
        return self.evaluate(x)


class Join(FittableModel):
    """
    Flattens output

    Parameters
    ----------
    n_inputs : int
        We need to tell it how many inputs to expect.
    name : str, optional
        A human-friendly name associated with this model instance
        (particularly useful for identifying the individual components of a
        compound model).
    meta : dict-like
        Free-form metadata to associate with this model.

    """

    def __init__(self, n_inputs, name=None, meta=None):
        self._inputs = tuple('x' + str(idx) for idx in range(n_inputs+1))
        self._outputs=tuple("x")
        super(Join, self).__init__(name=name, meta=meta)

    @property
    def inputs(self):
        """
        The name(s) of the input variable(s) on which a model is evaluated.
        """
        return self._inputs

    @property
    def outputs(self):
        """The name(s) of the output(s) of the model."""
        return self._outputs

    def evaluate(self, *x):
        return np.hstack(x)

    def __call__(self, *x):
        return self.__call__(x)


def model_split_join(xvals, models):
    ''' This returns a compound model which allows the
    tying of parameters. the input values xvals should be
    flattened before passing into the returned model.

    Parameters
    ----------
    xvals: list of arrays
        arrays of input coordinates which should then
        be flattend and input

    models: a list of `astropy.modeling.core.FittableModel`
        the models which are combined

    Returns
    -------
    combined_model: an instance of `astropy.modeling.core.FittableModel`
    '''

    if len(models) > 1:
        split_indexes = []
        for xx in xvals:
            split_indexes.append(len(xx))

        runnning_total = 0
	# the xvals should be flattend so.
        for n,ll in enumerate(split_indexes):
            running_total += ll
            split_indexes[n] = running_total

        splitin = Split(split_indexes)
        joinout = Join(len(split_indexes))
        mo = model[0]
        for m in models[1:]:
            mo &= m
        return splitin | mo | joinout
