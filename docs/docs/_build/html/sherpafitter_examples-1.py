import numpy as np

np.random.seed(0x1337)
true = Gaussian1D(amplitude=3, mean=0.9, stddev=0.5)
err = 0.8
step = 0.2
x = np.arange(-3, 3, step)
y = true(x) + err * np.random.uniform(-1, 1, size=len(x))

yerrs=err * np.random.uniform(0.2, 1, size=len(x))
binsize=(step / 2) * np.ones(x.shape)
# please note that the width of the bin will be 2*binsize!

fit_model = true.copy() # ofset fit model from true
fit_model.amplitude = 2
fit_model.mean = 0
fit_model.stddev = 0.2

import  matplotlib.pyplot as plt
plt.plot(x,true(x), label="True")
plt.errorbar(x, y, xerr=xerrs, yerr=yerrs, ls="", label="Data")
plt.plot(x,fit_model(x), label="Starting fit model")
plt.legend(loc=(0.02,0.7), frameon=False)
plt.xlim((-3,3))
plt.show()