from astropy.modeling.fitting import SherpaFitter
from astropy.modeling.models import Gaussian1D, Gaussian2D
import numpy as np
import matplotlib.pyplot as plt

sfitter = SherpaFitter(
    statistic='chi2', optimizer='levmar', estmethod='covariance')

np.random.seed(0x1337)

true = Gaussian1D(amplitude=3, mean=0.9, stddev=0.5)
err = 0.8
step = 0.2
x = np.arange(-3, 3, step)
y = true(x) + err * np.random.uniform(-1, 1, size=len(x))

yerrs = err * np.random.uniform(0.2, 1, size=len(x))
# binsize=step * np.ones(x.shape)  # please note these are binsize/2 not
# true errors!
# please note these are binsize/2 not true errors!
binsize = step * np.ones(x.shape)

fit_model = true.copy()  # ofset fit model from true
fit_model.amplitude = 2
fit_model.mean = 0
fit_model.stddev = 0.2

plt.plot(x, true(x), label="True")
plt.errorbar(x, y, xerr=binsize, yerr=yerrs, ls="", label="Data")
plt.plot(x, fit_model(x), label="Starting fit model")
plt.legend(loc=(0.02, 0.7), frameon=False)
plt.xlim((-3, 3))
plt.savefig("_generated/example_plot_data.png")
plt.close('all')


fitted_model = sfitter(fit_model, x, y, xbinsize=binsize, err=yerrs)

plt.plot(x, true(x), label="True")
plt.errorbar(x, y, xerr=binsize, yerr=yerrs, ls="", label="Data")
plt.plot(x, fit_model(x), label="Starting fit model")
plt.plot(x, fitted_model(x), label="Fitted model")
plt.legend(loc=(0.02, 0.6), frameon=False)
plt.xlim((-3, 3))
plt.savefig("_generated/example_plot_fitted.png")
plt.close('all')


param_errors = sfitter.est_errors(sigma=3)
min_model = fitted_model.copy()
max_model = fitted_model.copy()

for pname, pval, pmin, pmax in zip(*param_errors):
    getattr(min_model, pname).value = pval + pmin
    getattr(max_model, pname).value = pval + pmax


plt.plot(x, true(x), label="True")
plt.errorbar(x, y, xerr=binsize, yerr=yerrs, ls="")
plt.plot(x, fitted_model(x), label="Fitted model")
plt.plot(x, min_model(x), label="min model", ls="--")
plt.plot(x, max_model(x), label="max model", ls="--")
plt.legend(loc=(0.02, 0.6), frameon=False)
_ = plt.xlim((-3, 3))

plt.savefig("_generated/example_plot_error.png")
plt.close('all')


double_gaussian = Gaussian1D(
    amplitude=7, mean=-1.5, stddev=0.5) + Gaussian1D(amplitude=1, mean=0.9,
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


plt.errorbar(x, y, xerr=binsize, yerr=yerrs, ls="", label="data")
# once again xerrs are binsize/2 not true errors!
plt.plot(x, double_gaussian(x), label="True")
plt.legend(loc=(0.78, 0.8), frameon=False)
_ = plt.xlim((-3, 3))


plt.savefig("_generated/example_plot_data2.png")
plt.close('all')


fit_gg = double_gaussian.copy()
fit_gg.mean_0.value = -0.5
# sets the lower bound so we can force the parameter against it
fit_gg.mean_0.min = -1.25
fit_gg.mean_1.value = 0.8
fit_gg.stddev_0.value = 0.9
fit_gg.stddev_0.fixed = True

fitted_gg = sfitter(fit_gg, x, y, xbinsize=binsize, err=yerrs)
print("##Fit with contraints")
print(sfitter._fitmodel.sherpa_model)

free_gg = sfitter(double_gaussian.copy(), x, y, xbinsize=binsize, err=yerrs)
print
print("##Fit without contraints")
print(sfitter._fitmodel.sherpa_model)

plt.figure(figsize=(10, 5))
plt.plot(x, double_gaussian(x), label="True")
plt.errorbar(x, y, xerr=binsize, yerr=yerrs, ls="", label="data")
plt.plot(x, fit_gg(x), label="Pre fit")
plt.plot(x, fitted_gg(x), label="Fitted")
plt.plot(x, free_gg(x), label="Free")
plt.subplots_adjust(right=0.8)
plt.legend(loc=(1.01, 0.55), frameon=False)
plt.xlim((-3, 3))

plt.savefig("_generated/example_plot_fitted2.png")
plt.close('all')


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


plt.savefig("_generated/example_plot_simul.png")
plt.close("all")


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


plt.savefig("_generated/example_plot_simul2.png")
plt.close("all")


wfitter = SherpaFitter(
    statistic='chi2', optimizer='levmar', estmethod='covariance')


mo=wfitter(fit_gg, x=x, y=y+20, xbinsize=binsize, err=yerrs, bkg=(y+20)*0.1, bkg_scale=200.0)
mo2=wfitter(fit_gg, x=x, y=y+20, xbinsize=binsize, err=yerrs, bkg=(y+20)*0.1, bkg_scale=.01)


plt.errorbar(x, y+20,yerr=yerrs,xerr=binsize)
plt.plot(x, mo(x))
plt.plot(x, mo2(x),ls="--")


plt.savefig("_generated/example_plot_bkg.png")
plt.close("all")



np.random.seed(123456789)
x0low, x0high = 3000, 4000
x1low, x1high = 4000, 4800
dx = 15
x1, x0 = np.mgrid[x1low:x1high:dx, x0low:x0high:dx]
shape = x0.shape
x0, x1 = x0.flatten(), x1.flatten()



plt.rcParams['figure.figsize']=(15,5)

truth = Gaussian2D(x_mean=3512, y_mean=4418, x_stddev=150, y_stddev=150,
                   theta=20, amplitude=100)
mexp = truth(x0, x1).reshape(shape)
merr = abs(np.random.poisson(mexp) - mexp)

plt.subplot(1,3,1)
plt.imshow(mexp, origin='lower', cmap='viridis',
           extent=(x0low, x0high, x1low, x1high),
           interpolation='nearest', aspect='auto')
plt.title("True")
plt.subplot(1,3,2)
plt.imshow(merr, origin='lower', cmap='viridis',
           extent=(x0low, x0high, x1low, x1high),
           interpolation='nearest', aspect='auto')
plt.title("Noise")
plt.subplot(1,3,3)
plt.imshow((mexp + merr), origin='lower', cmap='viridis',
           extent=(x0low, x0high, x1low, x1high),
           interpolation='nearest', aspect='auto')
plt.title("True+Noise")

plt.savefig("_generated/example_plot_2d_data.png")
plt.close("all")


sfit = SherpaFitter(statistic="chi2")
fitmo = truth.copy()
fitmo.x_mean = 3650
fitmo.y_mean = 4250
fitmo.x_stddev = 100
fitmo.y_stddev = 100
fitmo.theta = 10
fitmo.amplitude = 50

fitmo = sfit(fitmo, x0.flatten(), x1.flatten(), mexp.flatten()+merr.flatten(), xbinsize=np.ones(x0.size)*dx, ybinsize=np.ones(x1.size)*dx, err=merr.flatten()+np.random.uniform(-0.5,0.5,x0.size))


plt.subplot(1,2,1)
plt.imshow(fitmo(x0,x1).reshape(shape), origin='lower', cmap='viridis',
           extent=(x0low, x0high, x1low, x1high),
           interpolation='nearest', aspect='auto')
plt.title("Fit Model")

res = (mexp + merr) - fitmo(x0, x1).reshape(shape)
plt.subplot(1,2,2)
plt.imshow(res, origin='lower', cmap='viridis',
           extent=(x0low, x0high, x1low, x1high),
           interpolation='nearest', aspect='auto')
plt.title("Residuals")


plt.savefig("_generated/example_plot_2d_fit.png")
plt.close("all")




print("Done")
