from astropy.modeling.fitting import SherpaFitter
from astropy.modeling.models import Gaussian1D, Gaussian2D

import numpy as np
import matplotlib.pyplot as plt

sfitter = SherpaFitter(statistic='chi2', optimizer='levmar', estmethod='confidence')
sfitter.est_config['max_rstat'] = 4
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
    print(pname, pval, pmin, pmax)
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
print()
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


cy=y.copy()
cy[cy<0]=0
cfitter = SherpaFitter(statistic='cstat', optimizer='levmar', estmethod='covariance')
cmo=cfitter(fit_gg, x=x, y=cy, xbinsize=binsize, err=yerrs, bkg=y, bkg_scale=0.3)

plt.errorbar(x, cy, yerr=yerrs, xerr=binsize)
plt.plot(x, cmo(x))
plt.savefig("_generated/example_plot_bkg.png")
plt.close("all")


np.random.seed(123456789)
x0low, x0high = 3000, 4000
x1low, x1high = 4000, 4800
dx = 15
x1, x0 = np.mgrid[x1low:x1high:dx, x0low:x0high:dx]
shape = x0.shape
x0, x1 = x0.flatten(), x1.flatten()


plt.rcParams['figure.figsize'] = (15, 5)

truth = Gaussian2D(x_mean=3512, y_mean=4418, x_stddev=150, y_stddev=150,
                   theta=20, amplitude=100)
mexp = truth(x0, x1).reshape(shape)
merr = abs(np.random.poisson(mexp) - mexp)

plt.subplot(1, 3, 1)
plt.imshow(mexp, origin='lower', cmap='viridis',
           extent=(x0low, x0high, x1low, x1high),
           interpolation='nearest', aspect='auto')
plt.title("True")
plt.subplot(1, 3, 2)
plt.imshow(merr, origin='lower', cmap='viridis',
           extent=(x0low, x0high, x1low, x1high),
           interpolation='nearest', aspect='auto')
plt.title("Noise")
plt.subplot(1, 3, 3)
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

fitmo = sfit(fitmo, x0.flatten(), x1.flatten(), mexp.flatten()+merr.flatten(), xbinsize=np.ones(x0.size)*dx, ybinsize=np.ones(x1.size)*dx, err=merr.flatten()+np.random.uniform(-0.5, 0.5, x0.size))


plt.subplot(1, 2, 1)
plt.imshow(fitmo(x0, x1).reshape(shape), origin='lower', cmap='viridis',
           extent=(x0low, x0high, x1low, x1high),
           interpolation='nearest', aspect='auto')
plt.title("Fit Model")

res = (mexp + merr) - fitmo(x0, x1).reshape(shape)
plt.subplot(1, 2, 2)
plt.imshow(res, origin='lower', cmap='viridis',
           extent=(x0low, x0high, x1low, x1high),
           interpolation='nearest', aspect='auto')
plt.title("Residuals")


plt.savefig("_generated/example_plot_2d_fit.png")
plt.close("all")

from astropy.modeling.models import Polynomial1D

x = np.arange(0, 10, 0.1)
y = 2+3*x**2+0.5*x
sfit = SherpaFitter(statistic="Cash")
print(sfit(Polynomial1D(2), x, y))

sampler = sfit.get_sampler()


def lognorm(x):
    # center on 10^20 cm^2 with a sigma of 0.5
    sigma = 0.5
    x0 = 1
    # nH is in units of 10^-22 so convert
    dx = np.log10(x) - x0
    norm = sigma / np.sqrt(2 * np.pi)
    return norm * np.exp(-0.5*dx*dx/(sigma*sigma))

sampler.set_prior("c0", lognorm)
_ = sampler(20000)


def plotter(xx, yy, c):
    px = []
    py = []
    for (xlo, xhi), y in zip(zip(xx[:-1], xx[1:]), yy):

        px.extend([xlo, xhi])
        py.extend([y, y])
    plt.plot(px, py, c=c)


def plot_hist(sampler, pname, nbins, c="b"):
    yy, xx = np.histogram(sampler.parameters[pname][sampler.accepted], nbins)
    plotter(xx, yy, c)
    plt.axvline(sampler.parameter_map[pname].val, c=c)

plt.figure(figsize=(3.2, 6))


plt.subplot(311)
plot_hist(sampler, 'c0', 100, 'k')
plt.text(0.1, 350, "c0")
plt.subplot(312)
plot_hist(sampler, 'c1', 100, 'r')
plt.text(-2.9, 350, "c2")
plt.ylabel("Number of accepted fits")
plt.subplot(313)
plot_hist(sampler, 'c2', 100, 'b')
plt.text(2.61, 300, "c3")
plt.xlabel("Parameter value")
plt.subplots_adjust(left=0.2)
plt.savefig("_generated/example_plot_mcmc_hist.png")
plt.close("all")


def plot_cdf(sampler, pname, nbins, c="b", sigfrac=0.682689):
    y, xx = np.histogram(sampler.parameters[pname][sampler.accepted], nbins)
    cdf = [y[0]]
    for yy in y[1:]:
        cdf.append(cdf[-1]+yy)
    cdf = np.array(cdf)
    cdf = cdf / float(cdf[-1])

    plotter(xx, cdf, c)
    plt.axvline(sampler.parameter_map[pname].val, c=c)
    med_ind = np.argmin(abs(cdf-0.5))
    plt.axvline((xx[med_ind]+xx[med_ind+1])/2, ls="--", c=c)
    siglo = (1-sigfrac)/2.0
    sighi = (1+sigfrac)/2.0
    lo_ind = np.argmin(abs(cdf-siglo))
    hi_ind = np.argmin(abs(cdf-sighi))
    plt.axvline((xx[lo_ind]+xx[lo_ind+1])/2, ls="--", c=c)
    plt.axvline((xx[hi_ind]+xx[hi_ind+1])/2, ls="--", c=c)

plt.figure(figsize=(3, 6))

plt.subplot(311)
plot_cdf(sampler, 'c0', 100, 'k')
plt.text(0.1, 0.89, "c0")
plt.subplot(312)
plot_cdf(sampler, 'c1', 100, 'r')
plt.text(-2.9, 0.89, "c1")
plt.ylabel("CDF")
plt.subplot(313)
plot_cdf(sampler, 'c2', 100, 'b')
plt.text(2.61, 0.89, "c2")
plt.xlabel("Parameter value")

plt.subplots_adjust(left=0.2)
plt.savefig("_generated/example_plot_mcmc_cdf.png")
plt.close("all")


print("Done")
