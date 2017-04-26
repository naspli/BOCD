# Collection of hyperparameter classes to be used by detection.py

from abc import ABCMeta, abstractmethod
from scipy.special import gamma
import numpy as np
import math

EPS = 0.0001


class Hyper:
    # parent class-structure for all hyperparameter objects
    __metaclass__ = ABCMeta

    def __init__(self):
        # effective number of observations of the prior
        self.nu = np.array([1], dtype=int)
        self.nu_next = self.nu
        # how this contributes to the sufficient statistic
        self.chi = np.array([self.chi_prior()], dtype=float)
        self.chi_next = self.chi

    def extend_boundary(self, r_len):
        self.nu = np.array([self.nu[0] for _ in r_len])
        self.chi = np.array([self.chi[0] for _ in r_len])

    def adjust(self, x):
        self.nu_next = np.append([1], self.nu + 1)
        self.chi_next = np.append([self.chi_prior()], self.chi + self.suff_stat(x))

    def update(self):
        self.nu = self.nu_next
        self.chi = self.chi_next

    # default prior for the hyperparameter
    @abstractmethod
    def chi_prior(self):
        pass

    # the sufficient statistic is an updating function for the hyperparameter
    @abstractmethod
    def suff_stat(self, x):
        pass

    # predictive probability
    # !!! I still haven't got this giving sensible answers for any distribution
    @abstractmethod
    def predictive(self, x):
        pass


class GaussianKnownMean(Hyper):
    # hyper-parameter methods for Gaussian-Known-Mean distribution

    def __init__(self, mu):
        Hyper.__init__(self)
        self.mu = mu

    def chi_prior(self):
        return np.array([1, 1])

    # the sufficient statistic is an updating function for the hyperparameter
    def suff_stat(self, x):
        a = 0.5
        b = self.nu * (x-self.mu)**2/(2*(self.nu+1))
        return np.array([a, b])

    # predictive probability
    def predictive(self, x):
        pass

class NaturalHyper(Hyper):
    # parent class-structure for natural hyperparameter objects
    # a general predictive technique can be used if we can derive f
    __metaclass__ = ABCMeta

    def __init__(self):
        Hyper.__init__(self)

    def predictive(self, x):
        self.adjust(x)
        predict = np.array([], dtype=float)
        for r in range(len(self.nu)):
            f0 = self.f(self.chi[r], self.nu[r])
            f1 = self.f(self.chi_next[r+1], self.nu_next[r+1])
            predict = np.append(predict, self.h(x)*f0/f1)
        return predict

    # h is the normalising function of the exponential model
    # it may or may not be a function of x
    @abstractmethod
    def h(self, x):
        pass

    # f is the normalising function of the conj-prior in terms of chi and nu
    @staticmethod
    @abstractmethod
    def f(chi, nu):
        pass


class NaturalGaussianKnownMean(NaturalHyper):
    # hyper-parameter methods for Gaussian-Known-Mean distribution

    def __init__(self, mu):
        Hyper.__init__(self)
        self.mu = mu

    def chi_prior(self):
        return EPS

    def h(self, x):
        return 1/math.sqrt(2*math.pi)

    # derived analytically using other results
    @staticmethod
    def f(chi, nu):
        return math.sqrt((chi**(nu+2))/(2**nu))/gamma(nu/2+1)

    def suff_stat(self, x):
        return (x-self.mu)**2


class NaturalGaussianKnownVariance(NaturalHyper):
    # hyper-parameter methods for Gaussian-Known-Variance distribution

    def __init__(self, var):
        Hyper.__init__(self)
        self.var = var

    def chi_prior(self):
        return 0

    def h(self, x):
        return math.exp(-x**2/(2*self.var))/math.sqrt(2*math.pi*self.var)

    # derived analytically using other results
    @staticmethod
    def f(chi, nu):
        return 1/(math.exp(chi**2/(2*nu))*math.sqrt(2*math.pi/nu))

    def suff_stat(self, x):
        return x/math.sqrt(self.var)
