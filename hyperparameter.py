# Collection of hyperparameter classes to be used by detection.py

# from scipy import stats
import numpy as np


class Hyper:
    # parent class-structure for all hyperparameter objects

    def __init__(self):
        # effective number of observations of the prior
        self.nu = np.array([self.nu_prior()], dtype=int)
        # how this contributes to the sufficient statistic
        self.chi = np.array([self.chi_prior()], dtype=float)

    def nu_update(self):
        self.nu = np.append(self.nu, self.nu[-1] + 1)

    # x is most recent data-point
    def chi_update(self, x):
        self.chi = np.append(self.chi, self.chi[-1] + self.suff_stat(x))

    def predictive(self, x):
        return None

    @staticmethod
    def nu_prior():
        return None

    @staticmethod
    def chi_prior():
        return None

    @staticmethod
    def suff_stat(x):
        return None


class GaussianKnownMean(Hyper):
    # hyper-parameter methods for Gaussian-Known-Mean distribution
    # (currently empty)

    def __init__(self, mean):
        Hyper.__init__(self)
        self.mu = mean

    def predictive(self, x):
        return None

    @staticmethod
    def nu_prior():
        return None

    @staticmethod
    def chi_prior():
        return None

    @staticmethod
    def suff_stat(x):
        return None
