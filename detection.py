# Detector is built to find changepoints in a dataset
#
# The object has parameters:
# bcon:  boundary condition. A changepoint either occurred
#        just before the data (True) or is modeled using the
#        a-priori gap dist (False)
# l:     lambda, the timescale of the a-priori gap dist,
#        which we assume to be geometric. This gives
#        constant "hazard function" 1/lambda
# hyper: hyper-parameter container object

from datetime import datetime
from hyperparameter import GaussianKnownMean, NaturalGaussianKnownVariance
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

EPS = 0.0001  # how to elegantly deal with infinity?


class Detector:

    def __init__(self, bcon, l, hyper):
        self.bcon = bcon
        self.l = l
        self.hyper = hyper
        self.r = []
        self.prediction = []

    # r0 is the initial run length, output is a probability
    def boundary(self, r0):
        if self.bcon:
            return 1 if r0 == 0 else 0
        p = 1/self.l
        dist = stats.geom(p)
        s = 1-dist.cdf(r0+1)
        z = (1-p)/p  # normalising const found analytically
        return s/z

    def r_initial(self):
        # 1.
        print("Initialize")
        r_pdf = []
        r = 0
        while self.boundary(r) > EPS:
            r_pdf.append(self.boundary(r))
            r += 1
        print(r_pdf)
        if len(r_pdf) > 1:
            self.hyper.extend_boundary(len(r_pdf))
        r_pdf = np.array(r_pdf)
        self.r = [r_pdf]
        return r_pdf

    def new_value(self, x):
        # 2.
        print("Observe New Datum")
        print(x)
        r_pdf = self.r[-1]

        # 3.
        print("Evaluate Predictive Probability")
        predict = self.hyper.predictive(x)
        print(predict)

        # 4. Calculate Growth Probabilities
        elem_mult = r_pdf * predict  # element-wise array multiplication
        growth = elem_mult * (1-1/self.l)

        # 5. Calculate Changepoint Probabilities
        changepoint = np.sum(elem_mult/self.l)

        # 6. Calculate Total Evidence
        evidence = changepoint + np.sum(growth)

        # 7.
        print("Determine Run Length Distribution")
        r_pdf = np.append(changepoint, growth)/evidence
        self.r.append(r_pdf)
        print(r_pdf)

        # 8. Update Sufficient Statistics
        self.hyper.update()

        # 9. Perform Prediction
        # self.prediction += np.sum(predict*r_pdf)

        return r_pdf


class Visualiser(Detector):

    def __init__(self, bcon, l, hyper, ax, X):
        Detector.__init__(self, bcon, l, hyper)
        self.line, = ax.plot([], [], 'k-')
        self.ax = ax
        self.X = X

        # Set up plot parameters
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.grid(True)

        # This vertical line represents the theoretical value, to
        # which the plotted distribution should converge.
        # self.ax.axvline(prob, linestyle='--', color='black')

    def init(self):
        self.line.set_data([], [])
        self.r_initial()
        return self.line,

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i == 0:
            return self.init()

        # Choose success based on exceed a threshold with a uniform pick
        y = self.new_value(X[i])
        r = np.arange(y.shape[0])
        self.ax.set_xlim(0, i)
        self.ax.set_ylim(0, max(y))
        self.line.set_data(r, y)
        return self.line,


def parser(filepath):
    dt = np.dtype([('date', datetime), ('x', float)])
    data = np.loadtxt(filepath, dtype=dt)
    return [x[1] for x in data]

if __name__ == "__main__":
    print("Running BOCD Demo...")
    X = parser("watergate-djia.dat")
    gkm = NaturalGaussianKnownVariance(0.00001)
    # bocd = Detector(True, 250, gkm)
    # bocd.r_initial()
    # for x in X:
    #     bocd.new_value(x)
    fig, ax = plt.subplots()
    vis = Visualiser(True, 250, gkm, ax, X)
    anim = FuncAnimation(fig, vis, frames=len(X), init_func=vis.init,
                         interval=100, blit=True)
    plt.show()
    print("Done!")