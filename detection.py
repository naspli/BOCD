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

from hyperparameter import GaussianKnownMean
from scipy import stats
import numpy as np

EPS = 0.0001  # how to elegantly deal with infinity?


class Detector:

    def __init__(self, bcon, l, hyper):
        self.bcon = bcon
        self.l = l
        self.hyper = hyper

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
        r_pdf = []
        r = 0
        while self.boundary(r) > EPS:
            r_pdf.append(self.boundary(r))
            r += 1
        return np.array(r_pdf)

    def run(self, X):

        # 1. Initialize
        r_pdf = self.r_initial()
        r_pdf_series = [r_pdf]

        # 2. Observe New Datum (change to a while loop for a data-stream)
        for x in X:

            # 3. Evaluate Predictive Probability
            predict = self.hyper.predictive(x)

            # 4. Calculate Growth Probabilities
            elem_mult = r_pdf * predict  # element-wise array multiplication
            growth = elem_mult * (1-1/self.l)

            # 5. Calculate Changepoint Probabilities
            changepoint = np.sum(elem_mult/self.l)

            # 6. Calculate Total Evidence
            evidence = changepoint + np.sum(growth)

            # 7. Determine Run Length Distribution
            r_pdf = np.append(changepoint, growth)/evidence
            r_pdf_series.append(r_pdf)

            # 8. Update Sufficient Statistics
            self.hyper.nu_update()
            self.hyper.chi_update(x)

            # 9. Perform Prediction
            # leave until the end


class Parser:
    pass

if __name__ == "__main__":
    print("Running BOCD Demo...")
    # X    = Parser( "dow-jones-example.dat" )
    # gkm  = GaussianKnownMean( 0 )
    # bocd = Detector( True, 250, gkm )
    # bocd.run(X)
    print("Done! (Incomplete)")