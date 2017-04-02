# BOCD
Python Implementation of Bayesian Online Changepoint Detection, as described by Adams &amp; McKay (2007) in its full generality.
Choose an input dataset, a conjugate-exponential model, and a few tuning parameters. The algorithm will keep the hyperparameters of the model updated according to the changepoints. At each time-step, the probability disribution of the run-length is calculated.

detection.py      
- the head file that contains a high-level implementation of the algorithm

hyperparameter.py 
- as the algorithm depends largely on the model, this file contains classes for various conjugate-exponential models. This is where the hyperparameters of the model are stored and updated.

well-log-example.dat
- test data
