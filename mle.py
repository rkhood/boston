'''
fit line by maximum likelihood estimation
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def log_likelihood(theta, x, y, lam):

        n = len(x)
        f = lambda x: theta.dot(x.T)
        log_l = -0.5 * np.sum((y-f(x))**2) / n - lam * abs(theta).sum()

        return log_l


def fit(x, y, lam=1):

        x = np.c_[np.ones(x.shape[0]), x]
        nll = lambda *args: -1 * log_likelihood(*args)
        result = optimize.minimize(nll, np.ones(x.shape[1]), args=(x, y, lam))

        return result.x


def predict(w, x):

        x = np.c_[np.ones(x.shape[0]), x]
        return w.dot(x.T)
