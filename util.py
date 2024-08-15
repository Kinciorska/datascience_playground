import numpy as np


def sigmoid(x, beta_1=1, beta_2=0):
    # return 1 / (1 + np.exp(-x))
    return 1 / (1 + np.exp(-beta_1 * (x - beta_2)))
