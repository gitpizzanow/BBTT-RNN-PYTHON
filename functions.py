import numpy as np


def CLIP(x):
    return np.clip(x,-5.0,5.0)


def Loss(y , yhat):
    return (y - yhat)**2
