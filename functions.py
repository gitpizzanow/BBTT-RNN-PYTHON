import numpy as np

def CLIP(x):
    return np.clip(x, -5.0, 5.0)

def Loss(y, yhat):
    return (y - yhat)**2

def mean_squared_error(y_true, y_pred):
    return np.mean([(yt - yp)**2 for yt, yp in zip(y_true, y_pred)])

def mean_absolute_error(y_true, y_pred):
    return np.mean([np.abs(yt - yp) for yt, yp in zip(y_true, y_pred)])

def regression_accuracy(y_true, y_pred, tolerance=0.1):
    correct = 0
    for yt, yp in zip(y_true, y_pred):
        if np.abs(yt - yp) < tolerance:
            correct += 1
    return correct / len(y_true)
