def CLIP(x):
    return 5.0 if x>5.0 else -0.5 if x<0.5 else x


def Loss(y , yhat):
    return (y - yhat)**2
