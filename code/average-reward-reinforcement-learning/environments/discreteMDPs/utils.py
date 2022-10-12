import numpy as np

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()




class Dirac:
    def __init__(self, value):
        self.v = value

    def rvs(self):
        return self.v

    def mean(self):
        return self.v