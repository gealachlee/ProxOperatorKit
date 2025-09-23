import numpy as np
from sklearn.preprocessing import normalize


def generat_sensing_mat(m, n):
    A = np.random.normal(size=(m, n))
    return normalize(A, norm='l2', axis=0)


