import numpy as np


def euclidean_distance(X, Y):
    a = (np.sum(X**2, axis=1) + np.sum(Y**2, axis=1)[:, np.newaxis]).T
    return np.sqrt(a - 2*np.dot(X, Y.T))


def cosine_distance(X, Y):
    nrm = np.sqrt(np.sum(X**2, axis=1) * np.sum(Y**2, axis=1)[:, np.newaxis]).T
    return 1 - np.dot(X, Y.T) / nrm
