import numpy as np
from nearest_neighbors import KNNClassifier


def kfold(n, n_folds):
    ind = np.arange(1, n_folds)*round(n / n_folds)
    m = np.arange(n)
    prev = 0
    rez = []
    val = np.split(m, ind)
    ind = np.append(ind, n)
    for i in range(len(ind)):
        rez += [tuple([np.hstack([m[:prev], m[ind[i]:]]), val[i]])]
        prev = ind[i]
    return rez


def accuracy(a, b):
    return np.sum(a == b) / b.size


def knn_cross_val_score(X, y, k_list, score='accuracy', cv=None, **kwargs):
    if cv is None:
        cv = kfold(X.shape[0], 5)
    dct = {}
    for k in k_list:
        dct[k] = np.array([])
    model = KNNClassifier(k=k_list[-1], **kwargs)
    for train, test in cv:
        model.fit(X[train], y[train])
        model.k = k_list[-1]
        model.cv = model.find_kneighbors(X[test], model.weights)
        for k_val in k_list[::-1]:
            model.k = k_val
            y_pred = model.predict(X[test])
            dct[k_val] = np.append(dct[k_val], accuracy(y_pred, y[test]))
    model.cv = None
    return dct
