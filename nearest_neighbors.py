import numpy as np
from sklearn.neighbors import NearestNeighbors
from distances import euclidean_distance
from distances import cosine_distance


class KNNClassifier:
    eps = 1e-5

    def __init__(self, k=5, strategy='brute', metric='euclidean', weights=True,
                 test_block_size=100):
        self.k = k
        self.cv = None
        self.strategy = strategy
        if metric == 'euclidean':
            self.metric = euclidean_distance
        else:
            self.metric = cosine_distance
        self.weights = weights
        self.test_block_size = test_block_size
        if strategy != 'my_own':
            self.knn_sklrn = NearestNeighbors(n_neighbors=self.k,
                                              algorithm=self.strategy,
                                              metric=metric)

    def fit(self, X, y):
        if self.strategy != 'my_own':
            self.knn_sklrn.fit(X, y)
        else:
            self.objs = np.array(X)
        self.classes = np.unique(y)
        self.y = np.array(y)

    def find_kneighbors(self, X, return_distance):
        if X.shape[0] <= self.test_block_size:
            if self.strategy != 'my_own':
                return self.knn_sklrn.kneighbors(X, self.k, return_distance)
            else:
                dst = self.metric(X, self.objs)
                ind = np.argsort(dst)[:, :self.k]
                if return_distance is False:
                    return ind
                dist = np.zeros((ind.shape[0], self.k))
                for i in range(ind.shape[0]):
                    dist[i] = dst[i][ind[i]]
                return dist, ind
        else:
            t_s = self.test_block_size
            if return_distance:
                if self.strategy != 'my_own':
                    dst, ind = self.knn_sklrn.kneighbors(X[:t_s], self.k, True)
                    rec_d, rec_i = self.find_kneighbors(X[t_s:], True)
                else:
                    dst, ind = self.find_kneighbors(X[:t_s], True)
                    rec_d, rec_i = self.find_kneighbors(X[t_s:], True)
                dst = np.vstack([dst, rec_d])
                ind = np.vstack([ind, rec_i])
                return dst, ind
            else:
                if self.strategy != 'my_own':
                    ind = self.knn_sklrn.kneighbors(X[:t_s], self.k, False)
                    rec_i = self.find_kneighbors(X[t_s:], False)
                else:
                    ind = self.find_kneighbors(X[:t_s], False)
                    rec_i = self.find_kneighbors(X[t_s:], False)
                return np.vstack([ind, rec_i])

    def predict(self, X):
        if self.cv is None:
            k_neigh = self.find_kneighbors(X, self.weights)
        else:
            if self.weights:
                k_neigh = self.cv[0][:, :self.k], self.cv[1][:, :self.k]
            else:
                k_neigh = self.cv[:, :self.k]
        if self.weights:
            g = (self.classes[0] == self.y[k_neigh[1]]) /\
                (k_neigh[0] + self.eps)
            g = np.sum(g, axis=1)[:, np.newaxis]
            for c in self.classes[1:]:
                mask = self.y[k_neigh[1]] == c
                d = mask / (k_neigh[0] + self.eps)
                g = np.hstack([g, np.sum(d, axis=1)[:, np.newaxis]])
            return self.classes[np.argmax(g, axis=1)]
        else:
            m2 = []
            for с in self.classes:
                m2 += [self.y[k_neigh] == с]
            m2 = np.array(m2)
            m2 = m2.reshape(self.classes.size, X.shape[0], self.k)
            arr = np.sum(m2, axis=(2)).T
            return self.classes[np.argmax(arr, axis=1)]
