"""
k-means
参考: https://qiita.com/navitime_tech/items/bb1bd01537bc2713444a

"""


import numpy as np

class KMEANS():
    def __init__(self, n_clusters, max_iter = 500, random_state = 0, convergence = True):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.convergence = convergence
        self.cluster_result = None

    def is_float(self, X):
        for i in range(len(X.columns)):
            if X[X.columns[i]].dtype != float:
                return False
        return True

    def fit(self, X):
        np.random.seed(self.random_state)
        clusters = np.random.randint(0, self.n_clusters, X.shape[0])

        self.cluster_result = self.update_convergence(X, clusters) if self.convergence else self.update(X, clusters)

    def predict(self):
        if len(self.cluster_result):
            return self.cluster_result
        else:
            print('先にfitを実行してください')

    def update_convergence(self, X, clusters):
        for i in range(self.max_iter):
            centroids = np.array([X[clusters == i].mean() for i in range(self.n_clusters)])

            new_clusters = np.array([np.linalg.norm(X - c, axis= 1) for c in centroids]).argmin(axis = 0)

            if np.allclose(clusters, new_clusters):
                break

        return new_clusters

    def update(self, X, clusters):
        for i in range(self.max_iter):
            centroids = np.array([X[clusters == i].mean() for i in range(self.n_clusters)])

            clusters = np.array([np.linalg.norm(X - c, axis= 1) for c in centroids]).argmin(axis = 0)

        return clusters
