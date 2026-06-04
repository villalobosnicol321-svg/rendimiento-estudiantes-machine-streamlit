"""Clasificador PCA + K-Means (sección 13 del notebook)."""
import numpy as np
from scipy.special import softmax as scipy_softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class PCAKMeansClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=10, n_clusters=6, random_state=42):
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.random_state = random_state

    def _to_dense(self, X):
        return X.toarray() if hasattr(X, "toarray") else np.array(X)

    def fit(self, X, y):
        X = self._to_dense(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        max_comp = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self.pca_ = PCA(n_components=max_comp, random_state=self.random_state)
        X_pca = self.pca_.fit_transform(X)
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        cluster_labels = self.kmeans_.fit_predict(X_pca)
        self.cluster_map_ = {}
        for c in range(self.n_clusters):
            mask = cluster_labels == c
            if mask.sum() > 0:
                counts = np.bincount(y[mask], minlength=len(self.classes_))
                self.cluster_map_[c] = int(counts.argmax())
            else:
                self.cluster_map_[c] = 0
        return self

    def predict(self, X):
        X = self._to_dense(X)
        X_pca = self.pca_.transform(X)
        labels = self.kmeans_.predict(X_pca)
        return np.array([self.cluster_map_[c] for c in labels])

    def predict_proba(self, X):
        X = self._to_dense(X)
        X_pca = self.pca_.transform(X)
        distances = self.kmeans_.transform(X_pca)
        clust_prob = scipy_softmax(-distances, axis=1)
        n_classes = len(self.classes_)
        class_prob = np.zeros((len(X), n_classes))
        for clust_idx, cls_idx in self.cluster_map_.items():
            class_prob[:, cls_idx] += clust_prob[:, clust_idx]
        row_sums = class_prob.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return class_prob / row_sums
