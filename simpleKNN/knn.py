import numpy as np
import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        distances = euclidean_dist_squared(self.X, X_hat)
        sorted_indices = np.argsort(distances, axis=0)
        voting_labels = self.y[sorted_indices[:self.k, :]]
        y_pred = np.apply_along_axis(utils.mode, 0, voting_labels)

        return y_pred




