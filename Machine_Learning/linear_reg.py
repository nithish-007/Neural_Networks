import numpy as np
from tqdm import tqdm

class LinearRegression():
    def __init__(self):
        self.lr = 0.001
        self.n_iter = 500
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn(1)

        for _ in tqdm(range(self.n_iter)):
            # calculate y_pred
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute Gradients
            del_w = (1/n_samples)*np.dot(X.T, (y_pred - y))
            del_b = (1/n_samples)*np.sum(y_pred-y)

            # update weights and bias
            self.weights = self.weights - self.lr*del_w
            self.bias = self.bias - self.lr*del_b

    def predict(self, X):
        return np.dot(X, self.weights)+self.bias