import numpy as np
from tqdm import tqdm

class LinearRegression():
    def __init__(self):
        self.lr = 0.001
        self.n_iter = 1000
        self.patience = 10

        self.weights = None
        self.bias = None

        self.best_weights = None
        self.best_bias = None
        self.least_loss = float("inf")

    def rmse(self, y_pred, y_true):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn(1)

        no_improve_count = 0
        for _ in tqdm(range(self.n_iter)):
            # calculate y_pred
            y_pred = np.dot(X, self.weights) + self.bias
            
            # compute loss
            loss = self.rmse(y_pred, y)

            # Save best weights and bias
            if loss < self.least_loss:
                self.least_loss = loss
                self.best_weights = self.weights.copy()
                self.best_bias = self.bias.copy()
                no_improve_count = 0  # reset
            else:
                no_improve_count += 1

            # Early stopping
            if no_improve_count >= self.patience:
                print(f"Early stopping at iteration {_} with best loss: {self.least_loss:.4f}")
                break
            # Compute Gradients
            del_w = (1/n_samples) * np.dot(X.T, (y_pred - y))
            del_b = (1/n_samples) * np.sum(y_pred-y)

            # update weights and bias
            self.weights -= self.lr*del_w
            self.bias -= self.lr*del_b

        self.weights = self.best_weights
        self.bias = self.best_bias

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias