import numpy as np
import torch
import torch.nn as nn


# PyTorch version
class BatchNormalization(nn.Module):

    def __init__(self, num_features):
        super(BatchNormalization, self).__init__()
        self.num_features = num_features
        self.scale = nn.Parameter(torch.ones(num_features))
        self.shift = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + 1e-8)
        out = self.scale * x_norm + self.shift
        return out


# Numpy version
class BatchNormalization:

    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.beta = None
        self.gamma = None
        self.mean = None
        self.variance = None
        self.X_norm = None

    def forward(self, X):
        N, D = X.shape

        self.mean = np.mean(X, axis=0)
        self.variance = np.var(X, axis=0)
        X_norm = (X - self.mean) / np.sqrt(self.variance + self.epsilon)
        self.X_norm = X_norm
        # 缩放和平移：乘以缩放参数 gamma，再加上平移参数 beta
        out = self.gamma * X_norm + self.beta
        return out

    def backward(self, dout):
        N, D = dout.shape
        dgamma = np.sum(dout * self.X_norm, axis=0)
        dbeta = np.sum(dout, axis=0)
        dX_norm = dout * self.gamma

        dX = (1. / N) * (1. / np.sqrt(self.variance + self.epsilon)) * (
            N * dX_norm - np.sum(dX_norm, axis=0) -
            self.X_norm * np.sum(dX_norm * self.X_norm, axis=0))

        return dX, dgamma, dbeta

    def initialize_parameters(self, D):
        self.gamma = np.ones(D)
        self.beta = np.zeros(D)
