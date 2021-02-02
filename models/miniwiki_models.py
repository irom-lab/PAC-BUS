import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression as Logit
from scipy.optimize import brentq
from models.layers import StochasticLinear as SLinear
from models.layers import NotStochasticLinear as Linear
from models.layers import BoundedStochasticModel


class SMiniWikiModel(BoundedStochasticModel):
    def __init__(self, input_dim, output_dim, radius):
        super().__init__(radius, mag_input=1)
        self.names = ('layer1',)
        self.layer1 = SLinear(input_dim, output_dim, bias=False)

    def forward(self, x):
        scale = self.get_scale()
        return self.layer1(x, scale)


class MiniWikiModel(BoundedStochasticModel):
    def __init__(self, input_dim, output_dim, radius):
        super().__init__(radius, mag_input=1)
        self.names = ('layer1',)
        self.layer1 = Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        scale = self.get_scale()
        return self.layer1(x, scale)


class LogReg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogReg, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class ConstrainedLogit(Logit):
    def __init__(self, radius=1.0, interval=[-4, 4], **kwargs):
        self.radius = radius
        self.interval = interval
        self.kwargs = kwargs

    def logit(self, logC, X, Y):
        super().__init__(C=10 ** logC, fit_intercept=False, **self.kwargs, max_iter=1000)
        super().fit(X, Y)
        return np.linalg.norm(self.coef_) - self.radius

    def fit(self, X, Y):
        brentq(self.logit, *self.interval, args=(X, Y), full_output=True)

