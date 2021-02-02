import torch
from models.layers import StochasticLinear as SLinear
from models.layers import NotStochasticLinear as Linear
from models.layers import BoundedStochasticModel
from models.layers import SSLinear


class CircleModel_Frame(BoundedStochasticModel):
    def __init__(self, radius):
        mag_input = 1
        super().__init__(radius, mag_input)
        self.names = ('layer1', 'out')

    def forward(self, x):
        elu = torch.nn.ELU()
        scale = self.get_scale()
        x = elu(self.layer1(x, scale))
        x = self.out(x, scale)
        return x


class CircleModel(CircleModel_Frame):
    def __init__(self, input_dim, output_dim, radius, hidden_neurons=50):
        super().__init__(radius)
        self.layer1 = Linear(input_dim, hidden_neurons, bias=True)
        self.out = Linear(hidden_neurons, output_dim, bias=True)


class SCircleModel(CircleModel_Frame):
    def __init__(self, input_dim, output_dim, radius, hidden_neurons=50):
        super().__init__(radius)
        self.layer1 = SLinear(input_dim, hidden_neurons, bias=True)
        self.out = SLinear(hidden_neurons, output_dim, bias=True)


class SSCircleModel(CircleModel_Frame):
    def __init__(self, input_dim, output_dim, radius, hidden_neurons=50):
        super().__init__(radius)
        self.layer1 = SSLinear(input_dim, hidden_neurons, bias=True)
        self.out = SSLinear(hidden_neurons, output_dim, bias=True)
