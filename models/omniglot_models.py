import torch.nn as nn
from models.layers import StochasticLinear as SLinear
from models.layers import StochasticConv2d as SConv2d
from models.layers import NotStochasticLinear as Linear
from models.layers import NotStochasticConv2d as Conv2d
from models.layers import BoundedStochasticModel


class OmniglotModel(BoundedStochasticModel):
    def __init__(self, n_way, n_filt=64, linear=Linear, conv=Conv2d, activation=nn.ReLU):
        super().__init__(radius=None)
        self.names = ('conv1', 'conv2', 'conv3', 'conv4', 'fc_out')
        self.conv1 = Conv2d((n_filt, 1, 3, 3), 2, 1)
        self.conv2 = Conv2d((n_filt, n_filt, 3, 3), 2, 1)
        self.conv3 = conv((n_filt, n_filt, 3, 3), 1, 0)
        self.conv4 = conv((n_filt, n_filt, 3, 3), 1, 0)
        self.relu = activation(inplace=True)
        self.bn = nn.BatchNorm2d(n_filt)
        self.fc_out = linear(n_filt, n_way)

    def forward(self, x):
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))
        x = self.relu(self.bn(self.conv3(x)))
        x = self.relu(self.bn(self.conv4(x)))
        x = x.mean(dim=[2, 3])
        x = self.fc_out(x)
        return x


class SOmniglotModel(OmniglotModel):
    def __init__(self, n_way, n_filt=64, ELU=False):
        if ELU:
            activation = nn.ReLU
        else:
            activation = nn.ELU
        super().__init__(n_way, n_filt=n_filt, linear=SLinear, conv=SConv2d, activation=activation)


class OmniglotModel1(nn.Module):
    def __init__(self, n_way, n_filt=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, n_filt, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(n_filt, n_filt, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(n_filt, n_filt, kernel_size=3)
        self.conv4 = nn.Conv2d(n_filt, n_filt, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(n_filt)
        self.fc_out = nn.Linear(n_filt, n_way)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, target):
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))
        x = self.relu(self.bn(self.conv3(x)))
        x = self.relu(self.bn(self.conv4(x)))
        x = x.mean(dim=[2, 3])
        x = self.fc_out(x)
        loss = self.criterion(x, target)
        return loss, x

