import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from loss import KLDiv_gaussian


class StochasticLayer(nn.Module):
    def __init__(self, weights_size, bias=True):
        super().__init__()
        self.weights_size = weights_size
        self.bias = bias

        self.mu = nn.Parameter(torch.ones(weights_size))
        self.logvar = nn.Parameter(torch.zeros(weights_size))
        self.layer = nn.Parameter(torch.zeros(weights_size))
        self.b_mu = nn.Parameter(torch.zeros(weights_size[0])) if bias else None
        self.b_logvar = nn.Parameter(torch.zeros(weights_size[0])) if bias else None
        self.b_layer = nn.Parameter(torch.zeros(weights_size[0])) if bias else None

        self._init_mu()
        self._init_logvar()

        self.stdev_xi = None
        self.b_stdev_xi = None

    def _init_mu(self):
        n = self.mu.size(1)
        stdev = math.sqrt(1./n)
        # self.mu.data.zero_()
        self.mu.data.uniform_(-stdev, stdev)
        if self.bias:
            # self.b_mu.data.zero_()
            self.b_mu.data.uniform_(-stdev, stdev)

    def _init_logvar(self, logvar=-2, b_logvar=-2):
        self.logvar.data.zero_()
        self.logvar.data += logvar
        if self.bias:
            self.b_logvar.data.zero_()
            self.b_logvar.data += b_logvar

    def init_xi(self):
        stdev = torch.exp(0.5 * self.logvar)
        xi = stdev.data.new(stdev.size()).normal_(0, 1)
        self.stdev_xi = stdev * xi
        if self.bias:
            b_stdev = torch.exp(0.5 * self.b_logvar)
            b_xi = b_stdev.data.new(b_stdev.size()).normal_(0, 1)
            self.b_stdev_xi = b_stdev * b_xi

    def init_get_layer_mag(self, mag_input):
        layer = self.mu + self.stdev_xi
        r = torch.norm(layer)
        if self.bias:
            b_layer = self.b_mu + self.b_stdev_xi
            b_r = torch.norm(b_layer)
            return r*mag_input + b_r
        return r*mag_input

    def init_as_base(self, scale=None):
        if scale is None:
            scale = 1.
        layer = self.mu + self.stdev_xi
        self.layer.copy_(layer * scale)
        if self.bias:
            b_layer = self.b_mu + self.b_stdev_xi
            self.b_layer.copy_(b_layer * scale)

    def init_as_base_project(self, scale=None):
        if scale is not None:
            self.layer.copy_(self.layer * scale)
            if self.bias:
                self.b_layer.copy_(self.b_layer * scale)

    def forward(self, x, scale=None):
        layer = self.layer.clone()
        layer = layer * scale if scale is not None else layer

        if self.bias:
            b_layer = self.b_layer.clone()
            b_layer = b_layer * scale if scale is not None else b_layer
        else:
            b_layer = None

        out = self.operation(x, layer, b_layer)
        return out

    def operation(self, x, weight, bias):
        raise NotImplementedError

    def get_layer_mag_new(self):
        if self.bias:
            return (self.layer.clone(), self.b_layer.clone())
        return (self.layer.clone(), )

    def get_layer_mag(self):
        layer = self.layer.clone()
        if self.bias:
            b_layer = self.b_layer.clone()
            layer = torch.cat((layer.flatten(), b_layer.flatten()), 0)

        # layer = self.layer.clone()
        # r = torch.norm(layer)
        # if self.bias:
        #     b_layer = self.b_layer.clone()
        #     b_r = torch.norm(b_layer)
        #     return r*mag_input + b_r
        # return r*mag_input

        return torch.norm(layer)

    def to_str(self, is_base=True):
        device = torch.device('cpu')
        print("mu", self.mu.data.flatten()[:5].to(device).numpy())
        # return
        # if is_base:
        #     print("layer", self.layer.data.flatten()[:5].to(device).numpy())
        # else:
        #     print("mu", self.mu.data.flatten()[:5].to(device).numpy())
        #     print("logvar", self.logvar.data.flatten()[:5].to(device).numpy())

    def calc_kl_div(self, prior):
        mu1 = self.mu
        logvar1 = self.logvar
        mu2 = prior.mu.clone().detach()
        logvar2 = prior.logvar.clone().detach()
        kl_div = KLDiv_gaussian(mu1, logvar1, mu2, logvar2, var_is_logvar=True)

        if self.bias:
            b_mu1 = self.b_mu
            b_logvar1 = self.b_logvar
            b_mu2 = prior.b_mu.clone().detach()
            b_logvar2 = prior.b_logvar.clone().detach()
            kl_div += KLDiv_gaussian(b_mu1, b_logvar1, b_mu2, b_logvar2, var_is_logvar=True)

        return kl_div

    def add_sub_mean(self, model, add, lr):
        if add is None:
            self.mu = nn.Parameter(self.mu * lr)
            return

        mu1 = self.mu
        mu2 = lr*model.mu
        if add:
            self.mu = nn.Parameter(mu1 + mu2)
        else:
            self.mu = nn.Parameter(mu1 - mu2)

        if self.bias:
            b_mu1 = self.b_mu
            b_mu2 = lr*model.b_mu
            if add:
                self.b_mu = nn.Parameter(b_mu1 + b_mu2)
            else:
                self.b_mu = nn.Parameter(b_mu1 - b_mu2)


class StochasticLinear(StochasticLayer):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__((output_dim, input_dim), bias=bias)

    def operation(self, x, weight, bias):
        return F.linear(x, weight, bias)


class NotStochasticLinear(StochasticLinear):
    def init_xi(self):
        self.stdev_xi = 0
        self.b_stdev_xi = 0


class MyConv2d(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.w = nn.Parameter(torch.ones(*param[:4]))
        self.b = nn.Parameter(torch.zeros(param[0]))
        torch.nn.init.kaiming_normal_(self.w)

    def forward(self, x):
        param = self.param
        x = F.conv2d(x, self.w, self.b, stride=param[4], padding=param[5])
        return x


class StochasticConv2d(StochasticLayer):
    def __init__(self, weights_size, stride=1, padding=0, bias=True):
        super().__init__(weights_size, bias=bias)
        self.stride = stride
        self.padding = padding

    def _init_mu(self):
        torch.nn.init.kaiming_normal_(self.mu)

    def operation(self, x, weight, bias):
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)


class NotStochasticConv2d(StochasticConv2d):
    def init_xi(self):
        self.stdev_xi = 0
        self.b_stdev_xi = 0


class BoundedStochasticModel(nn.Module):
    def __init__(self, radius, mag_input=1):
        super().__init__()
        if mag_input < 1:  # special considerations, appendix D
            mag_input = 1

        self.names = ()
        self.mag_input = torch.tensor(mag_input, dtype=torch.float)  # maximum magnitude of the input
        self.max_radius = torch.tensor(radius, dtype=torch.float) if radius is not None else None

    def forward(self, x):
        raise NotImplementedError()

    def get_scale(self):
        R = self.max_radius
        if R is not None:
            r = self.get_model_mag()
            return R / torch.max(R, r)
        else:
            return None

    def init_as_base(self):
        for name, layer in self.named_modules():
            if name in self.names:
                layer.init_xi()

        R = self.max_radius
        if R is not None:
            r = self.mag_input
            for name, layer in self.named_modules():
                if name in self.names:
                    r = layer.init_get_layer_mag(r)
            scale = R / torch.max(R, r)
        else:
            scale = None

        for name, layer in self.named_modules():
            if name in self.names:
                layer.init_as_base(scale)

    def get_model_mag(self):
        r = self.mag_input
        theta = torch.tensor([])
        for name, layer in self.named_modules():
            if name in self.names:
                lst = layer.get_layer_mag_new()
                for item in lst:
                    theta = torch.cat((theta, item.flatten()), 0)

        return r * torch.norm(theta)

    def to_str(self, *args, **kwargs):
        for name, layer in self.named_modules():
            if name in self.names:
                layer.to_str(*args, **kwargs)

    def init_logvar(self, *args, **kwargs):
        for name, layer in self.named_modules():
            if name in self.names:
                layer._init_logvar(*args, **kwargs)

    def init_mu(self, *args, **kwargs):
        for name, layer in self.named_modules():
            if name in self.names:
                layer._init_mu(*args, **kwargs)

    def calc_kl_div(self, prior, device=None):
        if device is not None:
            kl_div = torch.tensor(0., dtype=torch.float).to(device)
        else:
            kl_div = torch.tensor(0., dtype=torch.float)
        prior = prior.module
        for (name, layer), (prior_name, prior_layer) in zip(self.named_modules(), prior.named_modules()):
            if name in self.names:
                kl_div += layer.calc_kl_div(prior_layer)

        return kl_div

    def calc_addorsub(self, model, add, lr=1):
        model = model.module
        for (name, layer), (model_name, model_layer) in zip(self.named_modules(), model.named_modules()):
            if name in self.names:
                layer.add_sub_mean(model_layer, add, lr)

    def calc_ls_constants(self, device):
        L = torch.tensor(1., dtype=torch.float).to(device)
        S = torch.tensor(0., dtype=torch.float).to(device)
        for name, layer in self.named_modules():
            if name in self.names:
                lmag = layer.get_layer_mag()
                L = L*lmag
                S += L
        return L, S*L


class SSLayer(StochasticLayer):
    def __init__(self, weights_size, bias=True):
        super().__init__(weights_size, bias)
        self.eps = 1

    def forward(self, x, scale=None):
        if self.bias:
            b_var = torch.exp(self.b_logvar)
            bias_mean = self.b_mu
        else:
            b_var = None
            bias_mean = None

        out_mean = self.operation(x, self.mu, bias=bias_mean)

        w_var = torch.exp(self.logvar)
        out_var = self.operation(x.pow(2), w_var, bias=b_var)
        if self.eps > 0:
            noise = out_mean.data.new(out_mean.size()).normal_(0, self.eps)
        else:
            noise = 0

        layer_out = out_mean + noise * torch.sqrt(out_var)
        if scale is not None:
            layer_out *= scale

        return layer_out

    def get_layer_mag(self, mag_input):
        mu = self.mu.clone()
        r = torch.norm(mu)
        if self.bias:
            b_mu = self.b_mu.clone()
            b_r = torch.norm(b_mu)
            return r*mag_input + b_r
        return r*mag_input


class SSLinear(SSLayer):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__((output_dim, input_dim), bias=bias)

    def operation(self, x, weight, bias):
        return F.linear(x, weight, bias)
