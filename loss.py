import numpy as np
import torch


class ScaledCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    # assuming input is a probability distribution <- key assumption here
    def __init__(self, ncls, radius, convex=True):
        if ncls < 2:
            print("WARNING: Not enough classes for this Loss")
        super().__init__()

        self.k = torch.tensor(ncls, dtype=torch.float)
        self.radius = torch.tensor(radius, dtype=torch.float)
        self.scale = (1 / (self.calc_max_value() - self.calc_min_value()))
        self.shift = self.calc_min_value()
        self.convex = convex
        k = self.k
        if convex:
            if k == 2:
                self.S = radius * torch.tensor(np.sqrt(2/27), dtype=torch.float)
            self.L = radius * torch.sqrt(k-1)/k * self.scale
            self.S = radius * torch.sqrt((k-1)*(k-2) / k**3) * self.scale
        else:  # two layer, for circleclass models only
            self.L = torch.tensor(radius, dtype=torch.float)
            self.S = torch.tensor(2*radius**2, dtype=torch.float)


    def __call__(self, *args, **kargs):
        return (super().__call__(*args, **kargs) - self.shift) * self.scale

    def calc_max_value(self):
        k = self.k
        r = self.radius
        return -1*torch.log(np.e**(-r) / (np.e**(-r) + (k-1)))

    def calc_min_value(self):
        k = self.k
        r = self.radius
        return -1*torch.log(np.e**r / (np.e**r + (k-1)))


def pbs_maml_objective(criterion, learners, x_tr, y_tr, x_te, y_te, m_model, prior, params, epochsb, multi_out=False):
    # Getting relevant info:
    device = params['device']
    delta = params['delta']
    t = torch.tensor(params['num_tasks'], dtype=torch.float).to(device)
    m = len(y_tr[0])  # num samples trained on
    n = len(y_te[0])  # num of evaluation samples
    L = criterion.L
    S = criterion.S
    lrb = m_model.lr
    if params['sgm']:
        n_steps = epochsb * m
    else:
        n_steps = epochsb

    # print(m_model.calc_kl_div(prior))

    # Regularizer from PAC-Bayes bound
    kl_div = m_model.calc_kl_div(prior)   # kl_div_loss(m_model, prior)
    num_term = torch.log(2*torch.sqrt(t) / delta)
    reg_pac_bayes = torch.sqrt(torch.div(kl_div + num_term, 2*t))

    # calculate the average the CEL of base learners
    c = batch_criterion_loss(criterion, learners, x_tr, y_tr, x_te, y_te, device)

    # calculate alg stability constant, scaled for with-evaluation data bound
    # if it's sgm or gd, will be the same:
    if criterion.convex:
        eps = calc_eps('convex_sgm', L, S, m + n, n_steps, lrb)
    else:
        eps = calc_eps('nonconvex_sgm', L, S, m + n, n_steps, lrb, params['c'])
    if multi_out:
        return c, reg_pac_bayes + eps
    else:
        return c + reg_pac_bayes + eps


def wm_maml_objective(criterion, learners, x_tr, y_tr, x_te, y_te, m_model, prior, params, multi_out=False):
    device = params['device']
    delta = params['delta']
    t = torch.tensor(params['num_tasks'], dtype=torch.float).to(device)
    m = len(y_tr[0])  # num samples trained on
    n = len(y_te[0])  # num of evaluation samples
    k = torch.tensor(n + m, dtype=torch.float).to(device)

    # Regularizer from PAC-Bayes bound
    kl_div = m_model.calc_kl_div(prior)
    num_term = torch.sqrt(kl_div + torch.log(t * (k+1) / delta))
    rg1 = torch.sqrt(torch.tensor(1.) / (2 * (k - 1)))
    rg2 = torch.sqrt(torch.tensor(1.) / (2 * (t - 1)))
    # calculate the average the CEL of base learners
    c = batch_criterion_loss(criterion, learners, x_tr, y_tr, x_te, y_te, device)

    if multi_out:
        return c, rg1*num_term + rg2*num_term
    else:
        return c + rg1*num_term + rg2*num_term


def mlap_objective(criterion, learners, x_tr, y_tr, x_te, y_te, m_model, prior, priors, params, multi_out=False):
    device = params['device']
    delta = params['delta']
    t = torch.tensor(params['num_tasks'], dtype=torch.float).to(device)
    m = torch.tensor(len(y_tr[0]), dtype=torch.float).to(device)  # num samples trained on

    # Regularizer from PAC-Bayes bound
    kl_div_meta = m_model.calc_kl_div(prior)
    reg_1_avg = torch.tensor(0., dtype=torch.float).to(device)
    for learner, learner_prior in zip(learners, priors):
        kl_div_base = learner.calc_kl_div(learner_prior)
        reg_1_avg += torch.sqrt((kl_div_meta + kl_div_base + torch.log(2 * t * m / delta))/(2 * (m - 1)))
    reg_1_avg /= len(learners)

    reg_2 = torch.sqrt((kl_div_meta + torch.log(2 * t / delta))/(2 * (t - 1)))

    # calculate the average the CEL of base learners
    c = batch_criterion_loss(criterion, learners, x_tr, y_tr, [], [], device)

    if multi_out:
        return c, reg_1_avg + reg_2
    else:
        return c + reg_1_avg + reg_2


def batch_criterion_loss(criterion, learners, x_tr, y_tr, x_te, y_te, device):
    c = torch.tensor(0.).to(device)
    for i in range(len(y_tr)):
        x, y = x_tr[i], y_tr[i]
        c += criterion(learners[i](x[:]), y[:])

    for i in range(len(y_te)):
        x, y = x_te[i], y_te[i]
        c += criterion(learners[i](x[:]), y[:])

    c = torch.div(c, len(y_tr) + len(y_te))
    return c


def KLDiv_gaussian(mu1, var1, mu2, var2, var_is_logvar=True):
    if var_is_logvar:
        var1 = torch.exp(var1)
        var2 = torch.exp(var2)

    mu1 = torch.flatten(mu1)  # make sure we are 1xd so torch functions work as expected
    var1 = torch.flatten(var1)
    mu2 = torch.flatten(mu2)
    var2 = torch.flatten(var2)

    kl_div = 1/2 * torch.log(torch.div(var2, var1))
    kl_div += 1/2 * torch.div(var1 + torch.pow(mu2 - mu1, 2), var2)
    kl_div -= 1/2  # one for each dimension

    return torch.sum(kl_div)


def calc_eps(fn_type, L, S, m, n_steps, lr, c=None):
    # Lipschitz constant L
    # Lipschitz smoothness constant S
    # number of samples, m, used by alg A
    # n_steps: Number of gradient steps
    # lr: learning rate, assumes constant learning rate for base learner
    e_stab = 0
    if fn_type == "convex_sgm" or fn_type == "convex_gd":
        if lr > 2/S + 1e-6:
            print("WARNING: Step size too large")
        e_stab = 2*L**2/m * n_steps * lr
    elif fn_type == 'nonconvex_sgm':
        Sc = S * c
        p1 = (1 + 1/Sc)/(m - 1)
        p2 = (2*c*L**2)**(1 / (Sc + 1))
        p3 = n_steps ** (Sc / (Sc + 1))
        e_stab = p1 * p2 * p3
    elif fn_type == 'estimate':
        # large L and S
        e_stab = n_steps / m
    else:
        print("Not implemented")
        e_step = torch.tensor(np.inf)

    return e_stab


def kl_inv_l(q, c):
    import cvxpy as cvx
    solver = cvx.MOSEK
    # KL(q||p) <= c
    # try to solve: KLinv(q||c) = p

    # solve: sup  p
    #       s.t.  KL(q||p) <= c

    p_bernoulli = cvx.Variable(2)
    q_bernoulli = np.array([q, 1 - q])
    constraints = [c >= cvx.sum(cvx.kl_div(q_bernoulli, p_bernoulli)), 0 <= p_bernoulli[0], p_bernoulli[0] <= 1,
                   p_bernoulli[1] == 1.0 - p_bernoulli[0]]
    prob = cvx.Problem(cvx.Maximize(p_bernoulli[0]), constraints)
    opt = prob.solve(verbose=False, solver=solver)
    return p_bernoulli.value[0]


if __name__ == '__main__':
    pass

