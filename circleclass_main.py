from data_generators.circleclass_data import get_circle_dataset, create_folders
from models.circleclass_models import CircleModel, SCircleModel, SSCircleModel
from learners.maml_learner import PAC_MAML_l2l
from loss import ScaledCrossEntropyLoss as SCEL
from loss import kl_inv_l
import torch
import argparse
import warnings
import learn2learn as l2l
import time
import numpy as np
warnings.filterwarnings("ignore")

params = {}
parser = argparse.ArgumentParser()
parser.add_argument('--method', help="[\'maml\', \'pac_bus\', \'mr_maml\', \'mlap\']", default=None)
parser.add_argument('--prior', help="[\'train\', \'load\', default=None]", default=None)
parser.add_argument('--size', help="[\'small\', default=\'full\']", default='full')
parser.add_argument('--verbose', help="[\'True\', default=False", default=False)
parser.add_argument('--num_val', help="Number of iterations for sample convergence bound", default=1)
parser.add_argument('--trials', help="[\'full\', default='test\']", default='test')
args = parser.parse_args()

if args.method not in ['maml', 'pac_bus', 'mr_maml', 'mlap']:
    print("invalid options")
    exit()
print(args)
########################################################################################################################
M = 10
trials = [1, 2, 3, 4, 5] if args.trials == 'full' else ['test']
num_validation = int(args.num_val)
########################################################################################################################
method = args.method
prior = args.prior
verbose = True if args.verbose == "True" else False
learner_class = PAC_MAML_l2l
epochsb = 1 if method == 'pac_bus' or method == 'mlap' else 3
epochsm = 1
lrb = 0.05
p_epochsb = 3
p_epochsm = 5

if method == 'pac_bus' or method == 'mr_maml':
    bmodel_class = SCircleModel
elif method == 'mlap':
    bmodel_class = SSCircleModel
else:  # method == 'maml':
    bmodel_class = CircleModel
########################################################################################################################
NCLS = 2
radius = 4
num_pr = 50 if args.size == 'small' else 500
num_tr = 1000 if args.size == 'small' else 10000
num_te = 100 if args.size == 'small' else 1000
num_validate = 500
########################################################################################################################
device = torch.device('cpu')
params['device'] = device
params['c'] = torch.tensor(epochsb * M * lrb, dtype=torch.float)
params['batch_size'] = 1
params['delta'] = torch.tensor(0.01)
params['method'] = method
params['logvar'] = -8
params['weights_dir'] = 'Weights/'
params['sgm'] = True
params['lrm'] = 1e-3
########################################################################################################################
criterion = SCEL(NCLS, radius, convex=False)
create_folders((params['weights_dir'],))
########################################################################################################################
score = []
testloss = []
bound = []
if verbose: print("         M =", M)
for t in trials:
    if verbose: print("     trial =", t)
    model_name = method + '_m' + str(M)
    params['version'] = model_name
    params['prior_version'] = 'prior_' + model_name
    _, X_te = get_circle_dataset(num_tasks=num_te, num_learn=M, num_validate=num_validate, prior=False)

    bmodel = bmodel_class(2, NCLS, radius)
    bmodel.init_logvar(logvar=params['logvar']*2, b_logvar=params['logvar']*2)
    model = l2l.algorithms.MAML(bmodel, lr=lrb, first_order=False, allow_nograd=True)
    learner = learner_class(model, criterion, params)

    if prior == 'train':
        if verbose: print("Training prior")
        _, X_tr_p = get_circle_dataset(num_tasks=num_pr, num_learn=M, num_validate=num_validate, prior=True)
        _, X_tr = get_circle_dataset(num_tasks=num_tr, num_learn=M, num_validate=num_validate, prior=False)
        learner.meta_fit(*X_tr_p, verbose=verbose, epochsm=p_epochsm, epochsb=p_epochsb, batch_size=1)
        params['num_tasks'] = num_tr
    elif prior == 'load':
        if verbose: print("Loading prior")
        _, X_tr = get_circle_dataset(num_tasks=num_tr, num_learn=M, num_validate=num_validate, prior=False)
        learner.load_weights(version=params['prior_version'])
        params['num_tasks'] = num_tr
    else:
        if verbose: print("Will not use prior")
        _, X_tr = get_circle_dataset(num_tasks=num_tr+num_pr, num_learn=M, num_validate=num_validate, prior=False)
        params['num_tasks'] = num_tr+num_pr

    learner.model.init_logvar(logvar=params['logvar'])
    learner.save_weights(version=params['prior_version'])

    learner = learner_class(model, criterion, params)
    if verbose: print("Training model")

    learner.meta_fit(*X_tr, verbose=verbose, epochsm=epochsm, epochsb=epochsb, batch_size=None)
    if verbose: print("Scoring model")

    if method == 'maml':
        s, tl, _ = learner.meta_score(*X_te, epochsb=epochsb)
        b = 1
    elif method == 'mr_maml' or method == 'pac_bus':
        s, tl, _ = learner.meta_score(*X_te, epochsb=epochsb)
        params['delta'] = torch.tensor(0.009)
        deltap = 0.001
        _, loss, regularizer = learner.meta_score(*X_tr, epochsb=epochsb, num_iters=num_validation)
        rg = np.log(2 / deltap) / num_validation
        lossbound = kl_inv_l(loss, rg)
        b = lossbound + regularizer
    elif method == 'mlap':
        s, tl, _ = learner.meta_score(*X_te, epochsb=epochsb)
        _, loss, regularizer = learner.meta_score(*X_tr, epochsb=epochsb, num_iters=1)
        # don't bother validating this because it takes too many iterations to compute the bound
        b = regularizer
    else:
        s, tl, b = 0, 1, 1

    score.append(s)
    testloss.append(tl)
    bound.append(b)

print("score =", score)
print("testloss =", testloss)
print("bound =", bound)
