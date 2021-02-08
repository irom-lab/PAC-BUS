from learners.maml_learner import PAC_MAML_l2l
from learners.fli_learner import FMRL_l2l
from loss import ScaledCrossEntropyLoss as SCEL
from loss import kl_inv_l
import learn2learn as l2l
from models.miniwiki_models import MiniWikiModel, SMiniWikiModel
from data_generators.miniwiki_data import load_to_device, load_pb_to_device, create_folders
import warnings
import torch
import argparse
import numpy as np
import random
warnings.filterwarnings("ignore")

params = {}
parser = argparse.ArgumentParser()
parser.add_argument('--method', help="[\'maml\', \'pac_bus\', \'mr_maml\', \'fli_batch\']", default=None)
parser.add_argument('--prior', help="[\'train\', \'load\', default=None]", default=None)
parser.add_argument('--size', help="[\'small\', default=\'full\']", default='full')
parser.add_argument('--verbose', help="[\'True\', default=False]", default='False')
parser.add_argument('--num_val', help="Number of iterations for sample convergence bound", default=1)
parser.add_argument('--trials', help="[\'full\', default='test\']", default='test')
args = parser.parse_args()

if args.method not in ['maml', 'pac_bus', 'mr_maml', 'fli_batch']:
    print("invalid options")
    exit()
########################################################################################################################
MS = [1, 3, 5]
trials = [1, 2, 3, 4, 5] if args.trials == 'full' else ['test']
num_validation = int(args.num_val)
########################################################################################################################
method = args.method
prior = args.prior
verbose = True if args.verbose == "True" else False
learner_class = PAC_MAML_l2l if method != 'fli_batch' else FMRL_l2l
train_prior = prior is not None
if method == 'maml':
    epochsbs = [2, 5, 8]
    lrbs = [2.5, 5, 5]
elif method == 'pac_bus':
    epochsbs = [2, 4, 5] if train_prior else [5, 5, 5]
    lrbs = [2.5, 5, 5]
elif method == 'mr_maml':
    epochsbs = [2, 5, 8] if train_prior else [15, 10, 8]
    lrbs = [2.5, 5, 5]
else:  # method == 'fli_batch
    epochsbs = [1, 1, 1]
    epsilons = [0.2, 0.02, 0.002]   # FLI-Batch
    lrbs = [7, 7, 7]               # FLI-Batch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

epochsms = [1, 1, 1]

if method == 'maml' or method == 'fli_batch':
    bmodel_class = MiniWikiModel
    p_epochsbs = [2, 5, 8]
    p_epochsms = [5, 4, 3] if method != 'fli_batch' else [5, 20, 50]
else:
    bmodel_class = SMiniWikiModel
    p_epochsbs = [2, 5, 8]
    p_epochsms = [10, 8, 5]
########################################################################################################################
DIM = 50
NCLS = 4
radius = 1
num_pr = 30 if args.size == 'small' else 100
num_tr = 50 if args.size == 'small' else 1000
num_te = 20 if args.size == 'small' else 200
########################################################################################################################
device = torch.device('cpu')
params['device'] = device
params['batch_size'] = 10
params['delta'] = torch.tensor(0.01)
params['method'] = method
params['logvar'] = -2
params['gamma'] = 1.1
params['weights_dir'] = 'Weights/'
params['sgm'] = False  # false implies we will use GD instead of SGM
params['case'] = 'batch'  # 'batch' 'FAL' 'online'
params['lrm'] = 0.1
########################################################################################################################
criterion = SCEL(NCLS, radius)
create_folders((params['weights_dir'],))
########################################################################################################################
for i, M in enumerate(MS):
    if M == 0:
        continue
    if method == 'fli_batch':
        params['epsilon'] = epsilons[i]
    score = []
    testloss = []
    bound = []
    if verbose: print("         M =", M)
    for t in trials:
        if verbose: print("     trial =", t)
        corpus = str(t) + '_ncls' + str(NCLS) + 'm' + str(M)
        model_name = method + '_m' + str(M)
        params['version'] = model_name
        params['prior_version'] = 'prior_' + model_name
        w2v, X_te = load_to_device(device, corpus, 'test', DIM, NCLS, M, num_tasks=num_te, w2v=None)

        bmodel = bmodel_class(DIM, NCLS, radius)
        bmodel.init_logvar(logvar=params['logvar'])
        model = l2l.algorithms.MAML(bmodel, lr=lrbs[i], first_order=False, allow_nograd=True)
        plearner = PAC_MAML_l2l(model, criterion, params)
        learner = learner_class(model, criterion, params)

        if prior == 'train':
            if verbose: print("Training prior")
            _, X_tr_p = load_to_device(device, corpus, 'prior', DIM, NCLS, M, num_tasks=num_pr, w2v=w2v)
            _, X_tr = load_to_device(device, corpus, 'train', DIM, NCLS, M, num_tasks=num_tr, w2v=w2v)
            if method == 'fli_batch':
                learner.meta_fit(*X_tr_p, verbose=verbose, epochsm=p_epochsms[i], lrm=epsilons[i])
            else:
                plearner.meta_fit(*X_tr_p, verbose=verbose, epochsm=p_epochsms[i], epochsb=p_epochsbs[i], batch_size=10)
            params['num_tasks'] = num_tr
        elif prior == 'load':
            if verbose: print("Loading prior")
            _, X_tr = load_to_device(device, corpus, 'train', DIM, NCLS, M, num_tasks=num_tr, w2v=w2v)
            plearner.load_weights(version=params['prior_version'])
            params['num_tasks'] = num_tr
        else:
            if verbose: print("Will not use prior")
            _, X_tr = load_pb_to_device(device, corpus, 'prior', 'train', DIM, NCLS, M, num_tasks1=num_pr, num_tasks2=num_tr, w2v=w2v)
            params['num_tasks'] = num_tr + num_pr

        plearner.model.init_logvar(logvar=params['logvar'])
        plearner.save_weights(version=params['prior_version'])

        if verbose: print("Training model")
        if method == 'fli_batch':
            learner.meta_fit(*X_tr, verbose=verbose, epochsm=epochsms[i], lrm=epsilons[i])
        else:
            learner.meta_fit(*X_tr, verbose=verbose, epochsm=epochsms[i], epochsb=epochsbs[i], batch_size=None)
        if verbose: print("Scoring model")

        if method == 'fli_batch':
            s, tl, b = learner.meta_score(*X_te)
        elif method == 'maml':
            s, tl, _ = learner.meta_score(*X_te, epochsb=epochsbs[i])
            b = 1
        elif method == 'mr_maml' or method == 'pac_bus':
            s, tl, _ = learner.meta_score(*X_te, epochsb=epochsbs[i])
            params['delta'] = torch.tensor(0.009).to(device)
            deltap = 0.001
            _, loss, regularizer = learner.meta_score(*X_tr, epochsb=epochsbs[i], num_iters=num_validation)
            rg = np.log(2/deltap)/num_validation
            lossbound = kl_inv_l(loss, rg)
            b = lossbound + regularizer
        else:
            s, tl, b = 0, 1, 1

        score.append(s)
        testloss.append(tl)
        bound.append(b)

    print("score =", score)
    print("testloss =", testloss)
    print("bound =", bound)
