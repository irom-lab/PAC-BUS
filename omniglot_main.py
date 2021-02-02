import torch
import numpy as np
from data_generators.omniglot_data import OmniglotNShot
import argparse
import learn2learn as l2l
from models.omniglot_models import SOmniglotModel, OmniglotModel, OmniglotModel1
from learners.reptile_learner import ReptileLearner


argparser = argparse.ArgumentParser()
argparser.add_argument('--method', type=str, help="[\'maml\', \'pac_bus_h\', \'mr_maml_w\', \'fli_online\']")
argparser.add_argument('--nme', help="Whether samples will be not-mutually-exclusive", default='False')
argparser.add_argument('--n_way', type=int, default=20)
argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
argparser.add_argument('--batch', type=int, help='batch size', default=16)
argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
argparser.add_argument('--gpu', type=int, help='which GPU', default=0)
argparser.add_argument('--lrm', type=float, help='meta learning rate', default=0.001)
argparser.add_argument('--lrb', type=float, help='base learners learning rate', default=0.1)
argparser.add_argument('--n_filt', type=int, default=64)
argparser.add_argument('--seed', type=int, default=-1)
argparser.add_argument('--regscale', type=float, default=1e-6)
argparser.add_argument('--regscale2', type=float, default=1)
argparser.add_argument('--epochsb', type=int, default=5)
argparser.add_argument('--epochsm', type=int, default=100000)
args = argparser.parse_args()

if args.gpu == -1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:'+str(args.gpu))


nme = args.nme != 'False'
method = args.method
if method not in ['maml', 'pac_bus_h', 'mr_maml_w', 'fli_online']:
    print("invalid options, select one of: [\'maml\', \'pac_bus_h\', \'mr_maml_w\', \'fli_online\']")
    exit()

if args.seed >= 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

init_as_base = True
partition = 500
score_part = 2000
results_part = 10000

print(args)

if method == 'maml':
    bmodel = OmniglotModel(n_way=args.n_way).to(device)
elif method == 'fli_online':
    bmodel = ReptileLearner(OmniglotModel1, (args.n_way, args.n_filt), n_way=args.n_way, k_shot=args.k_spt, meta_batchsz=args.batch, beta=args.lrm, num_updates=args.epochsb).to(device)
elif method == 'mr_maml_w':
    bmodel = SOmniglotModel(n_way=args.n_way, n_filt=args.n_filt).to(device)
    bmodel.init_logvar(-6, -6)
else:  # method == 'pac_bus_h':
    bmodel = SOmniglotModel(n_way=args.n_way, n_filt=args.n_filt, ELU=True).to(device)
    bmodel.init_logvar(-6, -6)

reg1_scale = torch.tensor(args.regscale, dtype=torch.float).to(device)
reg2_scale = torch.tensor(args.regscale2, dtype=torch.float).to(device)  # only for pac_bus_h
delta = torch.tensor(0.01, dtype=torch.float).to(device)
T = torch.tensor(args.epochsm, dtype=torch.float).to(device)
m = torch.tensor(args.epochsb, dtype=torch.float).to(device)
c = T * args.lrb
if method != 'fli_online':
    model = l2l.algorithms.MAML(bmodel, lr=args.lrb, first_order=False, allow_nograd=True).to(device)
    prior = model.clone().to(device)
else:
    model = bmodel
    prior = None

optimizer = torch.optim.Adam(model.parameters(), lr=args.lrm)
criterion = torch.nn.CrossEntropyLoss().to(device)
db_train = OmniglotNShot('./data/omniglot', batchsz=args.batch, n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, nme=nme, device=None)

for step in range(1, args.epochsm+1):
    x_spt, y_spt, x_qry, y_qry = db_train.next('train')
    x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
    num_tasks = len(x_spt)

    score = 0
    if method != 'fli_online':
        meta_error = torch.tensor(0.).to(device)
        optimizer.zero_grad()
        learners = []
        for i in range(num_tasks):
            learners.append(model.clone())
            if init_as_base:
                learners[i].init_as_base()
            for be in range(args.epochsb):
                loss = criterion(learners[i](x_spt[i]), y_spt[i])
                learners[i].adapt(loss)

            pred_q = torch.nn.functional.softmax(learners[i](x_qry[i]), dim=1).argmax(dim=1)
            score += torch.eq(pred_q, y_qry[i]).sum().item() / len(y_qry[0, :])
            # loss = criterion(learners[i](x_qry[i]), y_qry[i])
            if method == 'pac_bus_h':
                kl_div = learners[i].calc_kl_div(prior, device)
                reg1 = torch.sqrt(kl_div + (torch.log(2*torch.sqrt(T) / delta)) / (2 * T))
                L, S = learners[i].calc_ls_constants(device)
                p1 = (1 + 1 / S*c) / (m - 1)
                p2 = (2 * c * L**2) ** (1 / (S*c + 1))
                p3 = T**(S*c / (S*c + 1))
                reg2 = p1 * p2 * p3
                meta_error += reg1_scale*reg1 + reg2_scale*reg2
            if method == 'mr_maml_w':
                kl_div = learners[i].calc_kl_div(prior, device)
                meta_error += reg1_scale*kl_div  # equation 5 from MLWM paper

            meta_error += criterion(learners[i](x_qry[i]), y_qry[i])

        meta_error /= num_tasks
        meta_error.backward(retain_graph=True)
        optimizer.step()
    else:
        accs = model(x_spt, y_spt, x_qry, y_qry)
        score += np.array(accs).sum()

    if step % partition == 0:
        print('step:', step, '\t score:', score / num_tasks)
        scores = []
        num_test_trials = 10
        if step % results_part == 0:
            print("Results:")
            num_test_trials = 500
        elif step % score_part == 0:
            print("Score many more test trials:")
            num_test_trials = 100
        for _ in range(num_test_trials):
            x_spt, y_spt, x_qry, y_qry = db_train.next('test')
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
            num_tasks = len(x_spt)
            if method != 'fli_online':
                accs = []
                for i in range(num_tasks):
                    learner = model.clone()
                    if init_as_base:
                        learner.init_as_base()
                    for be in range(args.epochsb):
                        loss = criterion(learner(x_spt[i]), y_spt[i])
                        learner.adapt(loss)
                    pred_q = torch.nn.functional.softmax(learner(x_qry[i]), dim=1).argmax(dim=1)
                    accs.append(torch.eq(pred_q, y_qry[i]).sum().item()/len(y_qry[0, :]))
                scores.append(np.mean(accs))
            else:
                acc = model.pred(x_spt, y_spt, x_qry, y_qry)
                scores.append(acc)

        print("test accs", np.round(np.mean(scores),5))
    else:
        print('step:', step, '\t score:', score / num_tasks, end="\r")
print()

