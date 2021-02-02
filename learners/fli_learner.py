from loss import *
from models.miniwiki_models import ConstrainedLogit

class FMRL_l2l():
    def __init__(self, model, criterion, params, method='batch'):
        self.params = params
        self.model = model
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lrm'])
        self.method = method  # ['online', 'batch']
        self.radius = 1
        self.tar = 1
        self.m_loss = []
        self.T = 0

    def meta_fit(self, x_tr, y_tr, x_te, y_te, epochsm, lrm, verbose=False):
        criterion = self.criterion
        self.m_loss = []
        self.T = len(x_tr)
        lossess = []
        bih_losses = []
        X = (x_tr, y_tr, x_te, y_te)
        for epoch in range(epochsm):
            learners = []
            for i in range(len(y_tr)):
                x = x_tr[i]
                y = y_tr[i]
                learners.append(self.model.clone())
                learners[i].init_as_base()

                if self.method == 'batch':  # Batch
                    learnerss = []
                    lossess.append([])
                    for j in range(len(x)):
                        learnerss.append(self.model.clone())
                        learnerss[j].init_as_base()
                        error = criterion(learnerss[j](x[:j+1]), y[:j+1])
                        lossess[i].append(error.detach().numpy())
                        learnerss[j].adapt(error)
                    if epochsm == 1:
                        bih_theta = self.bih_clogit(x, y)
                        bih_learner = self.model.clone()
                        bih_learner.init_as_base()
                        bih_learner.layer1.layer = torch.nn.Parameter(bih_theta.clone())
                        bih_losses.append(len(x) * float(self.criterion(bih_learner(x[:]), y[:])))
                        self.tar = self.TAR(lossess, bih_losses)

                    learners[i].calc_addorsub(learners[i], add=False, lr=1)
                    for j in range(len(x)):
                        learners[i].calc_addorsub(learnerss[j], add=True, lr=1)
                    learners[i].calc_addorsub(learners[i], add=None, lr=1/len(x))
                else:  # Online
                    error = criterion(learners[i](x[:]), y[:])
                    learners[i].adapt(error)

                learners[i].calc_addorsub(self.model, add=False, lr=1)
                self.model.calc_addorsub(learners[i], add=True, lr=lrm)

            meta_error = batch_criterion_loss(criterion, learners, *X, self.params['device'])
            self.m_loss.append(meta_error.detach().numpy())

            if verbose: print("Iteration", epoch + 1, ": Loss =", self.m_loss[-1], end='\r')
        if verbose: print()

    def meta_score(self, x_tr, y_tr, x_te, y_te):
        criterion = self.criterion
        X = (x_tr, y_tr, x_te, y_te)

        learners = []
        for i in range(len(y_tr)):
            x = x_tr[i]
            y = y_tr[i]
            learners.append(self.model.clone())
            learners[i].init_as_base()

            if self.method == 'batch':
                learnerss = []
                for j in range(len(x)):
                    learnerss.append(self.model.clone())
                    learnerss[j].init_as_base()
                    error = criterion(learnerss[j](x[:j+1]), y[:j+1])
                    learnerss[j].adapt(error)

                learners[i].calc_addorsub(learners[i], add=False, lr=1)
                for j in range(len(x)):
                    learners[i].calc_addorsub(learnerss[j], add=True, lr=1)
                learners[i].calc_addorsub(learners[-1], add=None, lr=1 / len(x))

            else:  # Online
                error = criterion(learners[i](x[:]), y[:])
                learners[i].adapt(error)

        meta_error = batch_criterion_loss(criterion, learners, *X, self.params['device'])

        r = 1
        if self.method == 'batch':
            p1 = np.mean(self.m_loss[-1])
            p2 = float(self.tar / len(x_tr[0]))
            p3 = float(np.sqrt(8 / self.T * np.log(1 / self.params['delta'])))
            r = p1 + p2 + p3

        scores = np.empty(len(y_tr))
        for i in range(len(y_tr)):
            scores[i] = sum((torch.argmax(learners[i](x_te[i]), axis=1) == y_te[i]).cpu().numpy())/len(y_te[i])
        score = np.mean(scores)

        return score, np.mean(meta_error.detach().numpy()), r

    def bih_clogit(self, x, y):
        clogit = ConstrainedLogit(radius=self.radius)
        clogit.fit(x, y)
        bih_theta = torch.tensor(clogit.coef_, dtype=torch.float)
        return bih_theta

    def TAR(self, lossess, bih_losses):
        T = len(lossess)
        tar = 0.
        for t in range(T):
            losses = lossess[t]
            tar += sum(losses)
            tar -= bih_losses[t]
        tar /= T
        return tar



