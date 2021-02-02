from loss import *
from threading import Thread as thread


class L2Ler:
    def __init__(self, model, criterion, params):
        self.params = params
        self.model = model
        self.criterion = criterion

    def meta_fit(self, x_tr, y_tr, x_te, y_te, verbose=False):
        raise NotImplementedError()

    def save_weights(self, version):
        v = str(version)
        loc = self.params['weights_dir'] + 'w_' + v + '.pt'
        torch.save(self.model.state_dict(),  loc)

    def load_weights(self, version, model=None):
        v = str(version)
        loc = self.params['weights_dir'] + 'w_' + v + '.pt'
        if model is not None:
            model.load_state_dict(torch.load(loc))
            return model
        else:
            self.model.load_state_dict(torch.load(loc))


class PAC_MAML_l2l(L2Ler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lrm'])

    def meta_fit(self, x_tr, y_tr, x_te, y_te, verbose=False, epochsm=1, epochsb=1, batch_size=None):
        optimizer = self.optimizer
        criterion = self.criterion
        m_loss = []
        method = self.params['method'] if batch_size is None else 'maml'
        batch_size = batch_size if batch_size is not None else len(y_tr)

        for epoch in range(epochsm):
            inds = [i for i in range(len(y_tr))]
            if batch_size < len(y_tr):
                np.random.shuffle(inds)
            for k in range(int(len(y_tr)/batch_size) + 1):
                batch = inds[k*batch_size:(k+1)*batch_size]
                if batch:  # just make sure it's not empty
                    learners = []
                    priors = []
                    x_train = []
                    y_train = []
                    x_test = []
                    y_test = []
                    processes = []
                    for ct, i in enumerate(batch):
                        learners.append(self.model.clone())
                        priors.append(self.model.clone())
                        learners[ct].init_as_base()
                        x_train.append(x_tr[i])
                        y_train.append(y_tr[i])
                        x_test.append(x_te[i])
                        y_test.append(y_te[i])
                        p = thread(target=self.btrain, args=(learners[ct], criterion, x_train[ct], y_train[ct], epochsb))
                        p.start()
                        processes.append(p)
                    for p in processes:
                        p.join()
                    X = (x_train, y_train, x_test, y_test)
                    if method == 'pac_bus':
                        prior = self.get_prior()
                        meta_error = pbs_maml_objective(criterion, learners, *X, self.model, prior, self.params, epochsb)
                    elif method == 'mr_maml':
                        prior = self.get_prior()
                        meta_error = wm_maml_objective(criterion, learners, *X, self.model, prior, self.params)
                    elif method == 'mlap':
                        prior = self.get_prior()
                        meta_error = mlap_objective(criterion, learners, *X, self.model, prior, priors, self.params)
                    else:  # MAML or FMRL
                        meta_error = batch_criterion_loss(criterion, learners, *X, self.params['device'])

                    optimizer.zero_grad()
                    meta_error.backward(retain_graph=True)
                    optimizer.step()
                    m_loss.append(meta_error.detach().numpy())

            if verbose: print("Iteration", epoch + 1, ": Loss =", m_loss[-1], end='\r')
        if verbose: print()

    def meta_score(self, x_tr, y_tr, x_te, y_te, epochsb=1, num_iters=1):
        criterion = self.criterion
        method = self.params['method']
        X = (x_tr, y_tr, x_te, y_te)
        m_loss = []
        regs = []

        for _ in range(num_iters):
            priors = []
            learners = []
            processes = []
            for i in range(len(y_tr)):
                learners.append(self.model.clone())
                priors.append(self.model.clone())
                learners[i].init_as_base()
                p = thread(target=self.btrain, args=(learners[i], criterion, x_tr[i], y_tr[i], epochsb))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            if method == 'pac_bus':
                prior = self.get_prior()
                meta_error, regularizer = pbs_maml_objective(criterion, learners, *X, self.model, prior, self.params, epochsb, multi_out=True)
            elif method == 'mr_maml':
                prior = self.get_prior()
                meta_error, regularizer = wm_maml_objective(criterion, learners, *X, self.model, prior, self.params, multi_out=True)
            elif method == 'mlap':
                prior = self.get_prior()
                meta_error, regularizer = mlap_objective(criterion, learners, *X, self.model, prior, priors, self.params, multi_out=True)
            else:
                meta_error = batch_criterion_loss(criterion, learners, *X, self.params['device'])
                regularizer = torch.tensor(1.)
            m_loss.append(meta_error.detach().numpy())
            regs.append(regularizer.detach().numpy())

        scores = np.empty(len(y_tr))
        for i in range(len(y_tr)):
            scores[i] = sum((torch.argmax(learners[i](x_te[i]), axis=1) == y_te[i]).cpu().numpy())/len(y_te[i])
        score = np.mean(scores)

        return score, np.mean(m_loss), np.mean(regs)

    @staticmethod
    def btrain(learner, bcriterion, x, y, epochsb, sgm=False):
        for step in range(epochsb):
            if sgm:
                for j in range(len(x)):
                    error = bcriterion(learner(x[j:j + 1]), y[j:j + 1])
                    learner.adapt(error)
            else:
                error = bcriterion(learner(x[:]), y[:])
                learner.adapt(error)
        return

    def get_prior(self):
        prior = self.model.clone()
        prior = self.load_weights(self.params['prior_version'], model=prior)
        return prior
