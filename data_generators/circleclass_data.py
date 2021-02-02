import numpy as np
import torch
import os

def create_folders(names):
    for name in names:
        if not os.path.exists(name):
            os.system("mkdir " + name)


class CircleDataGenerator():
    def __init__(self, prior):
        self.dimensions = 2
        self.theta = np.random.uniform(0, np.pi)

        if prior:
            self.dist_from_center = np.random.uniform(0.1, 0.5)
            self.radius = np.random.uniform(0.1, 0.8 - self.dist_from_center)
            self.theta = np.random.uniform(0, np.pi)
        else:
            self.dist_from_center = np.random.uniform(0, 0.4)
            self.radius = np.random.uniform(0.1, 1 - self.dist_from_center)
            self.theta = np.random.uniform(0, np.pi)

    def _sample(self, num_samples):
        r = np.random.uniform(0, 1, num_samples)

        theta = np.random.uniform(0, 2*np.pi, num_samples)
        x = np.zeros((num_samples, 2))
        x[:, 0] += r * np.cos(theta)
        x[:, 1] += r * np.sin(theta)

        # y = r >= self.radius
        dfc = self.dist_from_center
        shifted = np.sqrt((x[:, 0] - dfc * np.cos(self.theta))**2 + (x[:, 1] - dfc * np.sin(self.theta))**2)

        y = shifted >= self.radius

        return x, y

    def batch(self, num_samples=10):
        x, y = self._sample(num_samples)
        return self.np_to_tensor(x, y)

    @staticmethod
    def np_to_tensor(x, y):
        return (torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long))


def get_circle_dataset(num_tasks=1000, num_learn=10, num_validate=250, prior=False):
    cdg = CircleDataGenerator
    train_ds = []
    num_train = num_learn
    num_validation = num_validate
    for i in range(num_tasks):
        train_ds.append(cdg(prior))

    x_tr = []
    y_tr = []
    x_te = []
    y_te = []
    for i, t in enumerate(train_ds):
        x_train, y_train = t.batch(num_samples=num_train)
        x_tr.append(x_train)
        y_tr.append(y_train)

        x_validation, y_validation = t.batch(num_samples=num_validation)
        x_te.append(x_validation)
        y_te.append(y_validation)

    return train_ds, (x_tr, y_tr, x_te, y_te)
