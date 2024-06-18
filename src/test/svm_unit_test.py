from tasks import RBFLogisticRegression
from samplers import GaussianSampler
from models import SVMModel
import numpy as np
import random
import torch

def fix_seed():
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    torch.cuda.manual_seed(42)

fix_seed()
n_dim = 10
b_size = 128
n_points = 60

task = RBFLogisticRegression(n_dim, b_size)
sampler = GaussianSampler(n_dim)

# Randomly sample for 100 times
for _ in range(5):
    X = sampler.sample_xs(n_points, b_size)
    y = task.evaluate(X)

    model = SVMModel()
    res = model(X, y)
    acc = (res[:, -1] == y[:, -1]).sum() / b_size
    print(acc)


