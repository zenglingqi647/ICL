from tasks import RBFLogisticRegression
from samplers import GaussianSampler
import torch

task = RBFLogisticRegression(10, 5)
sampler = GaussianSampler(10)

# Randomly sample for 100 times
for i in range(10):
    data_sample = sampler.sample_xs(7, 5)
    result = task.evaluate(data_sample)
    # Check frequency of 1s and -1s
    print(result)
    print(result.mean())