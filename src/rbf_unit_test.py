from tasks import RBFLogisticRegression
import torch

task = RBFLogisticRegression(10, 5)
print(task.evaluate(torch.normal(torch.zeros((5, 10)))))