from tasks import RBFClassification
import torch

task = RBFClassification(10, 5)
print(task.evaluate(torch.normal(torch.zeros((5, 10)))))