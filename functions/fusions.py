import torch.nn.functional as F
import torch

def unweighted_fusion(outputs, weights):
    return torch.mean(torch.stack(outputs), dim=0)

def weighted_fusion(outputs, weights):
    weights = [F.softmax(weight, dim=0) for weight in weights]
    return torch.sum(torch.stack([output * weight for output, weight in zip(outputs, weights)]), dim=0)