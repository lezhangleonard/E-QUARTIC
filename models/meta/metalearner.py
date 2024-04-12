import torch
from torch import nn
import torch.nn.functional as F

class MetaLearner(nn.Module):
    def __init__(self, input_channels, out_classes) -> None:
        super(MetaLearner, self).__init__()
        self.input_channels = input_channels
        self.out_classes = out_classes
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels*out_classes, out_classes),
        )
    
    def forward(self, x):
        x = self.model(x)
        return x

class WeightedMetaLearner(nn.Module):
    def __init__(self, input_channels, out_classes) -> None:
        super(WeightedMetaLearner, self).__init__()
        self.input_channels = input_channels
        self.out_classes = out_classes
        self.weights = nn.Parameter(torch.ones(input_channels), requires_grad=True)
    
    def forward(self, x):
        x = F.softmax(x, dim=2)
        x = x * self.weights.unsqueeze(0).unsqueeze(-1)
        x = x.sum(dim=1)
        return x
    
class Bagging(nn.Module):
    def __init__(self, input_channels, out_classes) -> None:
        super(Bagging, self).__init__()
        self.input_channels = input_channels
        self.out_classes = out_classes
    
    def forward(self, x):
        x = F.softmax(x, dim=2)
        x = x.sum(dim=1)
        return x
