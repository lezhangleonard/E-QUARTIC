import torch
from torch import nn
from utils.utils import *

class LocalSelector(nn.Module):
    def __init__(self, ensemble, learner, in_dim, out_dim, k) -> None:
        super(LocalSelector, self).__init__()
        self.selector = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.75),
            nn.Linear(16, out_dim),
            TopK(1, out_dim),
        )
        initialize(self.selector)
        self.learner = learner.requires_grad_(False)
        self.ensemble = ensemble.requires_grad_(False)

    def forward(self, x):
        learner_output = self.learner(x)
        mask = self.selector(learner_output).transpose(0,1)
        learner_outputs = []
        for i, learner in enumerate(self.ensemble.learners):
            learner_outputs.append(learner(x) * mask[i].unsqueeze(-1))
        learner_outputs = torch.stack(learner_outputs, dim=1)
        output = learner_outputs.sum(dim=1)
        return learner_output + output

class TopK(nn.Module):
    def __init__(self, k, dim):
        super().__init__()
        assert k <= dim
        self.k = k
        self.dim = dim

    def forward(self, x):
        val, ind = torch.topk(-x, k=self.dim-self.k, dim=-1)
        x.scatter_(index=ind, dim=-1, value=0)
        return x