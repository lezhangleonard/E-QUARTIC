import torch
from torch import nn
import torch.nn.functional as F

class UnweightedVoting(nn.Module):
    def __init__(self, num_learners):
        super().__init__()
        self.num_learners = num_learners
    
    def forward(self, x):
        max_indices = torch.argmax(x, dim=2)
        votes = F.one_hot(max_indices, num_classes=x.shape[2])
        vote_counts = torch.sum(votes, dim=1)
        result_indices = torch.argmax(vote_counts, dim=1)
        output = F.one_hot(result_indices, num_classes=x.shape[2])
        return output.float()

class WeightedVoting(nn.Module):
    def __init__(self, num_learners, weights):
        super().__init__()
        self.num_learners = num_learners
        self.weights = weights
    
    def forward(self, x):
        max_indices = torch.argmax(x, dim=2)
        votes = F.one_hot(max_indices, num_classes=x.shape[2])
        weights = self.weights.unsqueeze(-1).to(votes.device)
        votes = votes * weights
        vote_counts = torch.sum(votes, dim=1)
        result_indices = torch.argmax(vote_counts, dim=1)
        output = F.one_hot(result_indices, num_classes=x.shape[2])
        return output.float()
        



        
