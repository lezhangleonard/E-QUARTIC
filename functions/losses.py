import torch
import torch.nn.functional as F

class Cosine_Similarity(torch.nn.Module):
    def __init__(self):
        super(Cosine_Similarity, self).__init__()
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x, y):
        return self.cos(x, y)

class Symmetrized_KL_Divergence(torch.nn.Module):
    def __init__(self):
        super(Symmetrized_KL_Divergence, self).__init__()

    def forward(self, x, y):
        x = F.softmax(x, dim=1)
        y = F.softmax(y, dim=1)
        return (F.kl_div(x, y) + F.kl_div(y, x)) / 2
    
class PairwiseLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(PairwiseLoss, self).__init__()
        self.alpha = alpha
        self.cos = Cosine_Similarity()
        self.kl = Symmetrized_KL_Divergence()
        
    def forward(self, y, y_i, y_j):
        loss = self.alpha * self.cos(y_i, y_j) + (1-self.alpha) * self.kl(y_i, y_j)
        return torch.mean(loss)
    
class JointLoss(torch.nn.Module):
    def __init__(self, ensemble: torch.nn.Module, base: torch.nn.Module, alpha=0.5, gamma=0.5):
        super(JointLoss, self).__init__()
        self.gamma = gamma
        self.pairwise = PairwiseLoss(alpha)
        self.ensemble = ensemble
        self.base = base
    
    def forward(self, x, y_hat, y_target):
        loss = self.base(y_hat, y_target)
        if self.ensemble is None:
            return loss
        for i in torch.arange(len(self.ensemble.learners)):
            self.ensemble.learners[i].eval()
            loss += self.gamma * self.pairwise(y_target, y_hat, self.ensemble.learners[i](x))
        return loss
        
