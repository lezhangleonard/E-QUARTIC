import torch
from torch import nn
import torch.nn.functional as F
from utils.utils import *
from torch.autograd import Function

class MetaSelector(nn.Module):
    def __init__(self, ensemble, input_channels, out_dim, k) -> None:
        super(MetaSelector, self).__init__()
        self.selector = nn.Sequential(
           nn.Conv2d(input_channels, 1, kernel_size=5, stride=1, padding=0),
           nn.ReLU(),
           nn.MaxPool2d(2),
           nn.Conv2d(1, 2, kernel_size=5, stride=1, padding=0),
           nn.ReLU(),
           nn.MaxPool2d(2),
           nn.Conv2d(2, 4, kernel_size=5, stride=1, padding=0),
           nn.ReLU(),
           nn.Flatten(),
           nn.Linear(4, out_dim),
           L2NormalizationLayer(),
           TopK(k, out_dim)
       )
        initialize(self.selector)
        self.ensemble = ensemble.requires_grad_(False)

    def forward(self, x):
        mask = self.selector(x)
        outputs = []
        for i, weak_learner in enumerate(self.ensemble.learners):
            learner_output = weak_learner(x)
            masked_output = learner_output * mask.unsqueeze(1)[:,:,i]
            outputs.append(masked_output)
        outputs = torch.stack(outputs, dim=1)
        combined_output = outputs.sum(dim=1)
        return combined_output


class TopK(nn.Module):
    def __init__(self, k, dim):
        super().__init__()
        assert k <= dim
        self.k = k
        self.dim = dim

    def forward(self, x):
        return TopKPerturbed.apply(x, self.k)


class L2NormalizationLayer(nn.Module):
    def forward(self, x):
        l2_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        l2_norm = torch.clamp(l2_norm, min=1e-9)
        normalized_x = x / l2_norm
        return normalized_x


class TopKPerturbed(Function):
    @staticmethod
    def forward(ctx, input, k, temp=1):
        values, indices = torch.topk(input, k)
        output = torch.zeros_like(input).scatter_(1, indices, 1)
        ctx.save_for_backward(input)
        ctx.k = k
        ctx.temp = temp
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        k = ctx.k
        temp = ctx.temp
        m = 1000
        noise = torch.randn_like(input.unsqueeze(0).expand(m, *input.shape)) * temp
        perturbed_input = input.unsqueeze(0) + noise
        values, indices = torch.topk(perturbed_input, k, dim=2)
        K_perturbed = torch.zeros_like(perturbed_input).scatter_(2, indices, 1)
        v_prime = 2 * noise
        grad_input = (K_perturbed * v_prime).mean(0)
        return grad_input, None, None

