import torch
from torch.utils.data import Dataset, TensorDataset
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from PIL import Image


class WeightedDataset(Dataset):
    def __init__(self, data, targets, alpha, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.weights = torch.ones(len(data))
        self.alpha = 5e-3
        self.k = 10

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        w = self.weights[index]
        if self.transform:
            # if x is not Image:
            #     x = ToPILImage()(x) if x.ndim == 3 else ToPILImage()(x.unsqueeze(0))
            # # x = Image.open(x)
            x = self.transform(x)
        return x, y, w

    def __len__(self):
        return len(self.data)

    def update_weights(self, indices, y_hat, y, eps=1e-12):
        y = y.to(self.weights.device)
        y_hat = y_hat.to(self.weights.device)
        y_onehot = torch.zeros_like(y_hat)
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        y_onehot = y_onehot.to(self.weights.device)
        y_hat = y_hat.unsqueeze(-1)
        y_onehot = y_onehot.unsqueeze(1)
        with torch.no_grad():
            x = torch.log(torch.clamp(y_hat, min=eps))
            x = torch.bmm(y_onehot, x)
            x = -self.alpha * (self.k - 1) / self.k * x
            x = torch.exp(torch.clamp(x, min=eps))
            x = x.squeeze()
            self.weights[indices] *= x

def extract_data_targets(subset, dataset):
    return [dataset.data[idx] for idx in subset.indices], [dataset.targets[idx] for idx in subset.indices]
    # return [Image.open(dataset._samples[idx][0]).convert("RGB") for idx in subset.indices], [dataset._samples[idx][1] for idx in subset.indices]