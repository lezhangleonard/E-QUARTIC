import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset, TensorDataset, Dataset
from torchvision.datasets import ImageFolder
from datasets.datasets import WeightedDataset
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image

class ValidationDataset(Dataset):
    def __init__(self, val_data, val_targets, transform=None):
        self.data = val_data
        self.targets = val_targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            # x = Image.open(x)
            # x = ToPILImage()(x) if x.ndim == 3 else ToPILImage()(x.unsqueeze(0))
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

# Define the transformation sequence
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(96, padding=8),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

def load_test_data(batch_size: int) -> DataLoader:
    testset = ImageFolder(root='./datasets/vww/test', transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return test_loader

def load_train_val_data(batch_size: int, train_val_split: float, weighted: bool=False, shuffle: bool=True) -> (DataLoader, DataLoader):
    dataset = ImageFolder(root='./datasets/vww/train', transform=test_transform)
    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])
    if weighted:
        train_data, train_targets = extract_data_targets(trainset, dataset)
        trainset = WeightedDataset(train_data, train_targets, alpha=2e-3, transform=train_transform)
    else:
        train_data, train_targets = extract_data_targets(trainset, dataset)
        trainset = ValidationDataset(train_data, train_targets, transform=train_transform)
    val_data, val_targets = extract_data_targets(valset, dataset)
    valset = ValidationDataset(val_data, val_targets, transform=test_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader

def extract_data_targets(subset: Subset, dataset: ImageFolder):
    indices = subset.indices
    data = [Image.open(dataset.samples[idx][0]) for idx in indices]
    targets = [dataset.samples[idx][1] for idx in indices]  # Associated labels
    return data, targets