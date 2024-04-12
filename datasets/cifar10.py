import torch
import torchvision
import torchvision.transforms as transforms
from datasets.datasets import *
from torchvision.transforms import ToPILImage


class ValidationDataset(Dataset):
    def __init__(self, val_data, val_targets, transform=None):
        self.data = val_data
        self.targets = val_targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = ToPILImage()(x) if x.ndim == 3 else ToPILImage()(x.unsqueeze(0))
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

class DeterministicFlip(object):
    def __init__(self, flip_horizontal=True):
        self.flip_horizontal = flip_horizontal

    def __call__(self, x):
        if self.flip_horizontal:
            flip_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=1.0),  # p=1.0 makes it deterministic
                transforms.ToTensor()
            ])
        else:
            flip_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomVerticalFlip(p=1.0),  # p=1.0 makes it deterministic
                transforms.ToTensor()
            ])
        return flip_transform(x)

class DeterministicRotation:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        x_pil = torchvision.transforms.functional.to_pil_image(x)
        rotated_x_pil = torchvision.transforms.functional.rotate(x_pil, self.angle)
        rotated_x = torchvision.transforms.functional.to_tensor(rotated_x_pil)
        return rotated_x


retrain_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(25),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])


def load_test_data(batch_size: int) -> torch.utils.data.DataLoader:
    testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return test_loader

def load_train_val_data(batch_size: int, train_val_split: float, weighted: bool=False, shuffle: bool=True) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    dataset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True)
    trainset, valset = torch.utils.data.random_split(dataset, [int(train_val_split*len(dataset)), len(dataset)-int(train_val_split*len(dataset))])
    if weighted:
        train_data, train_targets = extract_data_targets(trainset, dataset)
        trainset = WeightedDataset(train_data, train_targets, alpha=1e-5, transform=train_transform)
        val_data, val_targets = extract_data_targets(valset, dataset)
        valset = ValidationDataset(val_data, val_targets, transform=test_transform)
    else:
        train_data, train_targets = extract_data_targets(trainset, dataset)
        trainset = ValidationDataset(train_data, train_targets, transform=train_transform)
        val_data, val_targets = extract_data_targets(valset, dataset)
        valset = ValidationDataset(val_data, val_targets, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def load_test_data_retrain(batch_size: int) -> torch.utils.data.DataLoader:
    testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=retrain_transform)
    test_data, test_targets = testset.data, testset.targets
    indices = torch.randperm(len(test_data))[:1000]
    indices_list = indices.tolist()
    testset = ValidationDataset([test_data[i] for i in indices_list], [test_targets[i] for i in indices_list], transform=retrain_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return test_loader

def load_train_val_data_retrain(batch_size: int, train_val_split: float, weighted: bool=False, shuffle: bool=True) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    dataset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True)
    trainset, valset = torch.utils.data.random_split(dataset, [int(train_val_split*len(dataset)), len(dataset)-int(train_val_split*len(dataset))])
    if weighted:
        train_data, train_targets = extract_data_targets(trainset, dataset)
        trainset = WeightedDataset(train_data, train_targets, alpha=1e-5, transform=retrain_transform)
        val_data, val_targets = extract_data_targets(valset, dataset)
        valset = ValidationDataset(val_data, val_targets, transform=retrain_transform)
    else:
        train_data, train_targets = extract_data_targets(trainset, dataset)
        indices = torch.randperm(len(train_data))
        indices_list = indices.tolist()
        trainset = ValidationDataset([train_data[i] for i in indices_list], [train_targets[i] for i in indices_list], transform=retrain_transform)

        val_data, val_targets = extract_data_targets(valset, dataset)
        indices = torch.randperm(len(val_data))[:1000]
        indices_list = indices.tolist()
        valset = ValidationDataset([val_data[i] for i in indices_list], [val_targets[i] for i in indices_list], transform=retrain_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader