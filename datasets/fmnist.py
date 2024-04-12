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
            # x = ToPILImage()(x) if x.ndim == 3 else ToPILImage()(x.unsqueeze(0))
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

class ToFloatTensor(object):
    def __call__(self, tensor):
        if not tensor.is_floating_point():
            tensor = tensor.to(torch.float32)
        return tensor

class AddChannelDimension(object):
    def __call__(self, tensor):
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

class DeterministicRotation:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        x_pil = torchvision.transforms.functional.to_pil_image(x)
        rotated_x_pil = torchvision.transforms.functional.rotate(x_pil, self.angle)
        rotated_x = torchvision.transforms.functional.to_tensor(rotated_x_pil)
        return rotated_x

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

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(28, padding=4),
    AddChannelDimension(),
    ToFloatTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

valid_transform = transforms.Compose([
    AddChannelDimension(),
    ToFloatTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    AddChannelDimension(),
    ToFloatTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

retrain_transform = transforms.Compose([
    AddChannelDimension(),
    ToFloatTensor(),
    transforms.RandomRotation(30),
    transforms.Normalize((0.5,), (0.5,)),
])




def load_test_data(batch_size: int) -> torch.utils.data.DataLoader:
    testset = torchvision.datasets.FashionMNIST(root='./datasets', train=False, download=True)
    testset = ValidationDataset(testset.data, testset.targets, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return test_loader

def load_train_val_data(batch_size: int, train_val_split: float, weighted: bool=False, shuffle: bool=True) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    dataset = torchvision.datasets.FashionMNIST(root='./datasets', train=True, download=True)
    trainset, valset = torch.utils.data.random_split(dataset, [int(train_val_split*len(dataset)), len(dataset)-int(train_val_split*len(dataset))])
    if weighted:
        train_data, train_targets = extract_data_targets(trainset, dataset)
        trainset = WeightedDataset(train_data, train_targets, alpha=1e-5, transform=train_transform)
        val_data, val_targets = extract_data_targets(valset, dataset)
        valset = ValidationDataset(val_data, val_targets, transform=valid_transform)
    else:
        train_data, train_targets = extract_data_targets(trainset, dataset)
        trainset = ValidationDataset(train_data, train_targets, transform=train_transform)
        val_data, val_targets = extract_data_targets(valset, dataset)
        valset = ValidationDataset(val_data, val_targets, transform=valid_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def load_test_data_retrain(batch_size: int) -> torch.utils.data.DataLoader:
    testset = torchvision.datasets.FashionMNIST(root='./datasets', train=False, download=True)
    testset = ValidationDataset(testset.data[:100], testset.targets[:100], transform=retrain_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return test_loader

def load_train_val_data_retrain(batch_size: int, train_val_split: float, weighted: bool=False, shuffle: bool=True) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    dataset = torchvision.datasets.FashionMNIST(root='./datasets', train=True, download=True)
    trainset, valset = torch.utils.data.random_split(dataset, [int(train_val_split*len(dataset)), len(dataset)-int(train_val_split*len(dataset))])
    if weighted:
        train_data, train_targets = extract_data_targets(trainset, dataset)
        trainset = WeightedDataset(train_data, train_targets, alpha=1e-5, transform=retrain_transform)
        val_data, val_targets = extract_data_targets(valset, dataset)
        valset = ValidationDataset(val_data, val_targets, transform=retrain_transform)
    else:
        train_data, train_targets = extract_data_targets(trainset, dataset)
        trainset = ValidationDataset(train_data[:1000], train_targets[:1000], transform=retrain_transform)
        val_data, val_targets = extract_data_targets(valset, dataset)
        valset = ValidationDataset(val_data[:100], val_targets[:100], transform=retrain_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader