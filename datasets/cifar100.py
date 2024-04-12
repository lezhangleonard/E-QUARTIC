import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def load_test_data(batch_size: int, weighted=False) -> torch.utils.data.DataLoader:
    testset = torchvision.datasets.CIFAR100(root='./datasets', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return test_loader

def load_train_val_data(batch_size: int, train_val_split: float, weighted=False) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    dataset = torchvision.datasets.CIFAR100(root='./datasets', train=True, download=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(train_val_split*len(dataset)), len(dataset)-int(train_val_split*len(dataset))])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader