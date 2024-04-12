import torch
import datasets
import copy
from torch.nn.utils import prune
from datasets.datasets import *
from models.baseline.resnet8 import ResidualLayer
from models.baseline.mobilenet import conv_bn, conv_1x1_bn, InvertedResidual, MobileNetV2
from models.baseline.mobilenetv1 import ConvBN, ConvDW, MobileNetV1
import numpy as np

def evaluate(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: str):
    model = model.to(device)
    criterion = criterion.to(device)
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for sample, target in data_loader:
            sample, target = sample.to(device), target.to(device)
            output = model(sample)
            if criterion.__class__.__name__ == 'JointLoss':
                loss = criterion(sample, output, target)
            else:
                loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy += 100. * correct / len(data_loader.dataset)
            sample.detach(), target.detach(), output.detach(), pred.detach()
            del sample, target, output, pred
    return accuracy, loss

def evaluate_weak_learner(model: torch.nn.Module, ensemble: torch.nn.Module, k: int, data_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: str):
    model = model.to(device)
    ensemble = ensemble.to(device)
    criterion = criterion.to(device)
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for sample, target in data_loader:
            sample, target = sample.to(device), target.to(device)
            if ensemble.get_residuals(sample) is None:
                residual = None
            else:
                residual = ensemble.get_residuals(sample)[k - 1]
            output = model([sample, residual])
            if criterion.__class__.__name__ == 'JointLoss':
                loss = criterion(sample, output, target)
            else:
                loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy += 100. * correct / len(data_loader.dataset)
            sample.detach(), target.detach(), output.detach(), pred.detach()
            if residual is not None:
                residual.detach()
            del sample, target, output, pred, residual
    return accuracy, loss

def evaluate_meta_learner(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, learners: list, criterion: torch.nn.Module, device: str):
    model = model.to(device)
    criterion = criterion.to(device)
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for sample, target in data_loader:
            sample, target = sample.to(device), target.to(device)
            learner_outputs = []
            for i, learner in enumerate(learners):
                representation = learner.representation(sample)
                representation = representation.view(representation.size(0), -1)
                learner_outputs.append(learner.classifier[:-1](representation))
            learner_output = torch.stack(learner_outputs, dim=1)
            output = model(learner_output)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy += 100. * correct / len(data_loader.dataset)
            sample.detach(), target.detach(), output.detach(), pred.detach()
            del sample, target, output, pred
    return accuracy, loss

def evaluate_meta_learner_new(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, learners: list, criterion: torch.nn.Module, device: str):
    model = model.to(device)
    criterion = criterion.to(device)
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for sample, target in data_loader:
            sample, target = sample.to(device), target.to(device)
            output = model(sample)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy += 100. * correct / len(data_loader.dataset)
            sample.detach(), target.detach(), output.detach(), pred.detach()
            del sample, target, output, pred
    return accuracy, loss

def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, 
          epochs: int, criterion, optimizer, scheduler, device: str, weighted_dataset=False, weighted_train=False) -> (float, torch.nn.Module):
    if device == 'cuda':
        torch.cuda.empty_cache()
    min_lr = 1e-8
    best_accuracy = 0
    best_model = None
    model = model.to(device)
    criterion = criterion.to(device)
    model.train()
    for epoch in torch.arange(epochs):
        if weighted_dataset:
            for batch_idx, (sample, target, weight) in enumerate(train_loader):
                sample, target, weight = sample.to(device), target.to(device), weight.to(device)
                optimizer.zero_grad()
                output = model(sample)
                losses = criterion(output, target)
                weighted_losses = losses * weight
                loss = weighted_losses.mean()
                loss.backward()
                optimizer.step()
                sample.detach(), target.detach(), output.detach(), weight.detach()
                # del sample, target, output, weight
        else:
            for sample, target in train_loader:
                sample, target = sample.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(sample)
                if criterion.__class__.__name__ == 'JointLoss':
                    loss = criterion(sample, output, target)
                else:
                    loss = criterion(output, target)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                sample.detach(), target.detach(), output.detach()
                # del sample, target, output
        val_accuracy, val_losses = evaluate(model, val_loader, criterion, device)
        val_loss = val_losses.mean()
        if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            scheduler.step(metrics=val_loss)
        else:
            scheduler.step()
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = copy.deepcopy(model)
        print("Epoch {0}: Accuracy={1:.1f}%, Loss={2:.6f}".format(epoch, val_accuracy, val_loss))
        if scheduler._last_lr[-1] < min_lr:
            break
    del criterion, model
    return best_accuracy, best_model

def retrain(model: torch.nn.Module, ensemble, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, 
          epochs: int, criterion, optimizer, scheduler, device: str, weighted_dataset=False, weighted_train=False) -> (float, torch.nn.Module):
    if device == 'cuda':
        torch.cuda.empty_cache()
    min_lr = 1e-8
    best_accuracy = 0
    best_model = None
    model = model.to(device)
    criterion = criterion.to(device)
    model.train()
    for epoch in torch.arange(epochs):
        if weighted_dataset:
            for batch_idx, (sample, target, weight) in enumerate(train_loader):
                sample, target, weight = sample.to(device), target.to(device), weight.to(device)
                optimizer.zero_grad()
                output = model(sample)
                losses = criterion(output, target)
                weighted_losses = losses * weight
                loss = weighted_losses.mean()
                loss.backward()
                optimizer.step()
                sample.detach(), target.detach(), output.detach(), weight.detach()
                # del sample, target, output, weight
        else:
            for sample, target in train_loader:
                sample, target = sample.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(sample)
                if criterion.__class__.__name__ == 'JointLoss':
                    loss = criterion(sample, output, target)
                else:
                    loss = criterion(output, target)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                sample.detach(), target.detach(), output.detach()
                # del sample, target, output
        val_accuracy, val_losses = evaluate(model, val_loader, criterion, device)
        val_loss = val_losses.mean()
        if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            scheduler.step(metrics=val_loss)
        else:
            scheduler.step()
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = copy.deepcopy(model)
        print("Epoch {0}: Accuracy={1:.1f}%, Loss={2:.6f}".format(epoch, val_accuracy, val_loss))
        ensemble_accuracy, ensemble_loss = evaluate(ensemble, val_loader, criterion, device)
        print("Ensemble accuracy: {0:.1f}%".format(ensemble_accuracy))
        if scheduler._last_lr[-1] < min_lr:
            break
    del criterion, model
    return best_accuracy, best_model

def train_sgd_single_step(model, ensemble, train_loader, valid_loader, test_loader, epochs, criterion, optimizer, device):
    model = model.to(device)
    criterion = criterion.to(device)
    model.train()
    best_accuracy = 0
    best_model = None

    for epoch in range(epochs):
        for i, (sample, target) in enumerate(train_loader):
            sample, target = sample.to(device), target.to(device)
            break 

        optimizer.zero_grad()
        output = model(sample)
        loss = criterion(output, target)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        sample.detach(), target.detach(), output.detach()
        del sample, target, output

        learner_accuracy, _ = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch}: Accuracy={learner_accuracy:.1f}%")
        accuracy, _ = evaluate(ensemble, test_loader, criterion, device)
        print(f"Epoch {epoch}: Ensemble Accuracy={accuracy:.1f}%")

        if learner_accuracy > best_accuracy:
            best_accuracy = learner_accuracy
            best_model = copy.deepcopy(model)

    return best_model, best_accuracy



def train_metalearner(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, learners: list,
          epochs: int, criterion, optimizer, scheduler, device: str) -> (float, torch.nn.Module):
    if device == 'cuda':
        torch.cuda.empty_cache()
    min_lr = 1e-5
    best_accuracy = 0
    best_model = None
    model = model.to(device)
    criterion = criterion.to(device)
    model.train()
    for learner in learners:
        learner = learner.to(device)
        learner.eval()
    for epoch in torch.arange(epochs):
        for sample, target in train_loader:
            sample, target = sample.to(device), target.to(device)
            learner_outputs = []
            for learner in learners:
                representation = learner.representation(sample)
                representation = representation.view(representation.size(0), -1)
                learner_outputs.append(learner.classifier[:-1](representation))
            learner_output = torch.stack(learner_outputs, dim=1)
            optimizer.zero_grad()
            output = model(learner_output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            sample.detach(), target.detach(), output.detach()
        val_accuracy, val_losses = evaluate_meta_learner(model, val_loader, learners, criterion, device)
        val_loss = val_losses.mean()
        scheduler.step(metrics=val_loss)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = copy.deepcopy(model)
        print("Epoch {0}: Accuracy={1:.1f}%, Loss={2:.6f}".format(epoch, val_accuracy, val_loss))
        if scheduler._last_lr[-1] < min_lr:
            break
    del criterion, model
    return best_accuracy, best_model

def train_metalearner_new(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, learners: list,
          epochs: int, criterion, optimizer, scheduler, device: str) -> (float, torch.nn.Module):
    if device == 'cuda':
        torch.cuda.empty_cache()
    min_lr = 1e-5
    best_accuracy = 0
    best_model = None
    model = model.to(device)
    criterion = criterion.to(device)
    model.train()
    for epoch in torch.arange(epochs):
        for sample, target in train_loader:
            sample, target = sample.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(sample)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            sample.detach(), target.detach(), output.detach()
        val_accuracy, val_losses = evaluate_meta_learner_new(model, val_loader, learners, criterion, device)
        val_loss = val_losses.mean()
        scheduler.step(metrics=val_loss)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = copy.deepcopy(model)
        print("Epoch {0}: Accuracy={1:.1f}%, Loss={2:.6f}".format(epoch, val_accuracy, val_loss))
        if scheduler._last_lr[-1] < min_lr:
            break
    del criterion, model
    return best_accuracy, best_model

def update_dataset_weights(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, criterion, device: str, weighted_dataset=True, weighted_train=True) -> (float, torch.nn.Module):
    if device == 'cuda':
        torch.cuda.empty_cache()
    model = model.to(device)
    criterion = criterion.to(device)
    model.eval()
    if weighted_dataset:
        for batch_idx, (sample, target, weight) in enumerate(train_loader):
            sample, target, weight = sample.to(device), target.to(device), weight.to(device)
            output = model(sample)
            if weighted_train:
                update_dataset_weights_batch(train_loader, batch_idx, output, target)
            sample.detach(), target.detach(), output.detach(), weight.detach()
            del sample, target, output, weight

def train_weak_learner(model: torch.nn.Module, ensemble: torch.nn.Module, alpha, k: int, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, 
          epochs: int, criterion, optimizer, scheduler, device: str, weighted_dataset=False, weighted_train=False) -> (float, torch.nn.Module):
    if device == 'cuda':
        torch.cuda.empty_cache()
    min_lr = 1e-5
    best_accuracy = 0
    best_model = None
    model = model.to(device)
    ensemble = ensemble.to(device)
    criterion = criterion.to(device)
    alpha = alpha.to(device)
    model.train()
    for epoch in torch.arange(epochs):
        if weighted_dataset:
            for batch_idx, (sample, target, weight) in enumerate(train_loader):
                sample, target, weight = sample.to(device), target.to(device), weight.to(device)
                ensemble_prediction = ensemble(sample)
                ensemble_loss = criterion(ensemble_prediction, target)
                grad = torch.autograd.grad(ensemble_loss, ensemble_prediction, create_graph=True)[0]
                hess = torch.autograd.grad(grad, ensemble_prediction, grad_outputs=torch.ones_like(grad), create_graph=True)[0]
                target_tilde = -grad / hess
                optimizer.zero_grad()
                residual = ensemble.get_last_residual(sample)
                output = model([sample, residual])
                mse_criterion = torch.nn.MSELoss(reduction='none')
                losses = mse_criterion(output * alpha, target_tilde)
                weighted_losses = losses * hess
                loss = weighted_losses.mean()
                loss.backward()
                optimizer.step()
                if weighted_train:
                    update_dataset_weights(train_loader, batch_idx, output, target)
                sample.detach(), target.detach(), output.detach(), weight.detach(), alpha.detach()
                if residual is not None:
                    residual.detach()       
                del sample, target, output, weight, residual, alpha
        else:
            for sample, target in train_loader:
                sample, target = sample.to(device), target.to(device)
                ensemble_prediction = ensemble(sample)
                ensemble_loss = criterion(ensemble_prediction, target)
                grad = torch.autograd.grad(ensemble_loss, ensemble_prediction, create_graph=True)[0]
                hess = torch.autograd.grad(grad, ensemble_prediction, grad_outputs=torch.ones_like(grad), create_graph=True)[0]
                target_tilde = -grad / hess
                optimizer.zero_grad()
                residual = ensemble.get_last_residual(sample)
                output = model([sample, residual])
                mse_criterion = torch.nn.MSELoss()
                loss = mse_criterion(output * alpha, target_tilde)
                loss.backward()
                optimizer.step()
                sample.detach(), target.detach(), output.detach(), alpha.detach()
                if residual is not None:
                    residual.detach()
                del sample, target, output, residual, alpha
        val_accuracy, val_losses = evaluate_weak_learner(model, ensemble, k, val_loader, criterion, device)
        val_loss = val_losses.mean()
        scheduler.step(metrics=val_loss)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = copy.deepcopy(model)
        print("Epoch {0}: Accuracy={1:.1f}%, Loss={2:.6f}".format(epoch, val_accuracy, val_loss))
        if scheduler._last_lr[-1] < min_lr:
            break
    del criterion, model
    return best_accuracy, best_model

def initialize(model: torch.nn.Module, weights: str = None) -> torch.nn.Module:
    if isinstance(model, MobileNetV2):
        return model
    if weights is None:
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0)
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0)
            if isinstance(module, ResidualLayer):
                for sublayer in module.mainpath:
                    if isinstance(sublayer, torch.nn.Conv2d) or isinstance(sublayer, torch.nn.Linear):
                        torch.nn.init.kaiming_normal_(sublayer.weight)
                        if hasattr(module, 'bias') and module.bias is not None and module.bias:
                            sublayer.bias.data.fill_(0)
                for sublayer in module.shortcut:
                    if isinstance(sublayer, torch.nn.Conv2d) or isinstance(sublayer, torch.nn.Linear):
                        torch.nn.init.kaiming_normal_(sublayer.weight)
                        if hasattr(module, 'bias') and module.bias is not None and module.bias:
                            sublayer.bias.data.fill_(0)
    else:
        model.load_state_dict(weights)
    return model

def initialize_linear(model: torch.nn.Module, weights: str = None) -> torch.nn.Module:
    if isinstance(model, MobileNetV2):
        return model
    if weights is None:
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)
                module.bias.data.fill_(0)
    else:
        model.load_state_dict(weights)
    return model

def freeze_conv(model: torch.nn.Module, weights: str = None) -> torch.nn.Module:
    if isinstance(model, MobileNetV2):
        return model
    if weights is None:
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                module.weight.requires_grad = False
                module.bias.requires_grad = False
    else:
        model.load_state_dict(weights)
    return model

def get_device() -> str:
    torch.backends.cudnn.benchmark = True
    print("cudnn backends:", torch.backends.cudnn.version())
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.empty_cache()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device: {}".format(device))
    return device

def get_dataset(ds: str='mnist', batch_size: int=128, train_ratio: float=0.8, weighted: bool=False, shuffle: bool=True) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, int):
    if ds == 'mnist':
        input_channels = 1
        out_classes = 10
        input_shape = (1, 28, 28)
    elif ds == 'fmnist':
        input_channels = 1
        out_classes = 10
        input_shape = (1, 28, 28)
    elif ds == 'cifar10':
        input_channels = 3
        out_classes = 10
        input_shape = (3, 32, 32)
    elif ds == 'gsc':
        input_channels = 1
        out_classes = 10
        input_shape = (1, 41, 64)
    elif ds == 'gtsrb':
        input_channels = 3
        out_classes = 43
        input_shape = (3, 32, 32)
    elif ds == 'har':
        input_channels = 1
        out_classes = 7
        input_shape = (1, 128, 9)
    elif ds == 'vww':
        input_channels = 3
        out_classes = 2
        input_shape = (3, 96, 96)
    
    dataset = datasets.__dict__[ds]
    test_loader = dataset.load_test_data(batch_size=batch_size)
    train_loader, valid_loader = dataset.load_train_val_data(batch_size=batch_size, train_val_split=train_ratio, weighted=weighted, shuffle=shuffle)

    return train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes

def update_dataset_weights_batch(dataloader, batch_index, y_hat, y, eps=1e-9):
    indices = torch.arange(batch_index*dataloader.batch_size, min((batch_index+1)*dataloader.batch_size, dataloader.dataset.__len__()))
    dataloader.dataset.update_weights(indices, y_hat, y, eps=eps)
    dataloader.dataset.weights = dataloader.dataset.weights / dataloader.dataset.weights.max()
