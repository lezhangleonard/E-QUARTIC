import copy
import torch
from torch import nn

def train_learner(model: torch.nn.Module, learner: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, 
          epochs: int, criterion, optimizer, scheduler, device: str) -> (float, torch.nn.Module):
    if device == 'cuda':
        torch.cuda.empty_cache()
    min_lr = 1e-5
    best_accuracy = 0
    best_model = None
    model = model.to(device)
    learner = learner.to(device)
    criterion = criterion.to(device)
    learner.train()
    for epoch in torch.arange(epochs):
        for sample, target in train_loader:
            sample, target = sample.to(device), target.to(device)
            residual = None
            if len(model.learners) > 0:
                residual = model.get_feature(sample)[-1]
            optimizer.zero_grad()
            output = learner([sample, residual])
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            sample.detach(), target.detach(), output.detach()
        val_accuracy, val_losses = evaluate_learner(model, learner, val_loader, criterion, device)
        val_loss = val_losses.mean()
        scheduler.step(metrics=val_loss)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = copy.deepcopy(learner)
        print("Epoch {0}: Accuracy={1:.1f}%, Loss={2:.6f}".format(epoch, val_accuracy, val_loss))
        if scheduler._last_lr[-1] < min_lr:
            break
    del criterion, learner
    return best_accuracy, best_model

def evaluate_learner(model: torch.nn.Module, learner: torch.nn.Module, data_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: str):
    model = model.to(device)
    learner = learner.to(device)
    criterion = criterion.to(device)
    learner.eval()
    accuracy = 0
    with torch.no_grad():
        for sample, target in data_loader:
            sample, target = sample.to(device), target.to(device)
            residual = None
            if len(model.learners) > 0:
                residual = model.get_feature(sample)[-1]
            output = learner([sample, residual])
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy += 100. * correct / len(data_loader.dataset)
            sample.detach(), target.detach(), output.detach(), pred.detach()
            del sample, target, output, pred
    return accuracy, loss

def train_boosted_metalearner(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, ensemble,
          epochs: int, criterion, optimizer, scheduler, device: str) -> (float, torch.nn.Module):
    if device == 'cuda':
        torch.cuda.empty_cache()
    min_lr = 1e-5
    best_accuracy = 0
    best_model = None
    model = model.to(device)
    criterion = criterion.to(device)
    model.train()
    ensemble.eval()
    for epoch in torch.arange(epochs):
        for sample, target in train_loader:
            sample, target = sample.to(device), target.to(device)
            learner_outputs = ensemble.get_feature(sample)
            learner_output = torch.stack(learner_outputs, dim=1)
            learner_output = learner_output.view(learner_output.size(0), -1)
            optimizer.zero_grad()
            output = model(learner_output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            sample.detach(), target.detach(), output.detach()
        val_accuracy, val_losses = evaluate_boosted_metalearner(model, val_loader, ensemble, criterion, device)
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

def evaluate_boosted_metalearner(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, ensemble, criterion: torch.nn.Module, device: str):
    model = model.to(device)
    criterion = criterion.to(device)
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for sample, target in data_loader:
            sample, target = sample.to(device), target.to(device)
            learner_outputs = ensemble.get_feature(sample)
            learner_output = torch.stack(learner_outputs, dim=1)
            learner_output = learner_output.view(learner_output.size(0), -1)
            output = model(learner_output)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy += 100. * correct / len(data_loader.dataset)
            sample.detach(), target.detach(), output.detach(), pred.detach()
            del sample, target, output, pred
    return accuracy, loss