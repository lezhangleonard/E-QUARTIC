import torch
import torch.nn.functional as F
import itertools

def evaluate_subset(ensemble: torch.nn.Module, selected_learners: int, metalearner: torch.nn.Module, data_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: str):
    num_learners = len(ensemble.learners)
    assert selected_learners <= num_learners

    subsets = []
    for indices in itertools.combinations(range(num_learners), selected_learners):
        selection = [0] * num_learners
        for index in indices:
            selection[index] = 1
        subsets.append(selection)

    accuracy = 0
    for subset in subsets:
        ensemble = ensemble.to(device)
        metalearner = metalearner.to(device)
        criterion = criterion.to(device)
        ensemble.eval()
        metalearner.eval()
        with torch.no_grad():
            for sample, target in data_loader:
                sample, target = sample.to(device), target.to(device)
                learner_outputs = []
                for i, learner in enumerate(ensemble.learners):
                    representation = learner.representation(sample)
                    representation = representation.view(representation.size(0), -1)
                    learner_outputs.append(learner.classifier[:-1](representation) * subset[i])
                learner_output = torch.stack(learner_outputs, dim=1)
                output = metalearner(learner_output)
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                accuracy += 100. * correct / len(data_loader.dataset) / len(subsets)
                sample.detach(), target.detach(), output.detach(), pred.detach()
                del sample, target, output, pred
    return accuracy

def evaluate_random_subset_bagging(ensemble: torch.nn.Module, selected_learners: int, data_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: str):
    num_learners = len(ensemble.learners)
    assert selected_learners <= num_learners
    mask = torch.zeros(num_learners)
    mask[:selected_learners] = 1
    mask = mask[torch.randperm(num_learners)]
    ensemble = ensemble.to(device)
    criterion = criterion.to(device)
    ensemble.eval()
    accuracy = 0
    with torch.no_grad():
        for sample, target in data_loader:
            sample, target = sample.to(device), target.to(device)
            learner_outputs = []
            for i, learner in enumerate(ensemble.learners):
                representation = learner.representation(sample)
                representation = representation.view(representation.size(0), -1)
                learner_outputs.append(learner.classifier[:-1](representation) * mask[i])
            learner_output = torch.stack(learner_outputs, dim=1)
            output = learner_output.sum(dim=1)
            output = F.softmax(output, dim=1)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy += 100. * correct / len(data_loader.dataset)
            sample.detach(), target.detach(), output.detach(), pred.detach()
            del sample, target, output, pred
    return accuracy

