import torch

def evaluate_random_subset(ensemble: torch.nn.Module, selected_learners: int, metalearner: torch.nn.Module, data_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: str):
    num_learners = len(ensemble.learners)
    assert selected_learners <= num_learners
    mask = torch.zeros(num_learners)
    mask[:selected_learners] = 1
    mask = mask[torch.randperm(num_learners)]
    ensemble = ensemble.to(device)
    metalearner = metalearner.to(device)
    criterion = criterion.to(device)
    ensemble.eval()
    metalearner.eval()
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
            output = metalearner(learner_output)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy += 100. * correct / len(data_loader.dataset)
            sample.detach(), target.detach(), output.detach(), pred.detach()
            del sample, target, output, pred
    return accuracy, loss

