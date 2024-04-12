import torch
from sklearn.metrics import cohen_kappa_score

def get_model_predictions(model, dataloader, device):
    model.eval()
    predictions = []
    model.to(device)
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            inputs.detach(), outputs.detach()
            del inputs, outputs
    return predictions

def calculate_kappa(model1, model2, dataloader, device):
    predictions1 = get_model_predictions(model1, dataloader, device)
    predictions2 = get_model_predictions(model2, dataloader, device)

    return cohen_kappa_score(predictions1, predictions2)

def rank_learners_by_kappa_diversity(ensemble, train_dataloader, device):
    n = len(ensemble.learners)
    kappa_matrix = torch.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            kappa = calculate_kappa(ensemble.learners[i], ensemble.learners[j], train_dataloader, device)
            kappa_matrix[i, j] = kappa
            kappa_matrix[j, i] = kappa

    average_kappa = kappa_matrix.sum(dim=1) / (n - 1)
    print("Average kappa: ", average_kappa)
    ranked_indices = torch.argsort(average_kappa, descending=False)
    return ranked_indices

def evaluate_ranked_subset(ensemble, meta_learner, train_dataloader, test_loader, device):
    torch.use_deterministic_algorithms(True)
    ranking = rank_learners_by_kappa_diversity(ensemble, train_dataloader, device)
    ranking = ranking.tolist()
    ranking.remove(0)
    ranking.insert(0, 0)
    ranking = torch.tensor(ranking)
    print("Ranking: ", ranking)
    accuracies = []
    for k in range(len(ranking)):
        subset = torch.zeros(len(ranking))
        subset[ranking[:k+1]] = 1
        ensemble = ensemble.to(device)
        ensemble.eval()
        accuracy = 0
        with torch.no_grad():
            for sample, target in test_loader:
                sample, target = sample.to(device), target.to(device)
                outputs = []
                for i in range(len(ensemble.learners)):
                    if subset[i] == 1:
                        outputs.append(ensemble.learners[i](sample) * ensemble.alphas[i])
                output = torch.sum(torch.stack(outputs), dim=0)
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                accuracy += 100. * correct / len(test_loader.dataset)
                sample.detach(), target.detach(), output.detach(), pred.detach()
                del sample, target, output, pred
        accuracies.append(accuracy)
    return accuracies

