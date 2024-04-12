import torch
from utils.utils import *
from utils.prune import *
from models.baseline.alexnet import AlexNet
from models.baseline.lenet import LeNet
from models.baseline.resnet8 import ResNet8
from models.baseline.mobilenet import MobileNetV2
from models.baseline.mobilenetv1 import MobileNetV1
from models.baseline.dscnn import DSCNN
from models.baseline.fcnet import FCNN
from models.baseline.sqnet import SqueezeNet
from utils.scheduler import CosineAnnealingWarmRestartsWithDecay

def overproduce(pool, model_name, prune_size, dataset, device):
    models = {'alexnet': AlexNet, 'lenet': LeNet, 'resnet8': ResNet8, 'mobilenetv2': MobileNetV2, 'mobilenetv1': MobileNetV1, 'dscnn': DSCNN, 'fcnn': FCNN, 'sqnet': SqueezeNet}
    model_class = models[model_name]
    train_ratio = 0.8
    batch_size = 512
    prune_iter = 1
    prune_step = 2 
    train_iter = 1
    pool_size = 10
    train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes = get_dataset(ds=dataset, batch_size=batch_size, train_ratio=train_ratio, weighted=True, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for i in range(pool_size):
        get_baseModel(model_class, model_name, i, dataset, train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes, criterion, device, train_iter)
        prune_baseModel(pool, model_class, model_name, i, dataset, train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes, criterion, device, prune_size, prune_iter, prune_step)
    torch.save(pool, './checkpoint/pool_{}_{}_{}_adaboost'.format(model_name, dataset, prune_size))
    return pool

def get_baseModel(model_class, model_name, i, dataset, train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes, criterion, device, train_iter=2):
    epochs = 400
    best_accuracy = 0
    best_model = None
    for j in range(train_iter):
        print("Training model {0} iteration {1}".format(i, j))
        model = model_class(input_channels=input_channels, out_classes=out_classes, size=1)
        model = initialize(model, weights=None)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = CosineAnnealingWarmRestartsWithDecay(optimizer, T_0=32, T_mult=2, eta_min=1e-8, last_epoch=-1, verbose=True, decay_factor=0.8)
        accuracy, model = train(model, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device, weighted_dataset=True, weighted_train=False)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)
        print("{0} accuracy: {1:.1f}%".format(model_name, accuracy))
    print("{0} accuracy: {1:.1f}%".format(model_name, best_accuracy))
    torch.save(best_model.state_dict(), './checkpoint/{}_{}_{}_adaboost.pt'.format(model_name, dataset, i))

def prune_baseModel(pool, model_class, model_name, i, dataset, train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes, criterion, device, prune_size, prune_iter=4, prune_step=4):
    best_accuracy = 0
    best_model = None
    for j in torch.arange(prune_iter):
        model = model_class(input_channels=input_channels, out_classes=out_classes, size=1)
        model.load_state_dict(torch.load('./checkpoint/{}_{}_{}_adaboost.pt'.format(model_name, dataset, i)))
        accuracy, _ = evaluate(model, test_loader, criterion, device)
        print("{0} accuracy: {1:.1f}%".format(model_name, accuracy))
        print("Pruning learner {0} iteration {1}".format(i, j))
        structure = model.structures[1]
        target_structure = model.structures[prune_size]
        prev_structure = structure
        prev_model = model
        for k in range(1, prune_step+1):
            print('Pruning learner {0} iteration {1} Iterative pruning {2}'.format(i, j, k))
            current_structure = [int((target_structure[i]-structure[i])*k/prune_step+structure[i]) for i in torch.arange(len(structure))]
            print('current structure: {0}'.format(current_structure))
            pruned_model = model_class(input_channels=input_channels, out_classes=out_classes, size=1)
            pruned_model.set_structure(structure=current_structure)
            optimizer = torch.optim.Adam(pruned_model.parameters(), lr=1e-3, weight_decay=1e-5)
            if k == prune_step:
                epochs = 800
                scheduler = CosineAnnealingWarmRestartsWithDecay(optimizer, T_0=32, T_mult=2, eta_min=1e-8, last_epoch=-1, verbose=True, decay_factor=0.8)
            else:
                epochs = 400
                scheduler = CosineAnnealingWarmRestartsWithDecay(optimizer, T_0=32, T_mult=2, eta_min=1e-8, last_epoch=-1, verbose=True, decay_factor=0.8)
            if device == 'cuda':
                torch.cuda.empty_cache()
            pruned_model = prune_model(prev_model, pruned_model, prev_structure, current_structure)
            pruned_accuracy, pruned_model = train(pruned_model, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device, weighted_dataset=True, weighted_train=True)
            prev_model = copy.deepcopy(pruned_model)
            prev_structure = current_structure
        if pruned_accuracy > best_accuracy:
            best_accuracy = pruned_accuracy
            best_model = copy.deepcopy(pruned_model)
    test_accuracy, test_loss = evaluate(best_model, test_loader, criterion, device)
    print("{0} pruned accuracy: {1:.1f}%".format(model_name, test_accuracy))
    torch.save(best_model, './checkpoint/{}_{}_{}_adaboost_pruned.pt'.format(model_name, dataset, i))
    update_dataset_weights(best_model, train_loader, criterion, device)
    error_rate = torch.tensor(1. - best_accuracy / 100.)
    pool.add_weak_learner(best_model, alpha=torch.log((1-error_rate)/error_rate)/2)

def expand(ensemble, pool, valid_loader, test_loader, criterion, device):
    if len(pool.learners) == 0:
        print("No more learners in the pool")
        return ensemble, pool
    best_accuracy = 0
    best_learner = None
    best_alpha = 0
    for i, learner in enumerate(pool.learners):
        print("\rExpanding: {0}/{1}".format(i+1, len(pool.learners)), end="")
        learner.eval()
        learner.to(device)
        learner_alpha = pool.alphas[i]
        temp_ensemble = copy.deepcopy(ensemble).to(device)
        temp_ensemble.add_weak_learner(learner, learner_alpha)
        temp_ensemble.eval()
        ensemble_accuracy, _ = evaluate(temp_ensemble, valid_loader, criterion, device)
        if ensemble_accuracy > best_accuracy:
            best_accuracy = ensemble_accuracy
            best_learner = learner
            best_alpha = learner_alpha
    ensemble.add_weak_learner(copy.deepcopy(best_learner), best_alpha)
    pool.remove_learner(best_learner)
    test_accuracy, _ = evaluate(ensemble, test_loader, criterion, device)
    print("Expand: Ensemble of {0} at {1:.1f}%".format(len(ensemble.learners), test_accuracy))
    return ensemble, pool

def replace(ensemble, pool, valid_loader, test_loader, criterion, device):
    if len(pool.learners) == 0:
        print("No more learners in the pool")
        return ensemble, pool
    best_accuracy = 0
    best_learner = None
    for ensemble_learner in ensemble.learners:
        ensemble_learner.eval()
        ensemble_learner.to(device)
    for i, ensemble_learner in enumerate(ensemble.learners):
        print("\rReplacing: {0}/{1}".format(i+1, len(ensemble.learners)), end="")
        best_accuracy, _ = evaluate(ensemble, valid_loader, criterion, device)
        best_learner = ensemble_learner
        best_alpha = ensemble.alphas[i]      
        if len(pool.learners) == 0:
            print("No more learners in the pool")
            return ensemble, pool
        for j, pool_learner in enumerate(pool.learners):
            temp_ensemble = copy.deepcopy(ensemble).to(device)
            pool_learner.eval()
            pool_learner.to(device)
            pool_learner_alpha = pool.alphas[j]
            temp_ensemble.learners[i] = pool_learner
            temp_ensemble.alphas[i] = pool_learner_alpha
            ensemble_accuracy, _ = evaluate(temp_ensemble, valid_loader, criterion, device)
            if ensemble_accuracy > best_accuracy:
                best_accuracy = ensemble_accuracy
                best_learner = pool_learner
                best_alpha = pool_learner_alpha
        if best_learner != ensemble_learner:
            ensemble.learners[i] = copy.deepcopy(best_learner)
            ensemble.alphas[i] = best_alpha
            pool.remove_learner(best_learner)
            pool.add_weak_learner(copy.deepcopy(ensemble_learner), ensemble.alphas[i])
        test_accuracy, _ = evaluate(ensemble, test_loader, criterion, device)
        print("Replace: Ensemble of {0} at {1:.1f}%".format(len(ensemble.learners), test_accuracy))
    return ensemble, pool

def backfitting(ensemble, pool, prune_size, valid_loader, test_loader, criterion, device):
    max_iteration = 16
    iteration = 0
    with torch.no_grad():
        while len(pool.learners) > 0 and iteration < max_iteration:
            if len(ensemble.learners) < prune_size:
                print("Ensemble size: {0}".format(len(ensemble.learners)))
                print("Pool size: {0}".format(len(pool.learners)))
                print("Expanding ensemble")
                ensemble, pool = expand(ensemble, pool, valid_loader, test_loader, criterion, device)
            if len(ensemble.learners) > 2:
                print("Ensemble size: {0}".format(len(ensemble.learners)))
                print("Pool size: {0}".format(len(pool.learners)))
                print("Replacing ensemble")
                ensemble, pool = replace(ensemble, pool, valid_loader, test_loader, criterion, device)
            iteration += 1
        if len(ensemble.learners) == prune_size:
            print("Ensemble is full")
            return ensemble, pool
        if len(pool.learners) == 0:
            print("Pool is empty")
            return ensemble, pool
    print("iteration reached maximum")
    return ensemble, pool


def random_ensemble(ensemble, model_name, prune_size, dataset, device):
    models = {'alexnet': AlexNet, 'lenet': LeNet, 'resnet8': ResNet8, 'mobilenetv2': MobileNetV2, 'mobilenetv1': MobileNetV1, 'dscnn': DSCNN, 'fcnn': FCNN, 'sqnet': SqueezeNet}
    model_class = models[model_name]
    train_ratio = 0.8
    batch_size = 512
    prune_iter = 1
    prune_step = 1
    train_iter = 1
    pool_size = 4
    train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes = get_dataset(ds=dataset, batch_size=batch_size, train_ratio=train_ratio, weighted=True, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for i in range(pool_size):
        get_random_baseModel(model_class, model_name, i, dataset, train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes, criterion, device, train_iter)
        random_prune_baseModel(ensemble, model_class, model_name, i, dataset, train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes, criterion, device, prune_size, prune_iter, prune_step)
    torch.save(ensemble, './checkpoint/pool_{}_{}_{}_random_ensemble'.format(model_name, dataset, prune_size))
    return ensemble

def get_random_baseModel(model_class, model_name, i, dataset, train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes, criterion, device, train_iter=2):
    epochs = 450
    best_accuracy = 0
    best_model = None
    for j in range(train_iter):
        print("Training model {0} iteration {1}".format(i, j))
        model = model_class(input_channels=input_channels, out_classes=out_classes, size=1)
        model = initialize(model, weights=None)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = CosineAnnealingWarmRestartsWithDecay(optimizer, T_0=32, T_mult=2, eta_min=1e-8, last_epoch=-1, verbose=True, decay_factor=0.8)
        accuracy, model = train(model, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device, weighted_dataset=True, weighted_train=False)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)
        print("{0} accuracy: {1:.1f}%".format(model_name, accuracy))
    print("{0} accuracy: {1:.1f}%".format(model_name, best_accuracy))
    torch.save(best_model.state_dict(), './checkpoint/{}_{}_{}_random_baseline.pt'.format(model_name, dataset, i))

def random_prune_baseModel(ensemble, model_class, model_name, i, dataset, train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes, criterion, device, prune_size, prune_iter=4, prune_step=4):
    best_accuracy = 0
    best_model = None
    for j in torch.arange(prune_iter):
        model = model_class(input_channels=input_channels, out_classes=out_classes, size=1)
        model.load_state_dict(torch.load('./checkpoint/{}_{}_{}_random_baseline.pt'.format(model_name, dataset, i)))
        accuracy, _ = evaluate(model, test_loader, criterion, device)
        print("{0} accuracy: {1:.1f}%".format(model_name, accuracy))
        print("Pruning learner {0} iteration {1}".format(i, j))
        structure = model.structures[1]
        target_structure = model.structures[prune_size]
        prev_structure = structure
        prev_model = model
        for k in range(1, prune_step+1):
            print('Pruning learner {0} iteration {1} Iterative pruning {2}'.format(i, j, k))
            current_structure = [int((target_structure[i]-structure[i])*k/prune_step+structure[i]) for i in torch.arange(len(structure))]
            print('current structure: {0}'.format(current_structure))
            pruned_model = model_class(input_channels=input_channels, out_classes=out_classes, size=1)
            pruned_model.set_structure(structure=current_structure)
            optimizer = torch.optim.Adam(pruned_model.parameters(), lr=1e-3, weight_decay=1e-5)
            if k == prune_step:
                epochs = 450
                scheduler = CosineAnnealingWarmRestartsWithDecay(optimizer, T_0=32, T_mult=2, eta_min=1e-8, last_epoch=-1, verbose=True, decay_factor=0.8)
            else:
                epochs = 450
                scheduler = CosineAnnealingWarmRestartsWithDecay(optimizer, T_0=32, T_mult=2, eta_min=1e-8, last_epoch=-1, verbose=True, decay_factor=0.8)
            if device == 'cuda':
                torch.cuda.empty_cache()
            pruned_model = prune_random(prev_model, pruned_model, prev_structure, current_structure)
            pruned_accuracy, pruned_model = train(pruned_model, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device, weighted_dataset=True, weighted_train=True)
            prev_model = copy.deepcopy(pruned_model)
            prev_structure = current_structure
        if pruned_accuracy > best_accuracy:
            best_accuracy = pruned_accuracy
            best_model = copy.deepcopy(pruned_model)
    test_accuracy, test_loss = evaluate(best_model, test_loader, criterion, device)
    print("{0} pruned accuracy: {1:.1f}%".format(model_name, test_accuracy))
    torch.save(best_model, './checkpoint/{}_{}_{}_random_baseline_pruned.pt'.format(model_name, dataset, i))
    ensemble.add_weak_learner(best_model, alpha=1.0)