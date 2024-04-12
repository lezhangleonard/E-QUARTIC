import torch
from utils.utils import *
from utils.prune import *
from utils.subset import *
from models.baseline.alexnet import AlexNet
from models.baseline.lenet import LeNet, LeNet128x9
from models.baseline.resnet8 import ResNet8
from models.baseline.mobilenet import MobileNetV2
from models.meta.ensemble import Ensemble
from models.baseline.mobilenetv1 import MobileNetV1
from models.baseline.dscnn import DSCNN
from models.baseline.fcnet import FCNN
from models.baseline.sqnet import SqueezeNet
from functions.fusions import *
from functions.losses import *
from models.meta.metalearner import *
from torch.nn.utils import prune
from utils.eval import *
from utils.rank import *
from utils.adaboost import *


dataset = 'cifar10'
model_name = 'resnet8'
batch_size = 128
train_ratio = 0.8
learning_rate = 0.001
device = get_device()

models = {'alexnet': AlexNet, 'lenet': LeNet, 'lenet128x9': LeNet128x9, 'resnet8': ResNet8, 'mobilenetv2': MobileNetV2, 'mobilenetv1': MobileNetV1, 'dscnn': DSCNN, 'fcnn': FCNN, 'sqnet': SqueezeNet}
selectedModel = models[model_name]

# get dataloaders
train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes = get_dataset(ds=dataset, batch_size=batch_size, train_ratio=train_ratio, weighted=False)


# define baseline model
############################################
model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=1)
# model = initialize(model, weights=None)
# model.load_state_dict('./checkpoint/{}_{}_{}_adaboost.pt'.format(model_name, dataset, i))
# ############################################

# # get MACs/memory profile
# ############################################
# structures = model.structures
# for i in structures:
#     print("Number of learners: {0}".format(i))
#     model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=i)
#     # model = initialize(model, weights=None)
#     # get macs
#     macs = model.get_macs(input_shape)
#     print("{0} macs: {1:.2f}M".format(model_name, macs/1e6))
#     weight_size = model.get_weight_size()
#     print("{0} weight size: {1:.2f}K".format(model_name, weight_size*i/1e3))
# exit()
############################################

# get MACs/memory profile for MobileNetV2
############################################
# interverted_residual_settings = model.interverted_residual_settings
# for i in interverted_residual_settings:
#     print("Number of learners: {0}".format(i))
#     model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=i)
#     # get macs
#     macs = model.get_macs(input_shape)
#     print("{0} macs: {1:.2f}K".format(model_name, macs*i/1e3))
#     weight_size = model.get_weight_size()
#     print("{0} weight size: {1:.2f}K".format(model_name, weight_size*i/1e3))
############################################

# train baseline model
############################################
# epochs = 64
# criterion = torch.nn.CrossEntropyLoss()
# best_model = None
# best_accuracy = 0
# train_iter = 8
# for i in range(train_iter):
#     model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=1)
#     model = initialize(model, weights=None)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
#     accuracy, model = train(model, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device)
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_model = copy.deepcopy(model)
# accuracy, _ = evaluate(best_model, test_loader, criterion, device)
# print("{0} accuracy: {1:.1f}%".format(model_name, accuracy))
# torch.save(best_model.state_dict(), './checkpoint/{}_{}.pt'.format(model_name, dataset))
# exit()
############################################

# prune model
############################################
# prune_size = 8          # number of learners in ensemble
# prune_iter = 4          # number of iterations for each learner to find the best learner
# prune_step = 4          # number of steps to gradually prune each learner
# # create an ensemble
# ensemble = Ensemble(fusion=unweighted_fusion)
# criterion = torch.nn.CrossEntropyLoss()

# for i in torch.arange(prune_size):
#     # baseline model
#     epochs = 64
#     best_model = None
#     best_accuracy = 0
#     train_iter = 4
#     for j in range(train_iter):
#         print("Training model {0} iteration {1}".format(i, j))
#         model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=1)
#         model = initialize(model, weights=None)
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
#         accuracy, model = train(model, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device)
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_model = copy.deepcopy(model)
#     print("{0} accuracy: {1:.1f}%".format(model_name, accuracy))
#     torch.save(best_model.state_dict(), './checkpoint/{}_{}_{}.pt'.format(model_name, dataset, i))

#     # prune model
#     best_accuracy = 0
#     best_model = None
#     for j in torch.arange(prune_iter):
#         model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=1)
#         model.load_state_dict(torch.load('./checkpoint/{}_{}_{}.pt'.format(model_name, dataset, i)))
#         print("Pruning learner {0} iteration {1}".format(i, j))
#         if isinstance(model, MobileNetV2):
#             structure = []
#             for t, c, n, s in model.interverted_residual_settings[1]:
#                 structure.append(c)
#             target_structure = []
#             for t, c, n, s in model.interverted_residual_settings[prune_size]:
#                 target_structure.append(c)
#         else:
#             structure = model.structures[1]
#             target_structure = model.structures[prune_size]
#         prev_structure = structure
#         prev_model = model
#         for k in range(1, prune_step+1):
#             if k == prune_step: epochs = 64
#             else: epochs = 8
#             print('Pruning learner {0} iteration {1} Iterative pruning {2}'.format(i, j, k))
#             current_structure = [int((target_structure[i]-structure[i])*k/prune_step+structure[i]) for i in torch.arange(len(structure))]
#             print('current structure: {0}'.format(current_structure))
#             pruned_model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=1)
#             pruned_model.set_structure(structure=current_structure)
#             optimizer = torch.optim.Adam(pruned_model.parameters(), lr=1e-3, weight_decay=1e-4)
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)
#             pruned_model = prune_model(prev_model, pruned_model, prev_structure, current_structure)
#             pruned_accuracy, pruned_model = train(pruned_model, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device)
#             test_accuracy, test_loss = evaluate(pruned_model, test_loader, criterion, device)
#             prev_model = copy.deepcopy(pruned_model)
#             prev_structure = current_structure
#         print("{0} pruned accuracy: {1:.1f}%".format(model_name, test_accuracy))
#         # get best model
#         if test_accuracy > best_accuracy:
#             best_accuracy = test_accuracy
#             best_model = copy.deepcopy(pruned_model)

#     ensemble.add_weak_learner(best_model, alpha=1.0)
# # save ensemble
# torch.save(ensemble, './checkpoint/ensemble_{}_{}_{}'.format(model_name, dataset, prune_size))
############################################


### random ensemble
###################################
# criterion = torch.nn.CrossEntropyLoss(reduction='none')

# ensemble = Ensemble(fusion=unweighted_fusion)
# prune_size = 4          # number of learners in ensemble
# ensemble = random_ensemble(ensemble, model_name, prune_size, dataset, device)

# ensemble = torch.load('./checkpoint/pool_{}_{}_{}_random_ensemble'.format(model_name, dataset, prune_size))

# # # evaluate ensemble

# for learner in ensemble.learners:
#     learner_accuracy, learner_loss = evaluate(learner, test_loader, criterion, device)
#     print("Learner accuracy: {0:.1f}%".format(learner_accuracy))
# ensemble_accuracy, ensemble_loss = evaluate(ensemble, test_loader, criterion, device)
# print("Ensemble accuracy: {0:.1f}%".format(ensemble_accuracy))
# exit()
###################################

### Adaboost ensemble
###################################
pool = Ensemble(fusion=weighted_fusion)
prune_size = 8          # number of learners in ensemble
pool = overproduce(pool, model_name, prune_size, dataset, device)
# load pool ensemble
train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes = get_dataset(ds=dataset, batch_size=batch_size, train_ratio=train_ratio, weighted=False)
pool = torch.load('./checkpoint/pool_{}_{}_{}_adaboost'.format(model_name, dataset, prune_size))
ensemble = Ensemble(fusion=weighted_fusion)
criterion = torch.nn.CrossEntropyLoss(reduction='none')
ensemble, pool = backfitting(ensemble, pool, prune_size, valid_loader, test_loader, criterion, device)
torch.save(ensemble, './checkpoint/ensemble_backfitting_{}_{}_{}_adaboost'.format(model_name, dataset, prune_size))
exit()
###################################


# epochs = 64
# prune_size = 4          # number of learners in ensemble
# prune_iter = 8          # number of iterations for each learner to find the best learner
# prune_step = 4          # number of steps to gradually prune each learner
# train_iter = 4          # number of iterations for each learner to find the best learner
# train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes = get_dataset(ds=dataset, batch_size=batch_size, train_ratio=train_ratio, weighted=True)
# # create an ensemble
# ensemble = Ensemble(fusion=weighted_fusion)
# criterion = torch.nn.CrossEntropyLoss(reduction='none')

# for i in torch.arange(prune_size):
#     best_accuracy = 0
#     best_model = None
#     for j in range(train_iter):
#         print("Training model {0} iteration {1}".format(i, j))
#         model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=1)
#         model = initialize(model, weights=None)
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
#         accuracy, model = train(model, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device, weighted_dataset=True, weighted_train=False)
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_model = copy.deepcopy(model)
#     test_accuracy, test_loss = evaluate(best_model, test_loader, criterion, device)
#     print("{0} accuracy: {1:.1f}%".format(model_name, test_accuracy))
#     torch.save(best_model.state_dict(), './checkpoint/{}_{}_{}_adaboost.pt'.format(model_name, dataset, i))
    
#     best_accuracy = 0
#     best_model = None
#     for j in torch.arange(prune_iter):
#         model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=1)
#         model.load_state_dict(torch.load('./checkpoint/{}_{}_{}_adaboost.pt'.format(model_name, dataset, i)))
#         print("Pruning learner {0} iteration {1}".format(i, j))
#         structure = model.structures[1]
#         target_structure = model.structures[prune_size]
#         prev_structure = structure
#         prev_model = model
#         for k in range(1, prune_step+1):
#             if k == prune_step: epochs = 64
#             else: epochs = 16
#             print('Pruning learner {0} iteration {1} Iterative pruning {2}'.format(i, j, k))
#             current_structure = [int((target_structure[i]-structure[i])*k/prune_step+structure[i]) for i in torch.arange(len(structure))]
#             print('current structure: {0}'.format(current_structure))
#             pruned_model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=1)
#             pruned_model.set_structure(structure=current_structure)
#             optimizer = torch.optim.Adam(pruned_model.parameters(), lr=1e-3, weight_decay=1e-4)
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
#             pruned_model = prune_model(prev_model, pruned_model, prev_structure, current_structure)
#             pruned_accuracy, pruned_model = train(pruned_model, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device, weighted_dataset=True, weighted_train=True)
#             prev_model = copy.deepcopy(pruned_model)
#             prev_structure = current_structure
#         valid_accuracy, valid_loss = evaluate(pruned_model, valid_loader, criterion, device)
#         print("{0} pruned accuracy: {1:.1f}%".format(model_name, valid_accuracy))
#         if valid_accuracy > best_accuracy:
#             best_accuracy = valid_accuracy
#             best_model = copy.deepcopy(pruned_model)
#     test_accuracy, _ = evaluate(best_model, test_loader, criterion, device)
#     print("{0} pruned accuracy: {1:.1f}%".format(model_name, test_accuracy))
#     update_dataset_weights(best_model, train_loader, criterion, device)
#     error_rate = torch.tensor(1. - best_accuracy / 100.)
#     ensemble.add_weak_learner(best_model, alpha=torch.log((1-error_rate)/error_rate)/2)

# # save ensemble
# torch.save(ensemble, './checkpoint/ensemble_{}_{}_{}_adaboost'.format(model_name, dataset, prune_size))

# # # evaluate ensemble
# # ############################################
# load ensemble
prune_size = 2          # number of learners in ensemble
criterion = torch.nn.CrossEntropyLoss(reduction='none')
ensemble = torch.load('./checkpoint/ensemble_backfitting_{}_{}_{}_adaboost'.format(model_name, dataset, prune_size))
for learner in ensemble.learners:
    learner_accuracy, learner_loss = evaluate(learner, test_loader, criterion, device)
    print("Learner accuracy: {0:.1f}%".format(learner_accuracy))
ensemble_accuracy, ensemble_loss = evaluate(ensemble, test_loader, criterion, device)
print("Ensemble accuracy: {0:.1f}%".format(ensemble_accuracy))
# # ############################################
# train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes = get_dataset(ds=dataset, batch_size=batch_size, train_ratio=train_ratio, weighted=False)
# metalearner = WeightedMetaLearner(input_channels=prune_size, out_classes=out_classes)
# epochs = 1
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(metalearner.parameters(), lr=1e-3, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, cooldown=5, min_lr=0, eps=1e-8)
# accuracy, metalearner = train_metalearner(metalearner, train_loader, valid_loader, ensemble.learners, epochs, criterion, optimizer, scheduler, device)
# accuracy, loss = evaluate_meta_learner(metalearner, test_loader, ensemble.learners, criterion, device)
# print("Metalearner accuracy: {0:.1f}%".format(accuracy))
# torch.save(metalearner.state_dict(), './checkpoint/metalearner_{}_{}_{}_adaboost.pt'.format(model_name, dataset, prune_size))
# metalearner.load_state_dict(torch.load('./checkpoint/metalearner_{}_{}_{}_adaboost.pt'.format(model_name, dataset, prune_size)))


# bagginglearner = Bagging(input_channels=prune_size, out_classes=out_classes)

# # averaging accuracy
# num_selected = 2
# accuracy = evaluate_subset(ensemble, num_selected, metalearner, test_loader, criterion, device)
# print("Metalearner accuracy: {0:.1f}%".format(accuracy))
# accuracy = evaluate_subset(ensemble, num_selected, bagginglearner, test_loader, criterion, device)
# print("Bagging accuracy: {0:.1f}%".format(accuracy))

train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes = get_dataset(ds=dataset, batch_size=batch_size, train_ratio=train_ratio, weighted=False, shuffle=False)
accuracies = evaluate_ranked_subset(ensemble, ensemble, valid_loader, test_loader, device)
# print all accuracies
for i in range(len(accuracies)):
    print("Ranked subset accuracy with {0} learners: {1:.1f}%".format(i+1, accuracies[i]))

# K = 2
# size = 2
# epochs = 4
# iter_step = 1
# ensemble = GrowNet(fusion=weighted_fusion)
# for k in range(K):
#     best_accuracy = 0
#     best_model = None
#     alpha = torch.nn.Parameter(torch.tensor([1/K])).requires_grad_(True)
#     best_alpha = None  
#     for i in range(iter_step):
#         base_model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=size)
#         weak_learner = WeakLearner(base_model=base_model, residual_size=ensemble.get_last_residual_size())
#         # train weak learner
#         print("Training weak learner {0} iteration {1}".format(k, i))
#         criterion = torch.nn.CrossEntropyLoss()
#         optimizer = torch.optim.Adam([{'params': weak_learner.parameters()}, {'params': alpha}], lr=1e-3, weight_decay=1e-4)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
#         accuracy, weak_learner = train_weak_learner(weak_learner, ensemble, alpha, k, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device)
#         print("Weak learner accuracy: {0:.1f}%".format(accuracy))
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_model = copy.deepcopy(weak_learner)
#             best_alpha = copy.deepcopy(alpha)
#     # add weak learner to ensemble
#     ensemble.add_learner(best_model, best_alpha.item())

# evaluate ensemble
# ############################################
# for k in range(K):
#     learner = ensemble.learners[k]
#     learner_accuracy, learner_loss = evaluate_weak_learner(learner, ensemble, k, test_loader, criterion, device)
#     print("Learner accuracy: {0:.1f}%".format(learner_accuracy))
#     print("Learner weight: {0:.2f}".format(ensemble.alpha[k].item()))
# ensemble_accuracy, ensemble_loss = evaluate(ensemble, test_loader, criterion, device)
# print("Ensemble accuracy: {0:.1f}%".format(ensemble_accuracy))
############################################
