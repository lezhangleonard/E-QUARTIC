import torch
from utils.utils import *
from utils.prune import *
from utils.boost import *
from models.baseline.alexnet import AlexNet
from models.baseline.lenet import LeNet
from models.baseline.resnet8 import ResNet8
from models.baseline.mobilenet import MobileNetV2
from models.meta.ensemble import Ensemble
from functions.fusions import *
from functions.losses import *
from models.meta.metalearner import MetaLearner
from models.meta.grownet import GrowNet, WeakLearner
import copy

dataset = 'cifar10'
model_name = 'lenet'
batch_size = 32
train_ratio = 0.8
learning_rate = 0.001
device = get_device()

models = {'alexnet': AlexNet, 'lenet': LeNet, 'resnet8': ResNet8, 'mobilenetv2': MobileNetV2}
selectedModel = models[model_name]

# get dataloaders
train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes = get_dataset(ds=dataset, batch_size=batch_size, train_ratio=train_ratio, weighted=False)

num_learners = 8
epochs = 64
learner_iter = 4
criterion = nn.CrossEntropyLoss()
ensemble = torch.load('./checkpoint/ensemble_{}_{}_{}'.format(model_name, dataset, num_learners))

grownet = GrowNet()
for i in range(num_learners):
    print("Training learner {0}...".format(i))
    best_accuracy = 0
    best_learner = None
    if i == 0:
        base_model = copy.deepcopy(ensemble.learners[i])
    else:
        base_model = selectedModel(input_channels, out_classes, size=num_learners)
        initialize(base_model)
    for j in range(learner_iter):
        learner = WeakLearner(base_model, first=(i==0))
        optimizer = torch.optim.Adam(learner.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
        accuracy, learner = train_learner(grownet, learner, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_learner = copy.deepcopy(learner)
    accuracy, loss = evaluate_learner(grownet, best_learner, test_loader, criterion, device)
    print("Learner {0}: Accuracy={1:.1f}%, Loss={2:.6f}".format(i, accuracy, loss))
    grownet.add_learner(best_learner)
torch.save(grownet, './checkpoint/grownet_{}_{}_{}'.format(model_name, dataset, num_learners))

grownet = torch.load('./checkpoint/grownet_{}_{}_{}'.format(model_name, dataset, num_learners))
grownet_accuracy, grownet_loss = evaluate(grownet, test_loader, criterion, device)
print("Ensemble accuracy: {0:.1f}%".format(grownet_accuracy))

metalearner = MetaLearner(out_classes * num_learners, out_classes)
optimizer = torch.optim.Adam(metalearner.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
accuracy, metalearner = train_boosted_metalearner(metalearner, train_loader, valid_loader, grownet, epochs, criterion, optimizer, scheduler, device)
accuracy, loss = evaluate_boosted_metalearner(metalearner, test_loader, grownet, criterion, device)
print("MetaLearner accuracy: {0:.1f}%".format(accuracy))

