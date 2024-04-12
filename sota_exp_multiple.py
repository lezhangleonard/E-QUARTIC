import torch
from utils.utils import *
from utils.prune import *
from utils.subset import *
from sota.resnet8_multi import ResNet8
from functions.fusions import *
from functions.losses import *
from torch.nn.utils import prune
from utils.eval import *
from utils.scheduler import CosineAnnealingWarmRestartsWithDecay

dataset = 'cifar10'
model_name = 'resnet8'
batch_size = 512
train_ratio = 0.8
learning_rate = 0.001
device = get_device()


models = {'resnet8': ResNet8}
selectedModel = models[model_name]
train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes = get_dataset(ds=dataset, batch_size=batch_size, train_ratio=train_ratio, weighted=False)

# define baseline model
############################################
model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=1)
model = initialize(model, weights=None)
# ############################################

# # get MACs/memory profile
# # ############################################
structures = model.structures
for i in structures:
    print("Number of learners: {0}".format(i))
    model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=i)
    macs = model.get_macs(input_shape)
    print("{0} macs: {1:.2f}M".format(model_name, macs/1e6))
    weight_size = model.get_weight_size()
    print("{0} weight size: {1:.2f}K".format(model_name, weight_size/1e3))
exit()
# ############################################
epochs = 450
criterion = torch.nn.CrossEntropyLoss()
best_model = None
best_accuracy = 0
train_iter = 1
model_size = 2

for i in range(train_iter):
    model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=model_size)
    model = initialize(model, weights=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestartsWithDecay(optimizer, T_0=32, T_mult=2, eta_min=1e-8, last_epoch=-1, verbose=True, decay_factor=0.8)
    accuracy, model = train(model, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = copy.deepcopy(model)
accuracy, _ = evaluate(best_model, test_loader, criterion, device)
print("{0} accuracy: {1:.1f}%".format(model_name, accuracy))
torch.save(best_model, './checkpoint/{}_{}_{}_baseline_multi_models.pt'.format(model_name, dataset, model_size))
