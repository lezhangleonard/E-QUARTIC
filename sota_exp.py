import torch
from utils.utils import *
from utils.prune import *
from utils.subset import *
from sota.resnet8_exit import ResNet8
from sota.dscnn_exit import DSCNN
from sota.mobilenet_exit import MobileNetV1
from sota.fcnn_multi import FCNN
from functions.fusions import *
from functions.losses import *
from torch.nn.utils import prune
from utils.eval import *
from utils.scheduler import CosineAnnealingWarmRestartsWithDecay

def evaluate_exit(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: str):
    model = model.to(device)
    criterion = criterion.to(device)
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for sample, target in data_loader:
            sample, target = sample.to(device), target.to(device)
            output = model(sample)
            # output = output[0] * 0.2 + output[1] * 0.2 + output[2] * 0.3 + output[3] * 0.3
            output = output[-4]
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy += 100. * correct / len(data_loader.dataset)
            sample.detach(), target.detach(), pred.detach(), output.detach()
            del sample, target, output, pred
    return accuracy, loss


def train_exit(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, 
          epochs: int, criterion, optimizer, scheduler, device: str, weighted_dataset=False, weighted_train=False) -> (float, torch.nn.Module):
    if device == 'cuda':
        torch.cuda.empty_cache()
    min_lr = 1e-6
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
            output = output[0] * 0.2 + output[1] * 0.2 + output[2] * 0.3 + output[3] * 0.3
            # output = output[-1]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            sample.detach(), target.detach()
            for out in output:
                out.detach()
            # del sample, target, output
        val_accuracy, val_losses = evaluate_exit(model, val_loader, criterion, device)
        val_loss = val_losses.mean()
        scheduler.step()
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = copy.deepcopy(model)
        print("Epoch {0}: Accuracy={1:.1f}%, Loss={2:.6f}".format(epoch, val_accuracy, val_loss))
        if scheduler._last_lr[-1] < min_lr:
            break
    del criterion, model
    return best_accuracy, best_model

dataset = 'fmnist'
model_name = 'fcnn'
batch_size = 512
train_ratio = 0.8
learning_rate = 0.001
device = get_device()


models = {'resnet8': ResNet8, 'dscnn': DSCNN, 'mobilenet': MobileNetV1, 'fcnn': FCNN}
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
epochs = 150
criterion = torch.nn.CrossEntropyLoss()
best_model = None
best_accuracy = 0
train_iter = 1
model_size = 3

for i in range(train_iter):
    model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=model_size)
    model = initialize(model, weights=None)
    # model.freeze()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestartsWithDecay(optimizer, T_0=32, T_mult=2, eta_min=1e-8, last_epoch=-1, verbose=True, decay_factor=0.8)
    accuracy, model = train(model, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = copy.deepcopy(model)
accuracy, _ = evaluate(best_model, test_loader, criterion, device)
print("{0} accuracy: {1:.1f}%".format(model_name, accuracy))
torch.save(best_model, './checkpoint/{}_{}_{}_baseline_no_exit.pt'.format(model_name, dataset, model_size))


# epochs = 150
# model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=model_size)
# model = torch.load('./checkpoint/{}_{}_{}_baseline_no_exit.pt'.format(model_name, dataset, model_size))
# # model = initialize(model, weights=None)
# model.unfreeze()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# scheduler = CosineAnnealingWarmRestartsWithDecay(optimizer, T_0=32, T_mult=2, eta_min=1e-8, last_epoch=-1, verbose=True, decay_factor=0.8)
# accuracy, model = train_exit(model, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device)
# best_model = copy.deepcopy(model)
# print("{0} accuracy: {1:.1f}%".format(model_name, accuracy))
# torch.save(best_model, './checkpoint/{}_{}_{}_baseline_exit.pt'.format(model_name, dataset, model_size))
# accuracy, _ = evaluate_exit(model, test_loader, criterion, device)
# print("{0} accuracy: {1:.1f}%".format(model_name, accuracy))



# model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=model_size)
# model = torch.load('./checkpoint/{}_{}_{}_baseline_exit.pt'.format(model_name, dataset, model_size))
# accuracy, _ = evaluate_exit(model, test_loader, criterion, device)
# print("{0} accuracy: {1:.1f}%".format(model_name, accuracy))



